"""
Entry-point generation function and inner search dispatcher.

This module exposes two public callables:

- :func:`generate` — a patched replacement for
  ``transformers.PreTrainedModel.generate`` that routes to the
  bracket-constrained search instead of standard beam / sampling decoding.
- :func:`search_wrapper` — the inner dispatcher that sets up shared GPU
  buffers and calls the appropriate search variant based on ``search_mode``.

Usage::

    outputs = generate(
        self=model,
        inputs=input_ids,
        decoding_argument=decode_args.arguments,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        num_beams=4,
        length_penalty=0,
    )
"""
import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import GenerationMixin
from transformers.utils import logging

from ._constrained_beam_search import beam_search
from ._constrained_decoding import bracket_constraint_decode
from ._constrained_decoding_slow import bracket_constraint_decode as slow_bracket_constraint_decode
from ._utils import Candidate

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)


@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    decoding_argument: dict = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
) -> Candidate:
    """
    Run bracket-constrained generation for a *single* source sentence.

    This function replaces the standard ``model.generate()`` for the Codec
    workflow.  It prepares the model inputs, sets up the generation config,
    then dispatches to :func:`search_wrapper` (heuristic search) or
    :func:`~codec._constrained_beam_search.beam_search` (constrained beam
    search) depending on ``decoding_argument['search_mode']``.

    Parameters:
        self: The HuggingFace ``PreTrainedModel`` instance.  Pass the loaded
            model here (``generate(self=model, ...)``) because this function
            is not bound as a method.
        inputs: Source token IDs (encoder input), shape ``(1, src_len)``.
            Must be a **single** example (batch size 1) — the constrained
            search does not support data-batching.
        decoding_argument: Dictionary produced by
            :attr:`~codec._decoding_argument.BracketConstraintDecodingArgument.arguments`.
            Contains the template IDs, bracket stack, search hyperparameters,
            and the :class:`~codec._utils.Candidate` accumulator.
        generation_config: Optional HuggingFace ``GenerationConfig``.  If
            ``None`` the model's default config is used.
        logits_processor: Additional logits processors to apply on top of the
            defaults derived from ``generation_config`` (e.g. a
            ``ForcedBOSTokenLogitsProcessor`` for the target language ID).
        stopping_criteria: Custom stopping criteria.  The default
            ``MaxLengthCriteria`` is always added automatically.
        prefix_allowed_tokens_fn: Optional prefix-constrained decoding
            callback (forwarded to the logits processor stack but not used by
            the constrained search itself).
        synced_gpus: Enable for DeepSpeed ZeRO-3 multi-GPU setups to keep all
            GPUs in lock-step.  Defaults to ``True`` when ZeRO-3 is detected.
        assistant_model: Unused in the constrained search; kept for API
            compatibility with standard ``generate``.
        streamer: Unused in the constrained search; kept for API compatibility.
        **kwargs: Any ``GenerationConfig`` attribute can be overridden here
            (e.g. ``num_beams=4``, ``length_penalty=0``,
            ``forced_bos_token_id=<lang_id>``).

    Returns:
        The :class:`~codec._utils.Candidate` object from
        ``decoding_argument['candidate']``, populated with up to ``n_best``
        complete hypotheses in ``candidate.min_heap``.  Each heap entry is a
        tuple ``(score, -count, token_ids, acc_scores, [open_pos, close_pos])``.
    """
    if synced_gpus is None:
        synced_gpus = is_deepspeed_zero3_enabled() and dist.get_world_size() > 1

    # --- 1. Resolve generation config ----------------------------------
    self._validate_model_class()
    if generation_config is None:
        if self.generation_config._from_model_config:
            new_cfg = GenerationConfig.from_model_config(self.config)
            if new_cfg != self.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. "
                    "This is a deprecated strategy and will be removed in a future version. "
                    "Please use a generation configuration file instead."
                )
                self.generation_config = new_cfg
        generation_config = self.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    generation_config.validate()
    self._validate_model_kwargs(model_kwargs.copy())

    logits_processor = logits_processor or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask") is None:
            logger.warning(
                "The attention mask and the pad token id were not set.  Pass your input's "
                "`attention_mask` to obtain reliable results."
            )
        eos_id = generation_config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_id} for open-end generation.")
        generation_config.pad_token_id = eos_id

    # --- 2. Prepare model inputs ---------------------------------------
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attn = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attn = "encoder_outputs" not in model_kwargs
    if model_kwargs.get("attention_mask") is None and requires_attn and accepts_attn:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    if not self.config.is_encoder_decoder:
        if (
            generation_config.pad_token_id is not None
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected. "
                "Set `padding_side='left'` in the tokenizer for correct generation results."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # --- 3. Prepare decoder input_ids ----------------------------------
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    # --- 4. Validate / adjust max_length -------------------------------
    input_ids_seq_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control generation length. "
            "This behaviour is deprecated — use `max_new_tokens` instead.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        if not has_default_max_length:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and "
                f"`max_length` (={generation_config.max_length}) were set. "
                "`max_new_tokens` will take precedence."
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

    if (
        generation_config.min_length is not None
        and generation_config.min_length > generation_config.max_length
    ):
        raise ValueError(
            f"Unfeasible length constraints: min_length ({generation_config.min_length}) > "
            f"max_length ({generation_config.max_length})."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            f"`input_ids` is on {input_ids.device.type} but the model is on {self.device.type}. "
            f"Move input_ids to the model device before calling generate."
        )

    # --- 5. Build logits processors and stopping criteria --------------
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )
    stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    # --- 6. Dispatch to the appropriate search variant -----------------
    if decoding_argument["search_mode"] == 2:
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            max_length=1024,
            num_beam_hyps_to_keep=generation_config.num_beams,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=True,
        )
        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        return beam_search(
            self,
            input_ids,
            beam_scorer,
            num_return_candidates=decoding_argument["n_best"],
            candidate=decoding_argument["candidate"],
            template_ids=decoding_argument["template_ids"],
            bracket_stack=decoding_argument["bracket_stack"],
            bracket_mapping=decoding_argument["bracket_mapping"],
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    return search_wrapper(
        model=self,
        input_ids=input_ids,
        decoding_argument=decoding_argument,
        logits_processor=logits_processor,
        stopping_criteria=stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        synced_gpus=synced_gpus,
        **model_kwargs,
    )


@torch.no_grad()
def search_wrapper(
    model,
    input_ids: torch.LongTensor,
    decoding_argument: dict,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **model_kwargs,
) -> Candidate:
    """
    Set up shared GPU buffers and dispatch to the fast or slow heuristic search.

    This function is called by :func:`generate` for ``search_mode`` 0 and 1.
    It replicates the single encoder output and attention mask into a
    ``search_batch_size``-sized tensor so that the recursive search can reuse
    them without repeated allocation.

    Parameters:
        model: The HuggingFace encoder-decoder model.
        input_ids: Initial decoder input token IDs, shape ``(1, 1)`` (just the
            decoder-start / BOS token at this point).
        decoding_argument: Dictionary from
            :attr:`~codec._decoding_argument.BracketConstraintDecodingArgument.arguments`.
        logits_processor: Logits processor list (e.g. forced BOS / lang token).
        stopping_criteria: Stopping criteria list.
        max_length: Deprecated; prefer a ``MaxLengthCriteria`` stopping
            criterion.
        pad_token_id: Pad token ID of the model.
        eos_token_id: EOS token ID of the model.
        **model_kwargs: Must contain ``attention_mask`` and
            ``encoder_outputs`` (both with batch dimension 1) as produced by
            :func:`generate` after running the encoder.

    Returns:
        The :class:`~codec._utils.Candidate` object populated with completed
        hypotheses.
    """
    logits_processor = logits_processor or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()

    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated here; use "
            "`stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id if pad_token_id is not None else model.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.generation_config.eos_token_id

    device = input_ids.get_device()
    scores = torch.zeros(1)

    # Unpack decoding arguments.
    template_ids = decoding_argument["template_ids"]
    template_pointer = decoding_argument["template_pointer"]
    bracket_stack = decoding_argument["bracket_stack"]
    stack_pointer = decoding_argument["stack_pointer"]
    candidate = decoding_argument["candidate"]
    bracket_mapping = decoding_argument["bracket_mapping"]
    future_steps = decoding_argument["future_steps"]
    search_mode = decoding_argument["search_mode"]
    search_batch_size = decoding_argument["batch_size"]
    save_visualization = decoding_argument["save_visualization"]
    possible_opening_positions = decoding_argument["possible_opening_positions"]
    possible_closing_positions = decoding_argument["possible_closing_positions"]
    left_marker = decoding_argument["left_marker"]
    right_marker = decoding_argument["right_marker"]
    n_best = decoding_argument["n_best"]

    # Pre-allocate encoder output and attention mask tiled to search_batch_size
    # so child calls can take slices without re-allocating.
    assert (
        model_kwargs["attention_mask"].shape[0] == 1
        and model_kwargs["encoder_outputs"].last_hidden_state.shape[0] == 1
    ), "search_wrapper expects a single-example encoder output (batch size 1)."

    cache_tensors = {
        "attention_mask": model_kwargs["attention_mask"].repeat(search_batch_size, 1),
        "encoder_last_hidden_state": model_kwargs["encoder_outputs"].last_hidden_state.repeat(
            search_batch_size, 1, 1
        ),
    }

    if device >= 0:
        template_ids = template_ids.to(device)
        template_pointer = template_pointer.to(device)
        stack_pointer = stack_pointer.to(device)
        scores = scores.to(device)
        candidate.to_cuda(device)

    common_kwargs = dict(
        input_ids=input_ids,
        future_steps=future_steps,
        model=model,
        scores=scores,
        batch_size=search_batch_size,
        curr_new_tokens=0,
        template_ids=template_ids,
        template_pointer=template_pointer,
        bracket_stack=bracket_stack,
        bracket_mapping=bracket_mapping,
        stack_pointers=stack_pointer,
        candidate=candidate,
        cache_tensors=cache_tensors,
        save_visualization=save_visualization,
        left_marker=left_marker,
        right_marker=right_marker,
        possible_opening_positions=possible_opening_positions,
        possible_closing_positions=possible_closing_positions,
        n_best=n_best,
        model_kwargs=model_kwargs,
        logits_processor=logits_processor,
        prev_accumulated_scores=[[0]],
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    if search_mode == 0:
        # Fast variant: keeps all branches on GPU simultaneously.
        bracket_constraint_decode(**common_kwargs)
    else:
        # Slow variant: offloads KV cache to CPU between recursive calls.
        slow_bracket_constraint_decode(**common_kwargs)

    return candidate
