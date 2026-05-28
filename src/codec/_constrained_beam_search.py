"""
Constrained beam search (search mode 2).

An alternative to the heuristic branch-and-bound search implemented in
:mod:`~codec._constrained_decoding` and :mod:`~codec._constrained_decoding_slow`.
Uses standard beam search but restricts the token vocabulary at each step to
only the template token and any pending bracket marker, effectively forcing the
decoder to follow the template while choosing where to insert the markers.

This mode does not use a heuristic upper bound, so it may explore more paths
than the branch-and-bound variants.  It is selected by passing
``search_mode=2`` to
:class:`~codec._decoding_argument.BracketConstraintDecodingArgument`.
"""
import heapq
import warnings
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from transformers.generation.beam_search import BeamScorer
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from ._utils import Candidate


def beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    template_ids: torch.LongTensor,
    candidate: Candidate,
    num_return_candidates: int,
    bracket_mapping: dict[str, list[int]],
    bracket_stack: list[str],
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    **model_kwargs,
) -> Candidate:
    """
    Run constrained beam search and populate *candidate* with the top
    *num_return_candidates* complete hypotheses.

    The search restricts next-token choices at every step to exactly:
    - The *template token* at the current position, and
    - Any *bracket marker* token(s) if a marker is still pending.

    All other vocab entries are masked to ``-inf`` so that the beam only
    expands along paths that follow the template (with optional bracket
    insertions).

    Parameters:
        self: The HuggingFace ``PreTrainedModel`` instance (passed in because
            this function monkey-patches the model's ``generate`` method).
        input_ids: Initial decoder input token IDs, shape
            ``(batch_size * num_beams, cur_len)``.
        beam_scorer: HuggingFace ``BeamSearchScorer`` configured with the
            desired ``batch_size`` and ``num_beams``.
        template_ids: Full template token-ID sequence including special tokens,
            shape ``(T,)``.
        candidate: Accumulator for completed hypotheses.  Mutated in place.
        num_return_candidates: Maximum number of hypotheses to add to
            ``candidate.min_heap``.
        bracket_mapping: Maps each marker symbol to its token ID(s) in the
            model vocabulary.
        bracket_stack: Ordered list of markers still to insert (last element
            is the next marker to insert).
        logits_processor: HuggingFace logits processor list.
        stopping_criteria: HuggingFace stopping criteria list.
        max_length: Deprecated; use a ``MaxLengthCriteria`` stopping criterion
            instead.
        pad_token_id: Pad token ID of the model.
        eos_token_id: EOS token ID (or list of IDs) of the model.
        output_attentions: Whether to return attention weights.
        output_hidden_states: Whether to return hidden states.
        output_scores: Whether to return per-step scores.
        return_dict_in_generate: Whether to return a ``ModelOutput`` dict.
        synced_gpus: Enable under DeepSpeed ZeRO-3 to keep all GPUs in sync.
        **model_kwargs: Additional keyword arguments forwarded to the model
            (e.g. encoder outputs, attention mask).

    Returns:
        The mutated *candidate* object with up to *num_return_candidates*
        hypotheses in ``candidate.min_heap``.
    """
    logits_processor = logits_processor or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()

    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use "
            "`stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    if not stopping_criteria:
        warnings.warn(
            "You haven't defined any stopping criteria; this will likely loop forever.",
            UserWarning,
        )

    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, "
            f"but is {batch_beam_size}."
        )

    # Collect optional outputs.
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size))
        if (return_dict_in_generate and output_scores)
        else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # Initialise beam scores: first beam = 0, rest = -1e9 (ensures only the
    # first beam is expanded at the very first step).
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(batch_size * num_beams)

    # Per-beam pointers into the template and bracket stack.
    template_pointer = torch.ones(input_ids.shape[0], dtype=torch.long)
    stack_pointer = torch.ones(input_ids.shape[0], dtype=torch.long)

    this_peer_finished = False
    while True:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            cur_len += 1
            continue

        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
        next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

        # Mask out all tokens except the template token and pending bracket.
        inf_scores = torch.full_like(next_token_scores, -float("inf"))
        for i in range(len(template_pointer)):
            allowed = [template_ids[template_pointer[i]].item()]
            if stack_pointer[i] >= 0:
                allowed.extend(bracket_mapping[bracket_stack[stack_pointer[i]]])
            for tok in allowed:
                inf_scores[i, tok] = next_token_scores[i, tok]
        next_token_scores = inf_scores

        if return_dict_in_generate and output_scores:
            scores += (next_token_scores_processed,)
        if return_dict_in_generate and output_attentions:
            decoder_attentions += (
                (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            )
            if self.config.is_encoder_decoder:
                cross_attentions += (outputs.cross_attentions,)
        if return_dict_in_generate and output_hidden_states:
            decoder_hidden_states += (
                (outputs.decoder_hidden_states,)
                if self.config.is_encoder_decoder
                else (outputs.hidden_states,)
            )

        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )
        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        # Update per-beam template and stack pointers.
        new_template_pointer = template_pointer.clone()
        new_stack_pointer = stack_pointer.clone()
        for i in range(len(input_ids)):
            last_tok = input_ids[i][-1].item()
            prev_tp = template_pointer[beam_idx[i]]
            prev_sp = stack_pointer[beam_idx[i]]
            if last_tok == template_ids[prev_tp].item():
                new_template_pointer[i] = prev_tp + 1
                new_stack_pointer[i] = prev_sp
            elif last_tok in bracket_mapping["["] or last_tok in bracket_mapping["]"]:
                assert prev_sp >= 0
                new_template_pointer[i] = prev_tp
                new_stack_pointer[i] = prev_sp - 1
        template_pointer = new_template_pointer
        stack_pointer = new_stack_pointer

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple(beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices)))

        cur_len += 1
        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    # ------------------------------------------------------------------
    # Extract bracket positions from the finalised sequences and collect
    # valid hypotheses into the candidate heap.
    # ------------------------------------------------------------------
    output_sequences = sequence_outputs["sequences"]
    output_scores_seq = sequence_outputs["sequence_scores"]
    open_bracket_position = [-1] * output_sequences.shape[0]
    close_bracket_position = [-1] * output_sequences.shape[0]
    valid_hyp_ids = []

    assert len(template_ids) <= output_sequences.shape[1]
    for i in range(len(output_sequences)):
        template_p = 0
        input_p = 0
        while input_p < len(output_sequences[i]):
            if (
                template_p < len(template_ids)
                and template_ids[template_p].item() == output_sequences[i][input_p].item()
            ):
                template_p += 1
                input_p += 1
            else:
                if output_sequences[i][input_p] == pad_token_id:
                    break
                assert (
                    output_sequences[i][input_p] in bracket_mapping["["]
                    or output_sequences[i][input_p] in bracket_mapping["]"]
                )
                if output_sequences[i][input_p] in bracket_mapping["["]:
                    open_bracket_position[i] = input_p
                else:
                    close_bracket_position[i] = input_p
                input_p += 1
        if (
            open_bracket_position[i] > -1
            and close_bracket_position[i] > -1
            and output_scores_seq[i] > -float("inf")
        ):
            valid_hyp_ids.append(i)

    for i in valid_hyp_ids[:num_return_candidates]:
        heapq.heappush(
            candidate.min_heap,
            (
                output_scores_seq[i].item(),
                -i,
                output_sequences[i].unsqueeze(0),
                None,
                [open_bracket_position[i], close_bracket_position[i]],
            ),
        )

    return candidate
