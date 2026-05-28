"""
Fast bracket-constrained decoding (search mode 0).

Keeps all active hypotheses on the GPU simultaneously — high throughput but
high GPU memory usage.  For long templates or memory-constrained setups use
:mod:`~codec._constrained_decoding_slow` instead (search mode 1).

Algorithm overview
------------------
The function implements a best-first branch-and-bound search over all ways to
insert one or more ``[ ]`` span-marker pairs into a given target-language
template.  At each recursive call:

1. Run one forward pass over the *current batch* of hypotheses.
2. For each hypothesis, consider two types of next tokens:
   - The *template token*: advance through the template without inserting a
     marker.
   - A *bracket token*: insert the next pending marker from ``bracket_stack``.
3. Compute an upper-bound score for each candidate (current log-prob + model
   log-prob of the next token).  Prune any candidate whose upper bound is
   below the best complete hypothesis found so far.
4. Sort surviving candidates by upper bound (descending) and process them in
   sub-batches of size ``batch_size``, calling :func:`bracket_constraint_decode`
   recursively for each sub-batch.
"""
import heapq
from typing import Optional

import torch
import torch.nn as nn
from transformers.generation.logits_process import LogitsProcessorList

from ._utils import Candidate, Node


def bracket_constraint_decode(
    input_ids: torch.LongTensor,
    template_ids: torch.LongTensor,
    model,
    scores: torch.FloatTensor,
    curr_new_tokens: int,
    batch_size: int,
    template_pointer: torch.LongTensor,
    bracket_stack: list[str],
    bracket_mapping: dict[str, list[int]],
    stack_pointers: torch.LongTensor,
    candidate: Candidate,
    model_kwargs: dict,
    logits_processor: LogitsProcessorList,
    prev_accumulated_scores: list[list[float]],
    left_marker: str,
    right_marker: str,
    n_best: int = 5,
    cache_tensors: dict | None = None,
    possible_opening_positions: set[int] | None = None,
    possible_closing_positions: dict[int, set[int]] | None = None,
    open_bracket_position: list[int] | None = None,
    close_bracket_position: list[int] | None = None,
    prev_nodes: list[Node] | None = None,
    save_visualization: bool = True,
    future_steps: int = 3,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> None:
    """
    Recursively expand and prune the constrained search tree (fast variant).

    This function mutates *candidate* in place, adding completed hypotheses to
    ``candidate.min_heap`` as they are found.

    Parameters:
        input_ids: Current decoder token IDs for all active branches,
            shape ``(B, L)`` where *B* is the number of active branches.
        template_ids: Full template token-ID sequence including special tokens,
            shape ``(T,)``.
        model: HuggingFace encoder-decoder model.
        scores: Cumulative log-probabilities for each branch, shape ``(B,)``.
        curr_new_tokens: Number of tokens generated so far beyond the initial
            decoder prompt (used to detect the very first call).
        batch_size: Maximum number of branches to expand in a single recursive
            call.
        template_pointer: Index into *template_ids* for each branch indicating
            the next template token to emit or skip, shape ``(B,)``.
        bracket_stack: Global ordered list of markers still to insert.  Shared
            across all branches; individual progress is tracked via
            *stack_pointers*.
        bracket_mapping: Maps each marker symbol to its token ID(s) in the
            model vocabulary.
        stack_pointers: Per-branch index into *bracket_stack* pointing to the
            next marker to insert (``-1`` means all markers have been placed),
            shape ``(B,)``.
        candidate: Accumulator for completed hypotheses.  Mutated in place.
        model_kwargs: Keyword arguments forwarded to the model, including
            encoder outputs and the KV cache.
        logits_processor: HuggingFace logits processor list (e.g. for forced
            BOS / language token).
        prev_accumulated_scores: Per-branch list of per-step cumulative
            log-probabilities, used by the heuristic upper-bound.
        left_marker: Opening marker symbol (e.g. ``'['``).
        right_marker: Closing marker symbol (e.g. ``']'``).
        n_best: Maximum number of hypotheses to retain in ``candidate.min_heap``.
        cache_tensors: Pre-allocated tensors for the encoder hidden states and
            attention mask, tiled to ``batch_size`` to avoid reallocation.
        possible_opening_positions: Optional set of template token indices
            where the opening marker may be inserted.  ``None`` = unrestricted.
        possible_closing_positions: Optional dict mapping each opening position
            to the set of allowed closing positions.  ``None`` = unrestricted.
        open_bracket_position: Per-branch position of the last placed opening
            marker (``-1`` = not yet placed), list of length *B*.
        close_bracket_position: Per-branch position of the last placed closing
            marker (``-1`` = not yet placed), list of length *B*.
        prev_nodes: Per-branch parent :class:`Node` for the visualisation tree.
            Only used when ``save_visualization=True``.
        save_visualization: Build a :class:`Node` search tree for later
            inspection.
        future_steps: Look-ahead horizon for the heuristic upper-bound.  When
            ``-1`` the full remaining template is used.
        pad_token_id: Pad token ID of the model.
        eos_token_id: EOS token ID of the model; a branch is complete when its
            last token equals this value.
    """
    device = input_ids.get_device()

    # ------------------------------------------------------------------
    # 1. Check for completed branches (last token == EOS).
    # ------------------------------------------------------------------
    unfinished = torch.BoolTensor(input_ids.shape[0]).to(device).fill_(True)
    if curr_new_tokens > 0:
        for i in range(input_ids.shape[0]):
            if input_ids[i, -1] == eos_token_id:
                unfinished[i] = False
                # Only accept if all markers have been placed (stack exhausted).
                if stack_pointers[i] == -1:
                    entry = (
                        scores[i].item(),
                        -candidate.count,
                        input_ids[i].unsqueeze(0),
                        prev_accumulated_scores[i],
                        [open_bracket_position[i], close_bracket_position[i]],
                    )
                    if len(candidate.min_heap) < n_best:
                        heapq.heappush(candidate.min_heap, entry)
                        candidate.count += 1
                    elif scores[i].item() > candidate.score:
                        heapq.heappushpop(candidate.min_heap, entry)
                        candidate.count += 1
                        candidate.update_smallest_candidate()

    if not torch.any(unfinished):
        return

    # ------------------------------------------------------------------
    # 2. Discard finished branches and reorder KV cache accordingly.
    # ------------------------------------------------------------------
    if not torch.all(unfinished):
        unfinished_ids = torch.nonzero(unfinished).view(-1)
        input_ids = input_ids[unfinished_ids]
        scores = scores[unfinished_ids]
        template_pointer = template_pointer[unfinished_ids]
        stack_pointers = stack_pointers[unfinished_ids]
        prev_accumulated_scores = [prev_accumulated_scores[i] for i in unfinished_ids]
        if save_visualization:
            prev_nodes = [prev_nodes[i] for i in unfinished_ids]
        open_bracket_position = [open_bracket_position[i] for i in unfinished_ids]
        close_bracket_position = [close_bracket_position[i] for i in unfinished_ids]
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = _reorder_cache(model_kwargs["past_key_values"], unfinished_ids)

    # ------------------------------------------------------------------
    # 3. Forward pass.
    # ------------------------------------------------------------------
    if "attention_mask" in model_kwargs:
        model_kwargs["attention_mask"] = cache_tensors["attention_mask"][: input_ids.shape[0]]
    if "encoder_outputs" in model_kwargs:
        model_kwargs["encoder_outputs"].last_hidden_state = (
            cache_tensors["encoder_last_hidden_state"][: input_ids.shape[0]]
        )

    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
    outputs = model(**model_inputs, return_dict=True)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    )

    # shape: (B, vocab_size)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=input_ids.shape[1])
    next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
    next_token_scores = logits_processor(input_ids, next_token_scores)

    # ------------------------------------------------------------------
    # 4. Enumerate candidate next tokens per branch.
    # ------------------------------------------------------------------
    if open_bracket_position is None:
        open_bracket_position = [-1] * len(input_ids)
    if close_bracket_position is None:
        close_bracket_position = [-1] * len(input_ids)

    prev_batch_ids: list = []
    new_template_pointers: list = []
    new_stack_pointers: list = []
    flatten_next_token_upperbounds: list = []
    token_candidate_ids: list = []
    accumulated_scores: list = []
    nodes: list = []
    all_open_bracket_position: list = []
    all_close_bracket_position: list = []

    for i in range(next_token_scores.shape[0]):
        # Determine the pruning threshold for this branch.
        current_best_score = -float("inf")
        if candidate.accumulate_scores is not None:
            if future_steps < 0:
                current_best_score = candidate.accumulate_scores[-1]
            else:
                end_idx = max(input_ids.shape[1] + future_steps, candidate.close_bracket_position)
                end_idx = min(end_idx, len(candidate.accumulate_scores) - 1)
                current_best_score = candidate.accumulate_scores[end_idx]

        # --- Option A: emit the next template token (no marker inserted) ---
        token_idx = template_ids[template_pointer[i]].item()
        upperbound = next_token_scores[i, token_idx] + scores[i]
        node = Node(
            text=token_idx,
            log_prob=next_token_scores[i, token_idx].item(),
            acc_log_prob=upperbound.item(),
            upperbound=current_best_score,
        )
        if upperbound > current_best_score:
            token_candidate_ids.append(token_idx)
            flatten_next_token_upperbounds.append(upperbound)
            prev_batch_ids.append(i)
            new_template_pointers.append(template_pointer[i] + 1)
            new_stack_pointers.append(stack_pointers[i])
            accumulated_scores.append(prev_accumulated_scores[i] + [upperbound.item()])
            nodes.append(node)
            all_open_bracket_position.append(open_bracket_position[i])
            all_close_bracket_position.append(close_bracket_position[i])
        if prev_nodes:
            prev_nodes[i].add_child_node(node)

        # --- Option B: insert the next pending bracket marker ---
        # Skip if the opening position is restricted and this position is not allowed.
        if (
            possible_opening_positions
            and stack_pointers[i] >= 0
            and bracket_stack[stack_pointers[i]] == left_marker
            and template_pointer[i].item() not in possible_opening_positions
        ):
            continue
        # Skip if the closing position is restricted for the current opening.
        if stack_pointers[i] >= 0 and bracket_stack[stack_pointers[i]] == right_marker and possible_closing_positions:
            _allowed = possible_closing_positions.get(open_bracket_position[i])
            if _allowed and template_pointer[i].item() not in _allowed:
                continue

        if stack_pointers[i] >= 0:
            for bracket_id in bracket_mapping[bracket_stack[stack_pointers[i]]]:
                upperbound = next_token_scores[i, bracket_id] + scores[i]
                node = Node(
                    text=bracket_id,
                    log_prob=next_token_scores[i, bracket_id].item(),
                    acc_log_prob=upperbound.item(),
                    upperbound=current_best_score,
                )
                if prev_nodes:
                    prev_nodes[i].add_child_node(node)
                if upperbound > current_best_score:
                    token_candidate_ids.append(bracket_id)
                    flatten_next_token_upperbounds.append(upperbound)
                    prev_batch_ids.append(i)
                    new_template_pointers.append(template_pointer[i])
                    new_stack_pointers.append(stack_pointers[i] - 1)
                    accumulated_scores.append(prev_accumulated_scores[i] + [upperbound.item()])
                    nodes.append(node)
                    if bracket_stack[stack_pointers[i]] == right_marker:
                        all_close_bracket_position.append(len(input_ids[i]))
                        all_open_bracket_position.append(open_bracket_position[i])
                    else:
                        all_close_bracket_position.append(close_bracket_position[i])
                        all_open_bracket_position.append(len(input_ids[i]))

    if not flatten_next_token_upperbounds:
        return

    if curr_new_tokens == 0 and save_visualization:
        candidate.search_tree = nodes[0]

    # ------------------------------------------------------------------
    # 5. Sort candidates by upper bound and recurse sub-batch by sub-batch.
    # ------------------------------------------------------------------
    prev_batch_ids = torch.LongTensor(prev_batch_ids)
    new_template_pointers = torch.stack(new_template_pointers)
    new_stack_pointers = torch.stack(new_stack_pointers)
    token_candidate_ids = torch.LongTensor(token_candidate_ids)
    flatten_next_token_upperbounds = torch.stack(flatten_next_token_upperbounds)

    if device >= 0:
        token_candidate_ids = token_candidate_ids.to(device)
        prev_batch_ids = prev_batch_ids.to(device)

    order = torch.argsort(flatten_next_token_upperbounds, descending=True)
    sorted_upperbounds = flatten_next_token_upperbounds[order]
    sorted_origin_token_ids = token_candidate_ids[order]
    prev_batch_ids = prev_batch_ids[order]
    new_template_pointers = new_template_pointers[order]
    new_stack_pointers = new_stack_pointers[order]
    accumulated_scores = [accumulated_scores[i] for i in order]
    if save_visualization:
        nodes = [nodes[i] for i in order]
    all_close_bracket_position = [all_close_bracket_position[i] for i in order]
    all_open_bracket_position = [all_open_bracket_position[i] for i in order]

    idx_pointer = 0
    stop = False
    while idx_pointer < len(sorted_origin_token_ids):
        end_batch_idx = min(idx_pointer + batch_size, len(sorted_origin_token_ids))
        real_end_batch_idx = 0
        for i in range(idx_pointer, end_batch_idx):
            if sorted_upperbounds[i] <= candidate.score:
                stop = True
                break
            real_end_batch_idx = i
        real_end_batch_idx += 1

        batch_prev_ids = prev_batch_ids[idx_pointer:real_end_batch_idx]
        batch_next_input_ids = torch.cat(
            [input_ids[batch_prev_ids], sorted_origin_token_ids[idx_pointer:real_end_batch_idx].unsqueeze(1)],
            dim=-1,
        )
        batch_scores = sorted_upperbounds[idx_pointer:real_end_batch_idx]
        batch_template_pointer = new_template_pointers[idx_pointer:real_end_batch_idx]
        batch_stack_pointer = new_stack_pointers[idx_pointer:real_end_batch_idx]
        batch_accumulated_scores = accumulated_scores[idx_pointer:real_end_batch_idx]
        batch_nodes = nodes[idx_pointer:real_end_batch_idx] if save_visualization else []
        batch_open = all_open_bracket_position[idx_pointer:real_end_batch_idx]
        batch_close = all_close_bracket_position[idx_pointer:real_end_batch_idx]

        idx_pointer = real_end_batch_idx
        if len(batch_prev_ids) == 0:
            break

        copy_kwargs = dict(model_kwargs)
        if copy_kwargs["past_key_values"] is not None:
            copy_kwargs["past_key_values"] = _reorder_cache(copy_kwargs["past_key_values"], batch_prev_ids)

        bracket_constraint_decode(
            input_ids=batch_next_input_ids,
            template_ids=template_ids,
            model=model,
            scores=batch_scores,
            curr_new_tokens=curr_new_tokens + 1,
            batch_size=batch_size,
            template_pointer=batch_template_pointer,
            bracket_stack=bracket_stack,
            bracket_mapping=bracket_mapping,
            stack_pointers=batch_stack_pointer,
            candidate=candidate,
            model_kwargs=copy_kwargs,
            prev_nodes=batch_nodes,
            save_visualization=save_visualization,
            left_marker=left_marker,
            right_marker=right_marker,
            n_best=n_best,
            cache_tensors=cache_tensors,
            possible_opening_positions=possible_opening_positions,
            possible_closing_positions=possible_closing_positions,
            open_bracket_position=batch_open,
            close_bracket_position=batch_close,
            prev_accumulated_scores=batch_accumulated_scores,
            future_steps=future_steps,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
        )
        if stop:
            break


def _reorder_cache(past_key_values, beam_idx):
    """Re-index the KV cache along the batch dimension."""
    return tuple(
        tuple(state.index_select(0, beam_idx) for state in layer)
        for layer in past_key_values
    )
