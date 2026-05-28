"""
Slow bracket-constrained decoding (search mode 1).

Functionally identical to :mod:`~codec._constrained_decoding` but moves the
KV cache to CPU between recursive calls, trading GPU memory for speed.  This
variant is automatically selected for long templates (> 106 tokens) by
:class:`~codec._decoding_argument.BracketConstraintDecodingArgument`.
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
    Recursively expand and prune the constrained search tree (memory-efficient variant).

    Identical interface to
    :func:`~codec._constrained_decoding.bracket_constraint_decode`; see that
    function for full parameter documentation.

    The key difference is that after computing logits the KV cache is moved to
    CPU (``past_key_values`` offloaded via :func:`_move_past_to_cpu`) and only
    moved back to GPU for the sub-batch that is actually being expanded.  This
    reduces peak GPU memory at the cost of extra CPU↔GPU transfers.
    """
    device = input_ids.get_device()

    # ------------------------------------------------------------------
    # 1. Check for completed branches.
    # ------------------------------------------------------------------
    unfinished = torch.BoolTensor(input_ids.shape[0]).to(device).fill_(True)
    if curr_new_tokens > 0:
        for i in range(input_ids.shape[0]):
            if input_ids[i, -1] == eos_token_id:
                unfinished[i] = False
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
    # 2. Discard finished branches.
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
            model_kwargs["past_key_values"] = _reorder_cache(
                model_kwargs["past_key_values"], unfinished_ids
            )

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

    next_token_logits = outputs.logits[:, -1, :]
    next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=input_ids.shape[1])
    next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)
    next_token_scores = logits_processor(input_ids, next_token_scores)

    # ------------------------------------------------------------------
    # 4. Enumerate candidate next tokens.
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
        current_best_score = -float("inf")
        if candidate.accumulate_scores is not None:
            if future_steps < 0:
                current_best_score = candidate.accumulate_scores[-1]
            else:
                end_idx = max(input_ids.shape[1] + future_steps, candidate.close_bracket_position)
                end_idx = min(end_idx, len(candidate.accumulate_scores) - 1)
                current_best_score = candidate.accumulate_scores[end_idx]

        # Option A: template token
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

        # Option B: bracket marker
        if (
            possible_opening_positions
            and stack_pointers[i] >= 0
            and bracket_stack[stack_pointers[i]] == left_marker
            and template_pointer[i].item() not in possible_opening_positions
        ):
            continue
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
    # 5. Sort candidates, offload KV cache, then recurse sub-batch by sub-batch.
    # ------------------------------------------------------------------
    prev_batch_ids = torch.LongTensor(prev_batch_ids)
    new_template_pointers = torch.stack(new_template_pointers)
    new_stack_pointers = torch.stack(new_stack_pointers)
    token_candidate_ids = torch.LongTensor(token_candidate_ids)
    flatten_next_token_upperbounds = torch.stack(flatten_next_token_upperbounds)

    if device >= 0:
        token_candidate_ids = token_candidate_ids.to(device)
        prev_batch_ids = prev_batch_ids.to(device)

    # Free GPU memory held by this call's logit tensors before recursing.
    del next_token_scores, next_token_logits, outputs

    # Offload KV cache to CPU so child calls can allocate GPU memory freely.
    if model_kwargs["past_key_values"] is not None:
        model_kwargs["past_key_values"] = _move_past_to_cpu(model_kwargs["past_key_values"])

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
            copy_kwargs["past_key_values"] = _reorder_cache(
                copy_kwargs["past_key_values"], batch_prev_ids, device=device
            )

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


# ---------------------------------------------------------------------------
# KV-cache helpers
# ---------------------------------------------------------------------------

def _move_past_to_cpu(past_key_values):
    """Move all KV-cache tensors to CPU."""
    return tuple(
        tuple(state.cpu() for state in layer)
        for layer in past_key_values
    )


def _reorder_cache(past_key_values, beam_idx, device=None):
    """Re-index the KV cache; optionally move tensors to *device* first."""
    return tuple(
        tuple(
            (state.to(device) if device is not None else state).index_select(0, beam_idx)
            for state in layer
        )
        for layer in past_key_values
    )
