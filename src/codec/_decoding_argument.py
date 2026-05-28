"""
Decoding argument builder for the bracket-constrained search.
"""
import torch

from ._utils import Candidate, MODEL2BRACKET_IDS


# Template lengths (in tokens, including BOS/EOS/lang-id) at which the search
# automatically falls back to the slower but more memory-efficient algorithm
# or reduces the search batch size.
_THRESHOLD_SLOW = 106
_THRESHOLD_HALF_BATCH = 170
_THRESHOLD_QUARTER_BATCH = 200


class BracketConstraintDecodingArgument:
    """
    Builder that packages all arguments required by the constrained search into
    a single dictionary consumed by :func:`~codec._generation.generate`.

    The class also implements the automatic ``search_mode`` / ``batch_size``
    fall-back logic that kicks in for long templates (see
    :attr:`effective_search_mode` and :attr:`effective_batch_size`).

    Parameters:
        template_ids: Token IDs of the target-language *template* (the
            machine-translated sentence without span markers).  Must include
            the leading ``[EOS, lang_id]`` and trailing ``[EOS]`` special
            tokens, i.e. shape ``(template_len + 3,)``.
        bracket_stack: Ordered stack of marker symbols still to be inserted.
            The last element is the *next* marker to insert.  For *k* span
            pairs use ``[']', '['] * k`` so that ``[`` is inserted before
            ``]``.
        template_pointer: Index into *template_ids* at which decoding starts.
            Always initialise to ``1`` (skipping the leading EOS).
        model_name: MT model family; must be a key in
            :data:`~codec._utils.MODEL2BRACKET_IDS` (``'nllb'``, ``'mbart'``,
            or ``'m2m'``).
        future_steps: Number of look-ahead steps used by the heuristic
            upper-bound estimate (``alpha`` in the paper).  Pass ``-1`` to use
            the full remaining template length as the look-ahead horizon.
        search_mode: Algorithm variant.
            ``0`` — fast heuristic (high GPU memory usage).
            ``1`` — slow heuristic (low GPU memory usage; auto-selected for
            long templates).
            ``2`` — constrained beam search (no heuristic).
        batch_size: Number of search branches evaluated in parallel on the
            GPU.  For long templates this is automatically reduced to stay
            within memory.
        n_best: Number of top-scoring hypotheses to collect.
        left_marker: Symbol used as the opening span marker (default ``'['``).
        right_marker: Symbol used as the closing span marker (default ``']'``).
        possible_opening_positions: Optional set of *template token indices*
            at which the opening marker (``left_marker``) is allowed to be
            inserted.  When ``None`` all positions are considered (unconstrained
            search).  These positions are 0-based indices into *template_ids*
            (so position 2 corresponds to the first content token after the
            ``[EOS, lang_id]`` prefix).
        possible_closing_positions: Optional mapping from each opening position
            to the set of template token indices where the closing marker
            (``right_marker``) may be placed.  Keys must be a subset of
            *possible_opening_positions* (or all positions when that is
            ``None``).  When ``None`` all positions after the opening marker
            are considered.
        lb_score: Log-probability of the initial lower-bound hypothesis used
            for pruning.  Defaults to ``-inf`` (no pruning at the start).
        lb_text_ids: Token IDs of the initial lower-bound hypothesis.
            Usually ``None``; only useful when warm-starting the search.
        accumulate_scores: Cumulative log-probabilities of the lower-bound
            hypothesis, one per token.  Used by the heuristic future-cost
            estimate.  Usually ``None``.
        save_visualization: If ``True``, the search builds a :class:`Node`
            tree that can be inspected after decoding.
    """

    def __init__(
        self,
        template_ids: torch.LongTensor,
        bracket_stack: list[str],
        template_pointer: int,
        model_name: str,
        future_steps: int,
        search_mode: int,
        batch_size: int,
        n_best: int = 5,
        left_marker: str = "[",
        right_marker: str = "]",
        possible_opening_positions: set[int] | None = None,
        possible_closing_positions: dict[int, set[int]] | None = None,
        lb_score: float | None = None,
        lb_text_ids: torch.LongTensor | None = None,
        accumulate_scores: torch.FloatTensor | None = None,
        save_visualization: bool = False,
    ):
        bracket_mapping = MODEL2BRACKET_IDS[model_name]
        stack_size = len(bracket_stack)

        if lb_score is None:
            lb_score_tensor = torch.FloatTensor([[-float("inf")]])
        else:
            lb_score_tensor = torch.FloatTensor([[lb_score]])

        # Auto-adjust search mode and batch size for long templates so that
        # GPU memory usage stays manageable.
        template_len = len(template_ids)
        effective_search_mode = search_mode
        effective_batch_size = batch_size
        if template_len > _THRESHOLD_SLOW:
            effective_search_mode = 1
            effective_batch_size = batch_size * 3 // 4
        if template_len >= _THRESHOLD_HALF_BATCH:
            effective_batch_size = batch_size // 2
        if template_len > _THRESHOLD_QUARTER_BATCH:
            effective_batch_size = batch_size // 4

        self.arguments: dict = {
            "candidate": Candidate(
                text_ids=lb_text_ids,
                score=lb_score_tensor,
                accumulate_scores=accumulate_scores,
            ),
            "bracket_stack": bracket_stack,
            "bracket_mapping": bracket_mapping,
            "template_pointer": torch.LongTensor([template_pointer]),
            "stack_pointer": torch.LongTensor([stack_size - 1]),
            "template_ids": template_ids,
            "batch_size": effective_batch_size,
            "future_steps": future_steps,
            "search_mode": effective_search_mode,
            "save_visualization": save_visualization,
            "possible_opening_positions": possible_opening_positions,
            "possible_closing_positions": possible_closing_positions,
            "n_best": n_best,
            "left_marker": left_marker,
            "right_marker": right_marker,
        }
