"""
High-level ``Codec`` class for interactive and programmatic use.

Example usage in a notebook or script::

    from src.codec import Codec

    codec = Codec(
        model_name_or_path="ychenNLP/nllb-200-distilled-1.3B-easyproject",
        tokenizer_path="facebook/nllb-200-distilled-1.3B",
        src_lang="eng_Latn",
    )

    results = codec.translate(
        src_text="[A translator] always risks [source-language influence].",
        template="Ein Übersetzer riskiert immer den Einfluss der Ausgangssprache.",
        n_spans=2,
        tgt_lang="deu_Latn",
    )

    for r in results:
        print(r["text"], "  score:", r["score"])
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ._decoding_argument import BracketConstraintDecodingArgument
from ._generation import generate
from ._utils import tokenize_non_whitespace


class Codec:
    """
    Wrapper around the bracket-constrained MT search.

    Loads the model and tokenizer once, then exposes a :meth:`translate` method
    that can be called repeatedly for different inputs without reloading weights.

    Parameters:
        model_name_or_path: HuggingFace model identifier or local path for the
            MT model (e.g. ``"ychenNLP/nllb-200-distilled-1.3B-easyproject"``).
        tokenizer_path: HuggingFace identifier or local path for the tokenizer.
            Defaults to *model_name_or_path* when ``None``.
        src_lang: BCP-47 / FLORES-200 source language code used to initialise
            the tokenizer (e.g. ``"eng_Latn"``).
        mt_name: MT model family name; must be a key in
            :data:`~codec._utils.MODEL2BRACKET_IDS`.  Controls which token IDs
            are used for the bracket markers.  Defaults to ``"nllb"``.
        device: PyTorch device string (``"cuda"``, ``"cpu"``, ``"cuda:1"``,
            …).  Defaults to ``"cuda"`` when a GPU is available, otherwise
            ``"cpu"``.
        max_length: Maximum sequence length for tokenisation and generation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_path: Optional[str] = None,
        src_lang: str = "eng_Latn",
        mt_name: str = "nllb",
        device: Optional[str] = None,
        max_length: int = 1024,
    ):
        if tokenizer_path is None:
            tokenizer_path = model_name_or_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.mt_name = mt_name
        self.max_length = max_length
        self.src_lang = src_lang

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                src_lang=src_lang,
                max_length=max_length,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_lang_token_id(self, lang_code: str) -> list[int]:
        """Return ``[token_id]`` for *lang_code*, or raise a clear ``ValueError``."""
        try:
            return [self.tokenizer.lang_code_to_id[lang_code]]
        except KeyError:
            known = sorted(self.tokenizer.lang_code_to_id.keys())
            prefix = lang_code.split("_")[0]
            suggestions = [c for c in known if c.startswith(prefix)]
            hint = (
                f"  Possible matches for '{prefix}': {suggestions}"
                if suggestions else
                f"  Run list(codec.tokenizer.lang_code_to_id) to see all valid codes."
            )
            raise ValueError(
                f"Unknown NLLB language code '{lang_code}'.\n{hint}"
            ) from None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        src_text: str,
        template: str,
        n_spans: int = 1,
        tgt_lang: str = "deu_Latn",
        candidates: Optional[list[int]] = None,
        right_candidates: Optional[dict[int, list[int]]] = None,
        search_mode: int = 0,
        batch_size: int = 16,
        future_steps: int = -1,
        n_best: int = 5,
        num_beams: int = 4,
        left_marker: str = "[",
        right_marker: str = "]",
        use_whitespace_positions_only: bool = False,
        save_visualization: bool = False,
    ) -> list[dict]:
        """
        Translate *src_text* and insert span markers into *template*.

        Parameters:
            src_text: Source sentence with bracket markers indicating the span
                boundaries to project (e.g. ``"[A translator] always risks …"``).
            template: Target-language translation of the source sentence
                *without* bracket markers.  The search will insert markers at
                the positions that maximise the translation model score.
            n_spans: Number of ``[ ]`` span-marker pairs to insert.  Must
                match the number of span pairs in *src_text*.
            tgt_lang: FLORES-200 target language code (e.g. ``"deu_Latn"``).
            candidates: Optional list of token indices (0-based into the
                tokenised *template*, including the ``[EOS, lang_id]`` prefix)
                where the opening marker is allowed to be inserted.  When
                ``None`` all positions are explored (free search).
            right_candidates: Optional mapping from each opening position in
                *candidates* to the set of token indices where the closing
                marker is allowed.  Ignored when *candidates* is ``None``.
                When ``None`` all positions after the opening marker are
                explored.
            search_mode: Algorithm variant.
                ``0`` — fast heuristic (high GPU memory).
                ``1`` — slow heuristic (low GPU memory; auto-selected for
                long templates).
                ``2`` — constrained beam search.
            batch_size: Number of search branches expanded in parallel.
                Automatically reduced for long templates.
            future_steps: Look-ahead horizon for the heuristic upper bound.
                ``-1`` uses the full remaining template as the horizon.
            n_best: Number of top-scoring hypotheses to return.
            num_beams: Beam size (only used for ``search_mode=2``).
            left_marker: Opening span-marker symbol (default ``"["``).
            right_marker: Closing span-marker symbol (default ``"]"``).
            use_whitespace_positions_only: When ``True``, restrict marker
                insertion to token positions that start a new word (i.e. those
                whose SentencePiece token begins with ``▁``).  Useful as a
                lightweight alternative to providing explicit *candidates*.
            save_visualization: Build a search-tree
                :class:`~codec._utils.Node` object that can be inspected after
                decoding (stored in ``outputs.search_tree``).

        Returns:
            A list of up to *n_best* result dictionaries, sorted by descending
            score.  Each dict has the keys:

            - ``"text"`` (str): Decoded target sentence with markers inserted.
            - ``"score"`` (float): Cumulative log-probability of the hypothesis.
            - ``"open_pos"`` (int): Token index of the opening marker ``[``.
            - ``"close_pos"`` (int): Token index of the closing marker ``]``.
        """
        # --- Tokenise template -----------------------------------------
        if tgt_lang.startswith("zh"):
            pre_template_ids = tokenize_non_whitespace(template, self.tokenizer)
        else:
            pre_template_ids = self.tokenizer(
                template,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            ).input_ids

        template_ids = torch.LongTensor(
            [self.tokenizer.eos_token_id]
            + self._resolve_lang_token_id(tgt_lang)
            + pre_template_ids
            + [self.tokenizer.eos_token_id]
        )

        # --- Resolve whitespace-only position constraint ---------------
        possible_opening_positions: set[int] | None = None
        possible_closing_positions: dict[int, set[int]] | None = None

        if use_whitespace_positions_only:
            ws_positions: set[int] = set()
            for idx, tok in enumerate(self.tokenizer.tokenize(template)):
                if tok.startswith("▁"):
                    ws_positions.add(idx + 2)  # +2 for [EOS, lang_id] prefix
            possible_opening_positions = ws_positions
            possible_closing_positions = {k: ws_positions for k in ws_positions}
        elif candidates is not None:
            possible_opening_positions = set(candidates)
            if right_candidates is not None:
                possible_closing_positions = {k: set(v) for k, v in right_candidates.items()}

        # --- Tokenise source -------------------------------------------
        input_ids = self.tokenizer(
            src_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).input_ids.to(self.device)

        # --- Build decoding arguments ----------------------------------
        bracket_stack = [right_marker, left_marker] * n_spans
        decode_args = BracketConstraintDecodingArgument(
            template_ids=template_ids,
            bracket_stack=bracket_stack,
            template_pointer=1,
            model_name=self.mt_name,
            future_steps=future_steps,
            search_mode=search_mode,
            batch_size=batch_size,
            n_best=n_best,
            left_marker=left_marker,
            right_marker=right_marker,
            possible_opening_positions=possible_opening_positions,
            possible_closing_positions=possible_closing_positions,
            save_visualization=save_visualization,
        )

        # --- Run search ------------------------------------------------
        outputs = generate(
            self=self.model,
            inputs=input_ids,
            decoding_argument=decode_args.arguments,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=self.max_length,
            num_beams=num_beams,
            length_penalty=0,
        )

        # --- Decode and sort results -----------------------------------
        outputs.min_heap.sort(reverse=True)
        results: list[dict] = []
        for score, _neg_count, token_ids, _acc, (open_pos, close_pos) in outputs.min_heap:
            text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
            results.append({
                "text": text,
                "score": score,
                "open_pos": open_pos,
                "close_pos": close_pos,
            })
        return results
