"""
translate_annotations.py — project span annotations from source to target
via the Codec bracket-constrained MT search.

Input
-----
A JSONL file where each line is a JSON object with (at minimum):

    ``text``     (str)  Source sentence.
    ``template`` (str)  Target-language translation without annotations.
    ``label``    (list) Doccano-style span annotations:
                        each element is a 3-element list/tuple
                        ``[start, end, type]`` where *start* is the
                        0-indexed inclusive character offset of the span
                        in ``text``, *end* is the exclusive offset, and
                        *type* is an arbitrary category string.

                        An empty list means "no annotations for this sentence."

Output
------
The same JSONL with an additional key ``label_target`` containing projected
annotations in the same doccano format but with character offsets into
``template``.  Rows with no source annotations are passed through unchanged
(``label_target`` is set to ``[]``).

Usage
-----
.. code-block:: bash

    python src/nlp/translate.py \\
        --input_path  tests/test.jsonl \\
        --output_path tests/test.jsonl \\
        --model_name_or_path ychenNLP/nllb-200-distilled-1.3B-easyproject \\
        --tokenizer_path     facebook/nllb-200-distilled-1.3B \\
        --src_lang eng_Latn \\
        --tgt_lang deu_Latn

All ``--model_*`` arguments are forwarded to :class:`~src.codec.Codec`.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from timeit import default_timer as timer
from typing import Dict

from tqdm import tqdm

from src.codec import Codec


# ---------------------------------------------------------------------------
# Language-code resolution
# ---------------------------------------------------------------------------

# NLLB uses "{iso639-3}_{Script4}" codes, but deviates from strict ISO 639-2/T
# for a handful of languages.  This override dict patches those specific cases.
_NLLB_ISO3_OVERRIDES: Dict[str, str] = {
    "ar": "arb",   # Modern Standard Arabic
    "fa": "pes",   # Western Farsi
    "ms": "zsm",   # Standard Malay
    "lv": "lvs",   # Standard Latvian
    "sq": "als",   # Tosk Albanian
    "zh": "zho",   # Chinese (kept generic; NLLB has zho_Hans / zho_Hant)
    "no": "nno",   # Norwegian → Norwegian Nynorsk (NLLB has nno_Latn, not nor_Latn)
}


def _iso2_to_nllb(iso2: str) -> str:
    """Convert a 2-letter ISO-639-1 code to an NLLB ``"{iso639-3}_{Script}"`` code.

    For example: ``"de"`` → ``"deu_Latn"``, ``"ar"`` → ``"arb_Arab"``.

    Requires ``iso639-lang`` (``pip install iso639-lang``) for the 3-letter part
    and ``langcodes`` (``pip install langcodes``) for script resolution.
    Falls back to ``Latn`` if ``langcodes`` is unavailable.

    Raises:
        ValueError: If the 3-letter code cannot be determined.
    """
    iso2_lower = iso2.lower()
    iso3 = _NLLB_ISO3_OVERRIDES.get(iso2_lower)

    if iso3 is None:
        try:
            from iso639 import Lang
            lang = Lang(pt1=iso2_lower)
            iso3 = lang.pt2t or lang.pt3
            if not iso3:
                raise ValueError(
                    f"iso639-lang returned no 3-letter code for '{iso2}'. "
                    f"Add it to _NLLB_ISO3_OVERRIDES."
                )
        except ImportError:
            raise ValueError(
                f"Cannot resolve ISO 639-3 code for '{iso2}': 'iso639-lang' is not installed. "
                f"Install with: pip install iso639-lang"
            ) from None
        except Exception as e:
            raise ValueError(
                f"Cannot resolve ISO 639-3 code for '{iso2}': {e}. "
                f"Add it to _NLLB_ISO3_OVERRIDES."
            ) from e

    script = "Latn"
    try:
        import langcodes
        lc = langcodes.get(iso2).maximize()
        if lc.script:
            script = lc.script
    except Exception:
        pass  # langcodes unavailable or failed — Latn is a safe default

    return f"{iso3}_{script}"


def _iso3_to_iso2(iso3: str) -> str:
    """Convert an ISO 639-2/T or ISO 639-3 three-letter code to ISO 639-1 (2-letter).

    Requires ``iso639-lang`` (``pip install iso639-lang``).

    Raises:
        ValueError: If the code cannot be resolved.
    """
    try:
        from iso639 import Lang
    except ImportError:
        raise ValueError(
            f"Cannot convert ISO-3 code '{iso3}' to ISO-2: 'iso639-lang' is not installed. "
            f"Install with: pip install iso639-lang"
        ) from None
    try:
        try:
            lang = Lang(pt2t=iso3.lower())
        except Exception:
            lang = Lang(pt3=iso3.lower())
        if not lang.pt1:
            raise ValueError(f"No ISO 639-1 (2-letter) code found for '{iso3}'.")
        return lang.pt1
    except Exception as e:
        raise ValueError(f"Cannot convert ISO-3 code '{iso3}' to ISO-2: {e}") from e


def resolve_nllb_code(code: str, lang_format: str) -> str:
    """Return the NLLB language code for *code* given its *lang_format*.

    Parameters:
        code:        Language identifier supplied by the user.
        lang_format: One of:

                     - ``"nllb"``  — already an NLLB code (e.g. ``"deu_Latn"``),
                       returned unchanged.
                     - ``"iso2"``  — ISO 639-1 two-letter code (e.g. ``"de"``);
                       converted via :func:`_iso2_to_nllb`.
                     - ``"iso3"``  — ISO 639-2/T or ISO 639-3 three-letter code
                       (e.g. ``"deu"``); first converted to ISO-2, then to NLLB.

    Returns:
        An NLLB ``"{iso639-3}_{Script}"`` code.
    """
    if lang_format == "nllb":
        return code
    if lang_format == "iso2":
        return _iso2_to_nllb(code)
    if lang_format == "iso3":
        iso2 = _iso3_to_iso2(code)
        return _iso2_to_nllb(iso2)
    raise ValueError(f"Unknown lang_format '{lang_format}'. Expected 'nllb', 'iso2', or 'iso3'.")


# ---------------------------------------------------------------------------
# Span ↔ bracket helpers
# ---------------------------------------------------------------------------

def labels_to_bracketed(text: str, labels: list[list]) -> tuple[str, list[list]]:
    """Insert ``[`` / ``]`` markers into *text* at the positions given by *labels*.

    Labels are sorted by start offset before processing so the result is
    well-defined regardless of the order they appear in the input.

    Parameters:
        text:   Plain source sentence.
        labels: Doccano-style annotations — each element is
                ``[start_char, end_char, entity_type]``.

    Returns:
        A ``(bracketed_text, sorted_labels)`` tuple.  *sorted_labels* has the
        same elements as *labels* but ordered by start offset; the caller can
        use this order to map Codec's per-span results back to the correct
        entity type.
    """
    # Sort by start offset so we can insert markers left-to-right.
    sorted_labels = sorted(labels, key=lambda x: x[0])

    result = []
    prev_end = 0
    for start, end, _etype in sorted_labels:
        result.append(text[prev_end:start])
        result.append("[")
        result.append(text[start:end])
        result.append("]")
        prev_end = end
    result.append(text[prev_end:])
    return "".join(result), sorted_labels


def bracketed_to_label_offsets(
    bracketed_text: str,
    entity_type: str,
    left_marker: str = "[",
    right_marker: str = "]",
) -> list[int, int, str] | None:
    """Extract the character offsets of the first ``[…]`` span in
    *bracketed_text* and return them as a doccano label relative to the
    *de-bracketed* template string.

    Parameters:
        bracketed_text: Target sentence with exactly one pair of bracket
                        markers, as returned by Codec.
        entity_type:    Entity type string to include in the output label.
        left_marker:    Opening marker symbol (default ``"["``).
        right_marker:   Closing marker symbol (default ``"]"``).

    Returns:
        A ``[start, end, entity_type]`` list, or ``None`` if no complete
        marker pair was found.
    """
    # Find the first [ … ] pair.
    left_idx = bracketed_text.find(left_marker)
    right_idx = bracketed_text.find(right_marker, left_idx + 1)
    if left_idx == -1 or right_idx == -1:
        return None

    # The model often emits "[ span ]" with surrounding spaces that are part
    # of the bracketed string but not part of the span content.  Strip them
    # so that offsets point at the actual non-whitespace content boundaries.
    #
    # All offsets below are into the *bracketed* string; we correct for the
    # removed '[' (and its trailing space) when computing clean-string offsets.

    # Content starts after '[' plus any whitespace immediately following it.
    content_start_in_bracketed = left_idx + 1
    while content_start_in_bracketed < right_idx and bracketed_text[content_start_in_bracketed] == " ":
        content_start_in_bracketed += 1

    # Content ends before ']' minus any whitespace immediately preceding it.
    content_end_in_bracketed = right_idx
    while content_end_in_bracketed > content_start_in_bracketed and bracketed_text[content_end_in_bracketed - 1] == " ":
        content_end_in_bracketed -= 1

    # Convert to offsets in the clean (de-bracketed) string.
    # Characters before left_idx are unchanged; the '[' itself is removed.
    n_stripped_left = content_start_in_bracketed - left_idx - 1  # spaces after '['
    clean_start = left_idx  # '[' is gone, so content starts at left_idx in clean text
    # Between clean_start and the ']' we removed 1 char ('[') plus n_stripped_left spaces.
    clean_end = content_end_in_bracketed - 1 - n_stripped_left  # -1 for the removed '['
    return [clean_start, clean_end, entity_type]


def all_bracketed_to_label_offsets(
    bracketed_text: str,
    entity_types: list[str],
    left_marker: str = "[",
    right_marker: str = "]",
) -> list[list]:
    """Extract all ``[…]`` span offsets from a multiply-bracketed text.

    Like :func:`bracketed_to_label_offsets` but handles *k* bracket pairs in
    one pass, as produced when ``n_spans=k`` is passed to
    :meth:`~src.codec.Codec.translate`.  The i-th extracted pair is assigned
    ``entity_types[i]``.

    Parameters:
        bracketed_text: Target sentence with *k* bracket pairs.
        entity_types:   Ordered entity type strings, one per expected bracket
                        pair (must be sorted by source start offset, i.e. the
                        same order as :func:`labels_to_bracketed` produces).
        left_marker:    Opening marker symbol (default ``"["``).
        right_marker:   Closing marker symbol (default ``"]"``).

    Returns:
        A list of ``[start, end, entity_type]`` labels in the clean
        (de-bracketed) coordinate space.  Pairs with zero-length content are
        silently dropped.
    """
    result = []
    chars_removed = 0   # cumulative chars stripped from bracketed_text so far
    search_from = 0

    for entity_type in entity_types:
        left_idx = bracketed_text.find(left_marker, search_from)
        if left_idx == -1:
            break
        right_idx = bracketed_text.find(right_marker, left_idx + 1)
        if right_idx == -1:
            break

        # Strip surrounding whitespace (model typically emits "[ span ]").
        content_start = left_idx + 1
        while content_start < right_idx and bracketed_text[content_start] == " ":
            content_start += 1
        content_end = right_idx
        while content_end > content_start and bracketed_text[content_end - 1] == " ":
            content_end -= 1

        n_stripped_left  = content_start - left_idx - 1
        n_stripped_right = right_idx - content_end

        # Translate to clean-string offsets, accounting for all brackets
        # already removed (tracked in chars_removed).
        clean_start = left_idx  - chars_removed
        clean_end   = content_end - chars_removed - 1 - n_stripped_left

        # This pair consumed: '[' + leading spaces + ']' + trailing spaces.
        chars_removed += 1 + n_stripped_left + 1 + n_stripped_right

        if clean_start < clean_end:
            result.append([clean_start, clean_end, entity_type])

        search_from = right_idx + 1

    return result


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------

def process_row(
    codec: Codec,
    text: str,
    template: str,
    labels: list[list],
    tgt_lang: str,
    translate_kwargs: dict,
    decode_jointly: bool = True,
) -> list[list]:
    """Project *labels* from *text* onto *template* using Codec.

    Parameters:
        codec:             Loaded :class:`~src.codec.Codec` instance.
        text:              Plain source sentence.
        template:          Target-language sentence without markers.
        labels:            Doccano-style source annotations
                           (``[[start, end, type], …]``).
        tgt_lang:          FLORES-200 target language code.
        translate_kwargs:  Extra keyword arguments forwarded to
                           :meth:`~src.codec.Codec.translate` (e.g.
                           ``search_mode``, ``batch_size``).
        decode_jointly:    If ``True``, bracket *all* spans in the source and
                           call :meth:`~src.codec.Codec.translate` **once**
                           with ``n_spans=k``.  Faster for sentences with many
                           spans, but the search space grows as O(L^{2k}).
                           If ``False`` (default), one call per span
                           (``n_spans=1``) is made, which is more robust for
                           large k.

    Returns:
        Projected annotations as a list of ``[start, end, type]`` lists in
        source-start-offset order.  Spans for which no valid hypothesis is
        found are omitted.
    """
    if not labels:
        return []

    if decode_jointly:
        # --- Single call: all k spans bracketed at once --------------------
        src_bracketed, sorted_labels = labels_to_bracketed(text, labels)
        entity_types = [lbl[2] for lbl in sorted_labels]

        results = codec.translate(
            src_text=src_bracketed,
            template=template,
            n_spans=len(sorted_labels),
            tgt_lang=tgt_lang,
            **translate_kwargs,
        )

        if not results:
            return []
        return all_bracketed_to_label_offsets(results[0]["text"], entity_types)

    else:
        # --- One call per span (original behaviour) ------------------------
        sorted_labels = sorted(labels, key=lambda x: x[0])
        target_labels: list[list] = []

        for start, end, entity_type in sorted_labels:
            src_with_bracket = text[:start] + "[" + text[start:end] + "]" + text[end:]

            results = codec.translate(
                src_text=src_with_bracket,
                template=template,
                n_spans=1,
                tgt_lang=tgt_lang,
                **translate_kwargs,
            )

            if not results:
                continue

            label = bracketed_to_label_offsets(results[0]["text"], entity_type)
            if label is not None and label[0] < label[1]:
                target_labels.append(label)

        return target_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Validate: at least one of --tgt_lang / --lang_col must be usable.
    if args.tgt_lang is None and args.lang_col is None:
        raise ValueError(
            "Provide either --tgt_lang (constant for the whole file) or "
            "--lang_col (per-row field name).  At least one is required."
        )

    # Resolve language codes to NLLB format before loading the model.
    src_lang_nllb = resolve_nllb_code(args.src_lang, args.lang_format)

    # Constant target language (resolved once) or None (resolved per row).
    tgt_lang_nllb: str | None = None
    if args.tgt_lang is not None:
        tgt_lang_nllb = resolve_nllb_code(args.tgt_lang, args.lang_format)
        print(f"Target language: {args.tgt_lang!r} → {tgt_lang_nllb} (constant)")
    else:
        print(f"Target language: determined per-row from field '{args.lang_col}'")
    print(f"Source language: {args.src_lang!r} → {src_lang_nllb}")

    # Cache of already-resolved codes to avoid redundant lookups.
    _resolved_cache: dict[str, str] = {}

    def get_tgt_lang(record: dict) -> str:
        if tgt_lang_nllb is not None:
            return tgt_lang_nllb
        raw = record.get(args.lang_col)
        if raw is None:
            raise KeyError(
                f"Row has no field '{args.lang_col}'. "
                f"Pass --tgt_lang to use a constant target language instead."
            )
        if raw not in _resolved_cache:
            _resolved_cache[raw] = resolve_nllb_code(raw, args.lang_format)
        return _resolved_cache[raw]

    codec = Codec(
        model_name_or_path=args.model_name_or_path,
        tokenizer_path=args.tokenizer_path,
        src_lang=src_lang_nllb,
        mt_name=args.mt_name,
        max_length=args.max_length,
    )

    translate_kwargs = dict(
        search_mode=args.search_mode,
        batch_size=args.batch_size,
        future_steps=args.future_steps,
        n_best=1,  # we only need the top-1 result
    )

    # Count lines for the progress bar without loading everything into memory.
    with open(args.input_path) as f:
        total_lines = sum(1 for _ in f)

    acc_time = 0.0
    skipped = 0

    with open(args.input_path) as fin, open(args.output_path, "w") as fout:
        for line in tqdm(fin, total=total_lines, desc="Translating annotations"):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text: str = record[args.text_key]
            template: str = record[args.template_key]
            labels: list = record.get(args.label_key, [])

            t0 = timer()
            if labels:
                target_labels = process_row(
                    codec=codec,
                    text=text,
                    template=template,
                    labels=labels,
                    tgt_lang=get_tgt_lang(record),
                    translate_kwargs=translate_kwargs,
                    decode_jointly=not args.disable_joint_decoding,
                )
            else:
                target_labels = []
                skipped += 1

            acc_time += timer() - t0
            record[args.output_label_key] = target_labels
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_processed = total_lines - skipped
    print(f"Total decoding time : {acc_time:.1f}s")
    print(f"Avg time / example  : {acc_time / n_processed:.2f}s")
    print(f"Output written to   : {args.output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Project doccano span annotations from a source sentence onto its "
            "target-language translation using the Codec constrained MT search."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---------------------------------------------------------------
    io = p.add_argument_group("I/O")
    io.add_argument("--input_path", required=True,
                    help="Path to the input JSONL file.")
    io.add_argument("--output_path", required=True,
                    help="Path to write the output JSONL file.")

    # --- JSONL field names -------------------------------------------------
    fields = p.add_argument_group("JSONL field names")
    fields.add_argument("--text_key", default="text",
                        help="Key for the source sentence in each JSON object.")
    fields.add_argument("--template_key", default="template",
                        help="Key for the target-language template sentence.")
    fields.add_argument("--label_key", default="label",
                        help="Key for the source-side doccano annotations "
                             "(list of [start, end, type]).")
    fields.add_argument("--output_label_key", default="label_target",
                        help="Key under which projected target annotations are "
                             "written in the output.")

    # --- Language ----------------------------------------------------------
    lang = p.add_argument_group("Language")
    lang.add_argument("--src_lang", default="en",
                      help="Source language code.  Format must match --lang_format "
                           "(default: 'en' as ISO-2).")
    lang.add_argument("--tgt_lang", default=None,
                      help="Target language code (constant for all rows).  "
                           "Format must match --lang_format "
                           "(e.g. 'de' for ISO-2, 'deu' for ISO-3, 'deu_Latn' for NLLB).  "
                           "Mutually exclusive with --lang_col; one of the two is required.")
    lang.add_argument("--lang_col", default=None,
                      help="JSONL field whose value gives the target language for each row.  "
                           "Used when the target language varies across lines.  "
                           "The field value must be in the format specified by --lang_format.  "
                           "Ignored when --tgt_lang is set.")
    lang.add_argument("--lang_format", default="iso2",
                      choices=["iso2", "iso3", "nllb"],
                      help="Format of --src_lang / --tgt_lang: "
                           "'iso2' = ISO 639-1 two-letter (default), "
                           "'iso3' = ISO 639-2/T or 639-3 three-letter, "
                           "'nllb' = NLLB '{iso639-3}_{Script}' code (pass through unchanged).")

    # --- Model -------------------------------------------------------------
    model = p.add_argument_group("Model")
    model.add_argument("--model_name_or_path",
                       default="ychenNLP/nllb-200-distilled-1.3B-easyproject",
                       help="HuggingFace model ID or local path for the MT model.")
    model.add_argument("--tokenizer_path", default=None,
                       help="HuggingFace tokenizer ID or local path.  "
                            "Defaults to --model_name_or_path.")
    model.add_argument("--mt_name", default="nllb",
                       help="MT model family (nllb | mbart | m2m).")
    model.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length for tokenisation/generation.")

    # --- Search ------------------------------------------------------------
    search = p.add_argument_group("Search")
    search.add_argument("--search_mode", choices=[0, 1, 2], type=int, default=0,
                        help="0 = fast heuristic, 1 = slow heuristic, "
                             "2 = constrained beam search.")
    search.add_argument("--batch_size", type=int, default=16,
                        help="Number of search branches expanded in parallel.")
    search.add_argument("--future_steps", type=int, default=-1,
                        help="Look-ahead horizon for the heuristic upper bound.  "
                             "-1 uses the full remaining template.")
    search.add_argument("--disable_joint_decoding", action="store_true", default=False,
                        help="Process one span at a time (n_spans=1 per call) instead "
                             "of passing all k spans to codec.translate at once.  "
                             "Safer for sentences with many spans (avoids O(L^{2k}) "
                             "search explosion), but requires k model calls per sentence.")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    main(args)
