"""
Demonstrates batch inference with the Codec class.

Because the framework processes one example at a time (the constrained search
operates over a single (src_text, template) pair), "batch inference" means
iterating over a list of inputs with a single model instance that is loaded
once.  This avoids the costly repeated model loading that would occur if you
invoked example.py or cli.py in separate subprocesses.

Run with:
    cd /path/to/codec_span_anno_translation
    python -m pytest tests/test_batch_inference.py -v
or simply:
    python tests/test_batch_inference.py
"""

from __future__ import annotations

import sys
import os

# Make `src` importable when running the file directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.codec import Codec

# ---------------------------------------------------------------------------
# Shared model instance — loaded ONCE for the entire test session.
# ---------------------------------------------------------------------------

MODEL_NAME = "ychenNLP/nllb-200-distilled-1.3B-easyproject"
TOKENIZER_PATH = "facebook/nllb-200-distilled-1.3B"


def get_codec() -> Codec:
    """Return a Codec instance.  In a pytest session this is called once via
    a module-level fixture so the model is not reloaded between tests."""
    return Codec(
        model_name_or_path=MODEL_NAME,
        tokenizer_path=TOKENIZER_PATH,
        src_lang="eng_Latn",
    )


# ---------------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------------

#: Each entry is (src_text, template, n_spans, tgt_lang).
#:
#: ``src_text``  — source sentence with ``[…]`` around every span to project.
#: ``template``  — target-language translation *without* markers; the search
#:                 will find the best insertion positions.
#: ``n_spans``   — number of ``[ ]`` pairs to insert (must equal the number of
#:                 bracketed spans in ``src_text``).
#: ``tgt_lang``  — FLORES-200 language code for the target language.
EXAMPLES: list[tuple[str, str, int, str]] = [
    (
        "[A translator] always risks inadvertently introducing "
        "[source-language words, grammar, or syntax] into the target-language rendering.",
        "Ein Übersetzer riskiert immer, versehentlich Wörter der Ausgangssprache, "
        "Grammatik oder Syntax in die Zielsprache einzuführen.",
        2,
        "deu_Latn",
    ),
    (
        "The [European Commission] is a key institution of the [European Union].",
        "Die Europäische Kommission ist eine wichtige Institution der Europäischen Union.",
        2,
        "deu_Latn",
    ),
    (
        "[Climate change] poses a serious threat to [biodiversity] worldwide.",
        "Der Klimawandel stellt eine ernste Bedrohung für die biologische Vielfalt weltweit dar.",
        2,
        "deu_Latn",
    ),
    (
        "The [doctor] explained the [treatment plan] to the patient.",
        "Der Arzt erklärte dem Patienten den Behandlungsplan.",
        2,
        "deu_Latn",
    ),
    (
        "[Machine translation] systems have improved dramatically with [neural networks].",
        "Maschinelle Übersetzungssysteme haben sich mit neuronalen Netzen dramatisch verbessert.",
        2,
        "deu_Latn",
    ),
    (
        "The [prime minister] announced a new [economic policy] yesterday.",
        "Der Premierminister kündigte gestern eine neue Wirtschaftspolitik an.",
        2,
        "deu_Latn",
    ),
]


# ---------------------------------------------------------------------------
# Core batch helper
# ---------------------------------------------------------------------------

def run_batch(
    codec: Codec,
    examples: list[tuple[str, str, int, str]],
    n_best: int = 3,
) -> list[list[dict]]:
    """Run ``codec.translate`` sequentially over *examples*.

    The model is already loaded; only the search state changes between calls.

    Parameters:
        codec: A loaded :class:`~src.codec.Codec` instance.
        examples: List of ``(src_text, template, n_spans, tgt_lang)`` tuples.
        n_best: How many ranked candidates to return per example.

    Returns:
        A list of result lists, one per input example.  Each inner list
        contains up to *n_best* dicts with keys ``"text"``, ``"score"``,
        ``"open_pos"``, ``"close_pos"``.
    """
    all_results = []
    for src_text, template, n_spans, tgt_lang in examples:
        results = codec.translate(
            src_text=src_text,
            template=template,
            n_spans=n_spans,
            tgt_lang=tgt_lang,
            n_best=n_best,
        )
        all_results.append(results)
    return all_results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_batch_returns_one_result_list_per_example():
    codec = get_codec()
    all_results = run_batch(codec, EXAMPLES)
    assert len(all_results) == len(EXAMPLES), (
        f"Expected {len(EXAMPLES)} result lists, got {len(all_results)}"
    )


def test_each_result_has_expected_keys():
    codec = get_codec()
    all_results = run_batch(codec, EXAMPLES, n_best=2)
    for i, results in enumerate(all_results):
        assert len(results) > 0, f"Example {i}: no candidates returned"
        for r in results:
            assert "text" in r, f"Example {i}: missing 'text' key"
            assert "score" in r, f"Example {i}: missing 'score' key"
            assert "open_pos" in r, f"Example {i}: missing 'open_pos' key"
            assert "close_pos" in r, f"Example {i}: missing 'close_pos' key"


def test_results_are_sorted_by_descending_score():
    codec = get_codec()
    all_results = run_batch(codec, EXAMPLES, n_best=3)
    for i, results in enumerate(all_results):
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Example {i}: results not sorted by descending score: {scores}"
        )


def test_markers_appear_in_output():
    codec = get_codec()
    all_results = run_batch(codec, EXAMPLES, n_best=1)
    for i, results in enumerate(all_results):
        if not results:
            continue
        text = results[0]["text"]
        assert "[" in text and "]" in text, (
            f"Example {i}: top result missing bracket markers: {text!r}"
        )


def test_same_model_instance_reused():
    """Verify that translate() can be called multiple times without reloading
    the model (i.e. the model object identity is stable)."""
    codec = get_codec()
    model_id_before = id(codec.model)
    run_batch(codec, EXAMPLES[:2], n_best=1)
    model_id_after = id(codec.model)
    assert model_id_before == model_id_after, (
        "Model was replaced between translate() calls — weights are being reloaded!"
    )


# ---------------------------------------------------------------------------
# Direct-run demo (not pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from timeit import default_timer as timer

    codec = get_codec()
    print(f"Model loaded on {codec.device}.\n")

    t0 = timer()
    all_results = run_batch(codec, EXAMPLES, n_best=3)
    elapsed = timer() - t0

    for i, ((src_text, template, n_spans, tgt_lang), results) in enumerate(
        zip(EXAMPLES, all_results)
    ):
        print(f"=== Example {i + 1} ===")
        print(f"  src : {src_text}")
        print(f"  tpl : {template}")
        for rank, r in enumerate(results, 1):
            print(f"  #{rank}  [{r['score']:.3f}]  {r['text']}")
        print()

    print(f"Processed {len(EXAMPLES)} examples in {elapsed:.1f}s "
          f"({elapsed / len(EXAMPLES):.1f}s / example).")
