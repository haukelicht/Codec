#!/usr/bin/env bash

REPO_ROOT="$(git -C "$(pwd)" rev-parse --show-toplevel)"
VALIDATE="$REPO_ROOT/src/codec/validate.py"

# test.jsonl layout:
#   text        — English source sentence
#   source_text — original-language translation (used as the target template)
#   label       — doccano annotations on the English text
#   lang        — ISO-2 target language code (varies per row: de, fr, pt, da, …)

MTMODEL="XIE2021/nllb-200-3.3B-easyproject" #"ychenNLP/nllb-200-distilled-1.3B-easyproject"
TOKENIZER="facebook/nllb-200-distilled-600M" # see https://huggingface.co/XIE2021/nllb-200-3.3B-easyproject#code
BSMODEL="xlm-roberta-large" # for BERTScore matching

conda run -n codec --live-stream \
    env PYTHONPATH="$REPO_ROOT" \
    python "$VALIDATE" \
        --input_path         "output2.jsonl" \
        --output_path        "validation.xlsx" \
        --model_name_or_path "$MTMODEL" \
        --tokenizer_path     "$TOKENIZER" \
        --bertscore_model    "$BSMODEL" \
        --verbose --log-level INFO

echo "Done!"
