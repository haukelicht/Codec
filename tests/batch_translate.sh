#!/usr/bin/env bash

REPO_ROOT="$(git -C "$(pwd)" rev-parse --show-toplevel)"
TRANSLATE="$REPO_ROOT/src/codec/translate.py"

# test.jsonl layout:
#   text        — English source sentence
#   source_text — original-language translation (used as the target template)
#   label       — doccano annotations on the English text
#   lang        — ISO-2 target language code (varies per row: de, fr, pt, da, …)

conda run -n codec --live-stream \
    env PYTHONPATH="$REPO_ROOT" \
    python "$TRANSLATE" \
        --input_path  "test.jsonl" \
        --text_key     text \
        --label_key    label \
        --src_lang     en \
        --lang_col     lang \
        --lang_format  iso2 \
        --template_key source_text \
        --output_label_key source_label \
        --model_name_or_path ychenNLP/nllb-200-distilled-1.3B-easyproject \
        --tokenizer_path     facebook/nllb-200-distilled-1.3B \
        --batch_size 32 \
        --max_joint_spans 2 \
        --search_mode 1 --future_steps 1 \
        --output_path "output.jsonl"
        # --num_beams   10 \
