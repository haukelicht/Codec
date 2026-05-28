#!/bin/bash

MODEL="ychenNLP/nllb-200-distilled-1.3B-easyproject"
NLLB_TOKENIZER="facebook/nllb-200-distilled-1.3B"  # <--- Load from the original NLLB checkpoint

# SRC_TEXT="Only France and [ Britain ] backed Fischler's proposal ."
# TEMPLATE="Lediglich Frankreich und Groß Britanien unterstützten Fischers Vorschlag ."

# SRC_TEXT="[A translator] always risks inadvertently introducing [source-language words, grammar, or syntax] into the target-language rendering."
# TEMPLATE="Ein Übersetzer riskiert immer, versehentlich Wörter der Ausgangssprache, Grammatik oder Syntax in die Zielsprache einzuführen."

SRC_TEXT="We want to help [people who live on the streets]."
TEMPLATE="Wir wollen Menschen helfen, die auf der Straße leben."

conda run -n codec --live-stream python example.py \
    -s "${SRC_TEXT}" \
    -t "${TEMPLATE}" \
    -n 1 \
    --model_name_or_path ${MODEL} \
    --tokenizer_path ${NLLB_TOKENIZER}

