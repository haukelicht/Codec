"""
codec — bracket-constrained span-annotation translation.

Public API::

    from src.codec import Codec

    codec = Codec("ychenNLP/nllb-200-distilled-1.3B-easyproject")
    results = codec.translate(
        src_text="[A translator] always risks [source-language influence].",
        template="Ein Übersetzer riskiert immer den Einfluss der Ausgangssprache.",
        n_spans=2,
        tgt_lang="deu_Latn",
    )
"""
from .codec import Codec

__all__ = ["Codec"]
