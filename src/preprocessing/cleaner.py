"""Text cleaning utilities.

Two distinct cleaning paths are exposed because they have conflicting
goals:

* :func:`clean_for_traditional_ml` — aggressive normalization (lowercase,
  punctuation stripped, whitespace collapsed) appropriate for TF-IDF /
  bag-of-words features, where surface form variation only adds sparsity.

* :func:`clean_for_transformer` — minimal normalization only. Transformer
  tokenizers (WordPiece/BPE) are trained on natural, cased, punctuated
  text; aggressively stripping punctuation or lowercasing throws away
  signal the subword tokenizer and pretrained embeddings rely on.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_for_traditional_ml(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    """Normalize text for TF-IDF / classical ML pipelines."""
    text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT_RE.sub(" ", text)
    return normalize_whitespace(text)


def clean_for_transformer(text: str) -> str:
    """Light normalization only — preserve casing/punctuation for the
    transformer's own tokenizer to consume.
    """
    text = unicodedata.normalize("NFKC", text)
    return normalize_whitespace(text)
