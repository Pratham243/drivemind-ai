"""Lightweight tokenization utilities for the traditional-ML path.

The transformer path uses its own tokenizer (loaded from the pretrained
model via Hugging Face) and does not use this module.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def simple_tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, keeping contractions intact."""
    return _TOKEN_RE.findall(text.lower())
