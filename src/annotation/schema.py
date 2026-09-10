"""Annotation schema for DriveMind AI utterances.

Every labeled example in the dataset must conform to :class:`AnnotatedExample`.
This is the single source of truth for what a "valid" training example looks
like, used by both the dataset generator (which produces ground truth) and
the validator (which checks it).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Allowed entity keys and their expected Python types. Used by the validator
# to catch malformed entity values (PHASE 4 requirement).
ENTITY_TYPES: dict[str, type] = {
    "location": str,
    "destination": str,
    "minimum_power_kw": int,
    "temperature": int,
    "amount": str,
    "music_genre": str,
    "artist": str,
    "phone_contact": str,
    "message_body": str,
    "volume": int,
    "direction": str,
    "window": str,
    "seat": str,
    "color": str,
    "restaurant_cuisine": str,
    "tire": str,
    "mode": str,
}


class AnnotatedExample(BaseModel):
    """One fully annotated training example."""

    id: str
    text: str = Field(min_length=1)
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    split: Split
    source: str = "synthetic_template"
    difficulty: Difficulty = Difficulty.MEDIUM

    @field_validator("text")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be empty/whitespace-only")
        return v
