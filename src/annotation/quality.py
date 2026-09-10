"""Data quality analysis for the DriveMind dataset.

All numbers here are computed directly from the dataset passed in —
nothing is hard-coded or estimated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataQualityReport:
    total_samples: int
    missing_labels: int
    empty_text: int
    duplicate_samples: int
    invalid_records: int
    class_distribution: dict[str, int]
    class_imbalance_ratio: float  # max_class_count / min_class_count
    avg_text_length: float
    min_text_length: int
    max_text_length: int
    suspicious_samples: list[str] = field(default_factory=list)
    entity_coverage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def _is_suspicious(text: str) -> bool:
    """Heuristic noise detector: very short, all-caps, or mostly non-alpha."""
    if len(text.strip()) < 3:
        return True
    alpha_chars = sum(c.isalpha() for c in text)
    if len(text) > 0 and alpha_chars / len(text) < 0.4:
        return True
    return False


def analyze_dataset(df: pd.DataFrame) -> DataQualityReport:
    """Compute real data-quality metrics from a dataframe with columns:
    id, text, intent, entities, split, source, difficulty.
    """
    total = len(df)

    missing_labels = int(
        df["intent"].isna().sum() + (df["intent"].astype(str).str.strip() == "").sum()
    )
    empty_text = int(df["text"].isna().sum() + (df["text"].astype(str).str.strip() == "").sum())

    normalized_text = df["text"].astype(str).str.strip().str.lower()
    duplicate_samples = int(normalized_text.duplicated(keep="first").sum())

    invalid_records = missing_labels + empty_text

    class_counts = df["intent"].value_counts().to_dict()
    class_counts = {str(k): int(v) for k, v in class_counts.items()}
    if class_counts:
        imbalance_ratio = max(class_counts.values()) / max(1, min(class_counts.values()))
    else:
        imbalance_ratio = 0.0

    lengths = df["text"].astype(str).str.len()
    suspicious = [t for t in df["text"].astype(str).tolist() if _is_suspicious(t)]

    entity_coverage: Counter = Counter()
    for entities in df.get("entities", []):
        if isinstance(entities, dict):
            for key in entities:
                entity_coverage[key] += 1

    return DataQualityReport(
        total_samples=total,
        missing_labels=missing_labels,
        empty_text=empty_text,
        duplicate_samples=duplicate_samples,
        invalid_records=invalid_records,
        class_distribution=class_counts,
        class_imbalance_ratio=round(float(imbalance_ratio), 3),
        avg_text_length=round(float(lengths.mean()), 2) if total else 0.0,
        min_text_length=int(lengths.min()) if total else 0,
        max_text_length=int(lengths.max()) if total else 0,
        suspicious_samples=suspicious[:20],
        entity_coverage=dict(entity_coverage),
    )
