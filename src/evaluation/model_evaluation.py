"""Utilities to load the annotated dataset splits and run model evaluation,
shared by the baseline and transformer training/evaluation scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_annotated_dataframe(path: str | Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)
