"""Shared classification metrics used by both ML models (baseline &
transformer) so their reports are directly comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, Any]:
    """Compute accuracy, precision/recall/F1 (macro + weighted), and a
    confusion matrix. All values are derived directly from the given
    predictions — nothing here is a placeholder.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision_weighted = precision_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "n_samples": len(y_true),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def top_confusions(metrics: dict[str, Any], top_k: int = 10) -> list[dict[str, Any]]:
    """Extract the largest off-diagonal confusion-matrix cells as
    (true_label, predicted_label, count) for error analysis.
    """
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": labels[i], "predicted": labels[j], "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_k]
