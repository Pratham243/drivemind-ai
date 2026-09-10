"""Train and evaluate the TF-IDF + Logistic Regression baseline intent
classifier on the real generated dataset. Saves the trained model and a
real evaluation report (no fabricated numbers).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_yaml_config  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics, top_confusions  # noqa: E402
from src.evaluation.model_evaluation import get_split, load_annotated_dataframe  # noqa: E402
from src.models.baseline import BaselineIntentClassifier  # noqa: E402


def main() -> None:
    cfg = load_yaml_config("config.yaml")
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["baseline_model"]

    df = load_annotated_dataframe(PROJECT_ROOT / dataset_cfg["annotated_path"])
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    test_df = get_split(df, "test")

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    model = BaselineIntentClassifier(
        max_features=model_cfg["tfidf_max_features"],
        ngram_range=tuple(model_cfg["tfidf_ngram_range"]),
        C=model_cfg["logreg_C"],
        max_iter=model_cfg["logreg_max_iter"],
    )
    model.fit(train_df["text"].tolist(), train_df["intent"].tolist())

    all_labels = sorted(df["intent"].unique().tolist())

    val_preds = model.predict(val_df["text"].tolist())
    val_metrics = compute_classification_metrics(val_df["intent"].tolist(), val_preds, all_labels)
    print(f"Validation accuracy: {val_metrics['accuracy']}  macro-F1: {val_metrics['f1_macro']}")

    test_preds = model.predict(test_df["text"].tolist())
    test_metrics = compute_classification_metrics(
        test_df["intent"].tolist(), test_preds, all_labels
    )
    print(f"Test accuracy: {test_metrics['accuracy']}  macro-F1: {test_metrics['f1_macro']}")

    errors = top_confusions(test_metrics, top_k=10)

    model_path = PROJECT_ROOT / model_cfg["path"]
    model.save(model_path)
    print(f"Saved model to {model_path}")

    report = {
        "model": "tfidf_logreg_baseline",
        "config": model_cfg,
        "validation": val_metrics,
        "test": test_metrics,
        "top_confusions_test": errors,
    }
    report_path = PROJECT_ROOT / "reports" / "evaluation" / "baseline_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
