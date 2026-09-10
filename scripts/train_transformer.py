"""Fine-tune and evaluate the DistilBERT intent classifier.

Requires the optional ML dependencies:
    pip install -r requirements-ml.txt

CPU training of distilbert-base-uncased on ~2.7k short utterances for a
few epochs is deliberately kept feasible on a laptop (small max_length,
small batch size) per PHASE 8 ("do not make training impossible on a
normal laptop").
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_yaml_config  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics, top_confusions  # noqa: E402
from src.evaluation.model_evaluation import get_split, load_annotated_dataframe  # noqa: E402


def main() -> None:
    try:
        from src.models.transformer import TransformerIntentClassifier
    except ImportError as exc:
        print(f"ERROR: transformer dependencies not installed ({exc}).")
        print("Run: pip install -r requirements-ml.txt")
        sys.exit(1)

    cfg = load_yaml_config("config.yaml")
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["transformer_model"]

    df = load_annotated_dataframe(PROJECT_ROOT / dataset_cfg["annotated_path"])
    train_df = get_split(df, "train")
    val_df = get_split(df, "val")
    test_df = get_split(df, "test")
    all_labels = sorted(df["intent"].unique().tolist())

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Base model: {model_cfg['base_model']} | device: CPU/GPU auto-detected")

    model = TransformerIntentClassifier(
        base_model=model_cfg["base_model"], max_length=model_cfg["max_length"]
    )

    start = time.time()
    history = model.fit(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["intent"].tolist(),
        val_texts=val_df["text"].tolist(),
        val_labels=val_df["intent"].tolist(),
        epochs=model_cfg["epochs"],
        batch_size=model_cfg["batch_size"],
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=model_cfg["weight_decay"],
        all_labels=all_labels,
    )
    train_time_sec = round(time.time() - start, 1)
    print(f"Training completed in {train_time_sec}s")

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
        "model": "distilbert_transformer",
        "config": model_cfg,
        "training_history": history,
        "train_time_sec": train_time_sec,
        "test": test_metrics,
        "top_confusions_test": errors,
    }
    report_path = PROJECT_ROOT / "reports" / "evaluation" / "transformer_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
