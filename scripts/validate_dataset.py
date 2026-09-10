"""Validate the annotated dataset and produce a data-quality report.

Reads data/annotated/dataset.jsonl, runs schema validation and data
quality analysis, and writes:
  - reports/evaluation/data_quality_report.json
  - reports/figures/class_distribution.png
  - reports/figures/split_distribution.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation.quality import analyze_dataset  # noqa: E402
from src.annotation.validator import validate_dataset  # noqa: E402
from src.config.settings import load_yaml_config  # noqa: E402


def load_annotated(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    cfg = load_yaml_config("config.yaml")["dataset"]
    annotated_path = PROJECT_ROOT / cfg["annotated_path"]
    records = load_annotated(annotated_path)
    df = pd.DataFrame(records)

    print(f"Loaded {len(records)} annotated examples from {annotated_path}")

    # --- Schema / annotation validation ---
    validation_report = validate_dataset(records)
    print(f"Validation: {validation_report.valid}/{validation_report.total} valid records")
    if validation_report.issues:
        print(f"Issue breakdown: {dict(validation_report.issue_counts())}")

    # --- Data quality analysis ---
    quality_report = analyze_dataset(df)
    print(f"Total samples: {quality_report.total_samples}")
    print(f"Duplicates: {quality_report.duplicate_samples}")
    print(f"Missing labels: {quality_report.missing_labels}")
    print(f"Class imbalance ratio (max/min): {quality_report.class_imbalance_ratio}")
    print(f"Avg text length: {quality_report.avg_text_length} chars")

    # --- Save combined report ---
    reports_dir = PROJECT_ROOT / "reports" / "evaluation"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "data_quality_report.json"
    combined = {
        "validation": validation_report.to_dict(),
        "quality": quality_report.to_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {out_path}")

    # --- Figures ---
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.Series(quality_report.class_distribution).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 9))
    counts.plot(kind="barh", ax=ax, color="#3B6EA5")
    ax.set_xlabel("Number of examples")
    ax.set_title("DriveMind AI — Intent Class Distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    split_counts = df["split"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    split_counts.plot(kind="bar", ax=ax, color="#4C9F70")
    ax.set_ylabel("Number of examples")
    ax.set_title("Train / Val / Test Split")
    fig.tight_layout()
    fig.savefig(figures_dir / "split_distribution.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {figures_dir}")

    if validation_report.invalid > 0:
        print(f"WARNING: {validation_report.invalid} invalid records found.")


if __name__ == "__main__":
    main()
