"""End-to-end evaluation script (PHASE 19/20).

Produces, from real runs only:
  - reports/evaluation/model_comparison.json  (baseline vs transformer, if trained)
  - reports/evaluation/agent_evaluation.json  (agent-level metrics)
  - reports/evaluation/error_analysis.json    (concrete misclassified examples)
  - reports/figures/model_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_yaml_config  # noqa: E402
from src.evaluation.agent_evaluation import AgentEvalCase, evaluate_agent  # noqa: E402
from src.evaluation.model_evaluation import get_split, load_annotated_dataframe  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def build_agent_eval_cases() -> list[AgentEvalCase]:
    """The 10 canonical scenarios from PHASE 29, plus a few extra edge
    cases, used as the agent-level evaluation set.
    """
    return [
        AgentEvalCase("Make it warmer.", "increase_temperature", "increase_temperature"),
        AgentEvalCase(
            "Set the temperature to 22 degrees.",
            "set_temperature",
            "set_temperature",
            {"temperature": 22},
        ),
        AgentEvalCase("What's my battery level?", "battery_status", "get_battery_level"),
        AgentEvalCase(
            "Find a charging station near Berlin with at least 150 kW.",
            "find_charging_station",
            "find_charging_stations",
            {"location": "Berlin", "minimum_power_kw": 150},
        ),
        AgentEvalCase(
            "Navigate to Berlin Brandenburg Airport.",
            "start_navigation",
            "start_navigation",
            {"destination": "Berlin Brandenburg Airport"},
        ),
        AgentEvalCase("Play some jazz.", "play_music", "play_music", {"music_genre": "jazz"}),
        AgentEvalCase(
            "Set temperature to 100 degrees.",
            "set_temperature",
            "set_temperature",
            expect_safety_rejection=True,
        ),
        AgentEvalCase(
            "Find me an Italian restaurant nearby.", "restaurant_search", "search_restaurants"
        ),
        AgentEvalCase("How far can I drive?", "range_query", "get_range"),
        AgentEvalCase(
            "What's the weather in Munich?", "weather_query", "get_weather", {"location": "Munich"}
        ),
        AgentEvalCase("Pause the music.", "pause_music", "pause_music"),
        AgentEvalCase("Turn down the volume.", "change_volume", "change_volume"),
        AgentEvalCase("Call Mom.", "make_phone_call", "make_phone_call", {"phone_contact": "Mom"}),
        AgentEvalCase(
            "Open the front left window.",
            "open_window",
            "open_window",
            {"window": "front left window"},
        ),
        AgentEvalCase("Cancel navigation.", "cancel_navigation", "cancel_navigation"),
        AgentEvalCase(
            "Set volume to 500.", "change_volume", "change_volume", expect_safety_rejection=True
        ),
    ]


def run_agent_evaluation() -> dict:
    cases = build_agent_eval_cases()
    report = evaluate_agent(cases)
    print(
        f"Agent eval: intent_acc={report.intent_accuracy} tool_acc={report.tool_selection_accuracy} "
        f"entity_acc={report.entity_extraction_accuracy} task_success={report.task_success_rate} "
        f"safety_acc={report.safety_rejection_accuracy} avg_latency_ms={report.avg_latency_ms}"
    )
    out = REPORTS_DIR / "agent_evaluation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved {out}")
    return report.to_dict()


def run_model_comparison() -> dict | None:
    baseline_path = REPORTS_DIR / "baseline_metrics.json"
    transformer_path = REPORTS_DIR / "transformer_metrics.json"

    if not baseline_path.exists():
        print("No baseline_metrics.json found — run scripts/train_baseline.py first.")
        return None

    with open(baseline_path) as f:
        baseline = json.load(f)

    comparison = {"baseline": baseline["test"]}

    if transformer_path.exists():
        with open(transformer_path) as f:
            transformer = json.load(f)
        comparison["transformer"] = transformer["test"]
    else:
        print(
            "No transformer_metrics.json found — transformer comparison skipped "
            "(run scripts/train_transformer.py to include it)."
        )

    out = REPORTS_DIR / "model_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved {out}")

    # --- figure ---
    models = list(comparison.keys())
    metrics_to_plot = ["accuracy", "f1_macro", "f1_weighted"]
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.25
    import numpy as np

    x = np.arange(len(metrics_to_plot))
    for i, model_name in enumerate(models):
        values = [comparison[model_name][m] for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=model_name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.05)
    ax.set_title("DriveMind AI — Model Comparison (test set)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    return comparison


def run_error_analysis() -> dict:
    """Real misclassified examples pulled from the baseline model's test
    predictions (and transformer's, if available) — not fabricated.
    """
    from src.models.baseline import BaselineIntentClassifier

    cfg = load_yaml_config("config.yaml")
    df = load_annotated_dataframe(PROJECT_ROOT / cfg["dataset"]["annotated_path"])
    test_df = get_split(df, "test")

    baseline_path = PROJECT_ROOT / cfg["baseline_model"]["path"]
    errors = {"baseline_errors": [], "note": ""}
    if baseline_path.exists():
        model = BaselineIntentClassifier.load(baseline_path)
        preds = model.predict(test_df["text"].tolist())
        for text, true_intent, pred_intent, difficulty in zip(
            test_df["text"], test_df["intent"], preds, test_df["difficulty"]
        ):
            if true_intent != pred_intent:
                errors["baseline_errors"].append(
                    {
                        "text": text,
                        "true_intent": true_intent,
                        "predicted_intent": pred_intent,
                        "difficulty": difficulty,
                    }
                )
    errors["note"] = (
        f"{len(errors['baseline_errors'])} misclassified examples out of {len(test_df)} "
        "test examples for the baseline model."
    )
    print(errors["note"])

    out = REPORTS_DIR / "error_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    print(f"Saved {out}")
    return errors


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_model_comparison()
    run_agent_evaluation()
    run_error_analysis()


if __name__ == "__main__":
    main()
