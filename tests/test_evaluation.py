"""Tests for src/evaluation/metrics.py and src/evaluation/agent_evaluation.py.

These are exercised end-to-end by scripts/train_baseline.py and
scripts/evaluate.py, but had no direct unit coverage — added during an
engineering audit alongside the other test-coverage gaps it found.
"""

from __future__ import annotations

import pytest

from src.evaluation.agent_evaluation import AgentEvalCase, evaluate_agent
from src.evaluation.metrics import compute_classification_metrics, top_confusions
from src.models.model_manager import get_model_manager


def test_compute_classification_metrics_perfect_predictions():
    labels = ["a", "b", "c"]
    y_true = ["a", "b", "c", "a", "b"]
    y_pred = ["a", "b", "c", "a", "b"]
    metrics = compute_classification_metrics(y_true, y_pred, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["n_samples"] == 5


def test_compute_classification_metrics_known_error_rate():
    labels = ["a", "b"]
    y_true = ["a", "a", "a", "a", "b", "b", "b", "b"]
    y_pred = ["a", "a", "a", "b", "b", "b", "b", "b"]  # 1 wrong out of 8
    metrics = compute_classification_metrics(y_true, y_pred, labels)
    assert metrics["accuracy"] == pytest.approx(7 / 8, abs=1e-9)
    assert metrics["n_samples"] == 8
    cm = metrics["confusion_matrix"]
    assert cm == [[3, 1], [0, 4]]  # true a: 3 correct, 1 predicted b; true b: 4 correct


def test_top_confusions_extracts_and_ranks_off_diagonal_cells():
    labels = ["a", "b", "c"]
    metrics = {
        "labels": labels,
        "confusion_matrix": [
            [5, 2, 0],  # true a: 2 predicted as b
            [0, 5, 1],  # true b: 1 predicted as c
            [0, 0, 5],
        ],
    }
    confusions = top_confusions(metrics, top_k=10)
    assert confusions[0] == {"true": "a", "predicted": "b", "count": 2}
    assert confusions[1] == {"true": "b", "predicted": "c", "count": 1}
    assert len(confusions) == 2  # only off-diagonal, nonzero cells


def test_top_confusions_respects_top_k():
    labels = ["a", "b", "c", "d"]
    metrics = {
        "labels": labels,
        "confusion_matrix": [
            [0, 4, 3, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    }
    confusions = top_confusions(metrics, top_k=2)
    assert len(confusions) == 2
    assert confusions[0]["count"] == 4


@pytest.fixture(scope="module")
def _require_trained_model():
    if not get_model_manager().is_ready:
        pytest.skip("Baseline model not trained. Run scripts/train_baseline.py first.")


def test_evaluate_agent_reports_success_for_valid_case(_require_trained_model):
    cases = [
        AgentEvalCase(
            "What's my battery level?",
            "battery_status",
            "get_battery_level",
        ),
    ]
    report = evaluate_agent(cases)
    assert report.n_cases == 1
    assert report.intent_accuracy == 1.0
    assert report.task_success_rate == 1.0
    assert report.failures == []


def test_evaluate_agent_reports_failure_for_wrong_expected_tool(_require_trained_model):
    cases = [
        AgentEvalCase(
            "What's my battery level?",
            "battery_status",
            "start_navigation",  # deliberately wrong
        ),
    ]
    report = evaluate_agent(cases)
    assert report.tool_selection_accuracy == 0.0
    assert len(report.failures) == 1
    assert "tool:" in report.failures[0]["problems"][0]


def test_evaluate_agent_scores_safety_rejection_case_correctly(_require_trained_model):
    cases = [
        AgentEvalCase(
            "Set temperature to 100 degrees.",
            "set_temperature",
            "set_temperature",
            expect_safety_rejection=True,
        ),
    ]
    report = evaluate_agent(cases)
    assert report.safety_rejection_accuracy == 1.0
    assert report.task_success_rate == 1.0  # correctly-rejected unsafe call counts as success
