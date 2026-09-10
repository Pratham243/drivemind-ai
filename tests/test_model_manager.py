"""Tests for ModelManager: NLU model selection and fallback behavior.

Uses the class directly (not the cached get_model_manager() singleton) so
each test can freely manipulate .settings.nlu_model / ._model without
cross-test interference from functools.lru_cache.
"""

from __future__ import annotations

import pytest

from src.config.settings import PROJECT_ROOT, load_yaml_config
from src.models.model_manager import ModelManager

_baseline_path = PROJECT_ROOT / load_yaml_config("config.yaml")["baseline_model"]["path"]


@pytest.fixture
def manager_with_no_model():
    """A ModelManager instance with its model forcibly unset, bypassing
    whatever is actually trained on disk — tests the untrained guard path
    in isolation.
    """
    manager = ModelManager()
    manager._model = None
    manager.active_model_name = "untrained"
    return manager


def test_defaults_to_baseline_when_trained():
    if not _baseline_path.exists():
        pytest.skip("Baseline model not trained. Run scripts/train_baseline.py first.")
    manager = ModelManager()
    assert manager.active_model_name == "baseline"
    assert manager.is_ready


def test_predict_intent_raises_clearly_when_untrained(manager_with_no_model):
    assert not manager_with_no_model.is_ready
    with pytest.raises(RuntimeError, match="No intent classification model"):
        manager_with_no_model.predict_intent("hello")


def test_falls_back_to_baseline_when_transformer_requested_but_missing():
    """If NLU_MODEL=transformer but no trained transformer artifact
    exists at the configured path, ModelManager must not crash — it
    should log a warning and fall back to the baseline model.
    """
    if not _baseline_path.exists():
        pytest.skip("Baseline model not trained. Run scripts/train_baseline.py first.")

    manager = ModelManager()
    manager.settings.nlu_model = "transformer"

    transformer_path = PROJECT_ROOT / load_yaml_config("config.yaml")["transformer_model"]["path"]
    transformer_trained = (transformer_path / "model.safetensors").exists() or (
        transformer_path / "pytorch_model.bin"
    ).exists()
    if transformer_trained:
        pytest.skip(
            "Transformer is actually trained in this environment; fallback path not exercised."
        )

    manager._load()
    assert manager.active_model_name == "baseline"
    assert manager.is_ready
