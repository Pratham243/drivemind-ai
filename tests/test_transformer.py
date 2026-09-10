"""Tests for the DistilBERT transformer intent classifier.

Skipped (not failed) when the optional ML dependencies aren't installed
or the transformer hasn't been trained yet — mirroring the same pattern
used for the baseline-model-dependent tests in test_nlu.py/test_agent.py.
Run `pip install -r requirements-ml.txt && python scripts/train_transformer.py`
to exercise these.
"""

from __future__ import annotations

import pytest

from src.config.settings import PROJECT_ROOT, load_yaml_config

try:
    from src.models.transformer import TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False

_MODEL_PATH = PROJECT_ROOT / load_yaml_config("config.yaml")["transformer_model"]["path"]

pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="torch/transformers not installed (pip install -r requirements-ml.txt)",
)


@pytest.fixture(scope="module")
def transformer_model():
    if (
        not (_MODEL_PATH / "model.safetensors").exists()
        and not (_MODEL_PATH / "pytorch_model.bin").exists()
    ):
        pytest.skip("Transformer model not trained. Run scripts/train_transformer.py first.")
    from src.models.transformer import TransformerIntentClassifier

    return TransformerIntentClassifier.load(_MODEL_PATH)


def test_transformer_loads_from_saved_artifacts(transformer_model):
    assert transformer_model.model is not None
    assert transformer_model.tokenizer is not None
    assert len(transformer_model.id2label) > 0


def test_transformer_predicts_known_intents(transformer_model):
    intent, confidence = transformer_model.predict_one("Set the temperature to 22 degrees")
    assert intent == "set_temperature"
    assert 0.0 <= confidence <= 1.0


def test_transformer_predicts_battery_status(transformer_model):
    intent, _ = transformer_model.predict_one("What is my battery level?")
    assert intent == "battery_status"


def test_transformer_batch_predict_matches_predict_one(transformer_model):
    texts = ["Play some jazz", "Navigate to Berlin Brandenburg Airport"]
    batch_preds = transformer_model.predict(texts)
    single_preds = [transformer_model.predict_one(t)[0] for t in texts]
    assert batch_preds == single_preds


def test_transformer_handles_messy_whitespace_and_unicode(transformer_model):
    # Regression guard for the transformer pipeline actually routing text
    # through clean_for_transformer (NFKC + whitespace normalization)
    # before tokenization rather than silently skipping preprocessing.
    intent, _ = transformer_model.predict_one("  Set   the temperature to 22 degrees  ")
    assert intent == "set_temperature"
