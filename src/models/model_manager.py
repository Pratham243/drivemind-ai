"""Selects and lazily loads the configured intent-classification model.

Reads ``NLU_MODEL`` from settings (``baseline`` | ``transformer``) and
loads the corresponding trained artifact. The transformer path is only
attempted if the optional dependencies + trained artifact are present;
otherwise it logs a warning and falls back to the baseline so the system
never hard-fails just because a heavy dependency is missing.
"""

from __future__ import annotations

import functools
import logging

from src.config.settings import PROJECT_ROOT, get_settings, load_yaml_config
from src.models.baseline import BaselineIntentClassifier

logger = logging.getLogger("drivemind.model_manager")


class ModelManager:
    """Loads and exposes a single active intent classifier."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.config = load_yaml_config("config.yaml")
        self.active_model_name: str = "baseline"
        self._model = None
        self._load()

    def _load(self) -> None:
        requested = self.settings.nlu_model
        baseline_path = PROJECT_ROOT / self.config["baseline_model"]["path"]

        if requested == "transformer":
            transformer_path = PROJECT_ROOT / self.config["transformer_model"]["path"]
            try:
                from src.models.transformer import TransformerIntentClassifier

                self._model = TransformerIntentClassifier.load(transformer_path)
                self.active_model_name = "transformer"
                logger.info("Loaded transformer intent classifier from %s", transformer_path)
                return
            except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
                logger.warning(
                    "Could not load transformer model (%s). Falling back to baseline.", exc
                )

        if baseline_path.exists():
            self._model = BaselineIntentClassifier.load(baseline_path)
            self.active_model_name = "baseline"
            logger.info("Loaded baseline intent classifier from %s", baseline_path)
        else:
            logger.warning(
                "No trained baseline model found at %s. Run scripts/train_baseline.py first. "
                "Predictions will raise until a model is trained.",
                baseline_path,
            )
            self._model = None
            self.active_model_name = "untrained"

    def predict_intent(self, text: str) -> tuple[str, float]:
        if self._model is None:
            raise RuntimeError(
                "No intent classification model is loaded. Run scripts/train_baseline.py."
            )
        return self._model.predict_one(text)

    @property
    def is_ready(self) -> bool:
        return self._model is not None


@functools.lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager()
