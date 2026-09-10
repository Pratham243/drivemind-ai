"""Intent classification front-end used by the NLU router.

Thin wrapper around :class:`~src.models.model_manager.ModelManager` — kept
as its own module so the NLU layer (this package) depends on an interface
rather than reaching into ``src.models`` directly, which keeps the model
implementation swappable (baseline/transformer/future models) without
touching router logic.
"""

from __future__ import annotations

from src.models.model_manager import get_model_manager


class IntentClassifier:
    def __init__(self) -> None:
        self._manager = get_model_manager()

    def classify(self, text: str) -> tuple[str, float]:
        """Return (intent, confidence) for the given utterance."""
        return self._manager.predict_intent(text)

    @property
    def model_name(self) -> str:
        return self._manager.active_model_name

    @property
    def is_ready(self) -> bool:
        return self._manager.is_ready
