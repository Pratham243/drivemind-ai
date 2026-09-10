"""NLU router: the single entry point that turns raw text into
(intent, entities, confidence, model_used).

Routing policy:
  1. Run the configured ML classifier (baseline or transformer).
  2. If its confidence is below ``confidence_threshold``, treat the
     utterance as ambiguous and fall back to the LLM layer for structured
     extraction (PHASE 10: "LLM used for ambiguous requests").
  3. Always run deterministic entity extraction, scoped to the resolved
     intent, and merge the ML/LLM entities on top of the rule-based fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.nlu.entity_extractor import extract_entities
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.llm_provider import get_llm_provider


@dataclass
class NLUResult:
    text: str
    intent: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    model_used: str = "baseline"
    used_llm_fallback: bool = False


class NLURouter:
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self._classifier: IntentClassifier | None = None
        self._llm = get_llm_provider()

    def _get_classifier(self) -> IntentClassifier:
        if self._classifier is None:
            self._classifier = IntentClassifier()
        return self._classifier

    def process(self, text: str) -> NLUResult:
        classifier = self._get_classifier()

        if classifier.is_ready:
            intent, confidence = classifier.classify(text)
            model_used = classifier.model_name
        else:
            intent, confidence, model_used = "general_question", 0.0, "untrained"

        used_llm = False
        if confidence < self.confidence_threshold:
            llm_result = self._llm.structured_output(text)
            if llm_result.get("intent"):
                intent = llm_result["intent"]
                confidence = float(llm_result.get("confidence", confidence))
                model_used = f"{model_used}+llm_fallback({self._llm.name})"
                used_llm = True

        entities = extract_entities(text, intent)
        if used_llm:
            # LLM-provided entities take precedence where present.
            llm_entities = llm_result.get("entities", {}) if isinstance(llm_result, dict) else {}
            entities = {**entities, **llm_entities}

        return NLUResult(
            text=text,
            intent=intent,
            confidence=round(confidence, 4),
            entities=entities,
            model_used=model_used,
            used_llm_fallback=used_llm,
        )
