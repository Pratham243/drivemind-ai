"""LLM abstraction layer.

Design goal (PHASE 10/32): the application must never hard-depend on an
external paid API. ``get_llm_provider()`` returns a working
:class:`MockLLMProvider` whenever no ``OPENAI_API_KEY`` is configured (or
the ``openai`` package isn't installed), so the whole pipeline — including
structured intent extraction and response generation — keeps working in a
fully offline/free demo environment. When a key *is* present,
:class:`OpenAIProvider` is used transparently instead; callers never need
to know which one is active except via ``.name``.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from src.config.settings import get_settings, load_intents

logger = logging.getLogger("drivemind.llm")


class LLMProvider(ABC):
    """Common interface both the mock and real LLM providers implement."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Free-form text generation."""

    @abstractmethod
    def structured_output(self, user_message: str) -> dict[str, Any]:
        """Return a dict with at least ``intent`` and ``entities`` keys,
        used as a fallback NLU path for utterances the ML models are
        unsure about.
        """


class MockLLMProvider(LLMProvider):
    """Deterministic, offline stand-in for a real LLM.

    Implements the same structured-extraction contract using simple
    keyword heuristics over the known intent taxonomy, plus templated
    response generation. This keeps the *agent* and *API* fully
    demonstrable without any external service or API key.
    """

    name = "mock"

    def __init__(self) -> None:
        self._intents = load_intents()

    def generate(self, prompt: str, system: str | None = None) -> str:
        return (
            f"[MockLLMProvider] I understood your request: '{prompt.strip()}'. "
            "(Configure OPENAI_API_KEY to use a real LLM for richer responses.)"
        )

    def structured_output(self, user_message: str) -> dict[str, Any]:
        from src.nlu.entity_extractor import extract_entities

        lowered = user_message.lower()
        best_intent = "general_question"
        # Very small keyword heuristic used only as a last-resort fallback
        # when the trained ML classifier has low confidence.
        keyword_map = {
            "warm": "increase_temperature",
            "cold": "decrease_temperature",
            "cool": "decrease_temperature",
            "degree": "set_temperature",
            "battery": "battery_status",
            "charge": "find_charging_station",
            "charger": "find_charging_station",
            "range": "range_query",
            "navigat": "start_navigation",
            "direction": "start_navigation",
            "drive to": "start_navigation",
            "music": "play_music",
            "song": "next_song",
            "volume": "change_volume",
            "call": "make_phone_call",
            "message": "send_message",
            "text": "send_message",
            "window": "open_window",
            "seat": "seat_adjustment",
            "light": "ambient_lighting",
            "restaurant": "restaurant_search",
            "eat": "restaurant_search",
            "weather": "weather_query",
            "traffic": "traffic_query",
            "park": "find_parking",
            "tire": "tire_pressure",
        }
        for keyword, intent in keyword_map.items():
            if keyword in lowered:
                best_intent = intent
                break

        entities = extract_entities(user_message, best_intent)
        return {
            "intent": best_intent,
            "entities": entities,
            "confidence": 0.55,
            "provider": self.name,
        }


class OpenAIProvider(LLMProvider):
    """Real LLM provider backed by the OpenAI Chat Completions API.

    Only imported/instantiated when an API key is configured — see
    :func:`get_llm_provider`.
    """

    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # imported lazily; optional dependency

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._intents = load_intents()

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.3
        )
        return response.choices[0].message.content or ""

    def structured_output(self, user_message: str) -> dict[str, Any]:
        intent_names = ", ".join(sorted(self._intents.keys()))
        system = (
            "You are an in-car assistant NLU module. Given a user utterance, "
            "respond ONLY with a JSON object of the form "
            '{"intent": "<one of the allowed intents>", "entities": {...}}. '
            f"Allowed intents: {intent_names}. "
            "Use only entity keys relevant to the chosen intent. "
            "Respond with JSON only, no prose."
        )
        try:
            raw = self.generate(user_message, system=system)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            payload = json.loads(match.group(0) if match else raw)
            payload.setdefault("entities", {})
            payload["confidence"] = 0.9
            payload["provider"] = self.name
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI structured_output failed (%s); returning fallback.", exc)
            return {
                "intent": "general_question",
                "entities": {},
                "confidence": 0.0,
                "provider": self.name,
                "error": str(exc),
            }


@functools.lru_cache
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_available:
        try:
            return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)
        except ImportError:
            logger.warning("openai package not installed; falling back to MockLLMProvider.")
    return MockLLMProvider()
