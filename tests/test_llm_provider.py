"""Tests for the LLM provider abstraction, including OpenAIProvider's
JSON-extraction and error-handling logic — mocked, no real network calls
or API key required. Skipped if the optional `openai` package isn't
installed (see requirements-llm.txt).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import openai  # noqa: F401

    OPENAI_INSTALLED = True
except ImportError:
    OPENAI_INSTALLED = False

pytestmark = pytest.mark.skipif(
    not OPENAI_INSTALLED,
    reason="openai package not installed (pip install -r requirements-llm.txt)",
)


def _make_provider():
    from src.nlu.llm_provider import OpenAIProvider

    return OpenAIProvider(api_key="sk-fake-test-key", model="gpt-4o-mini")


def _mock_completion(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_structured_output_extracts_json_from_markdown_wrapped_response():
    provider = _make_provider()
    content = (
        'Sure! ```json\n{"intent": "find_charging_station", '
        '"entities": {"location": "Berlin", "minimum_power_kw": 150}}\n```'
    )
    with patch.object(
        provider._client.chat.completions, "create", return_value=_mock_completion(content)
    ):
        result = provider.structured_output("Find a charger near Berlin with 150kW")

    assert result["intent"] == "find_charging_station"
    assert result["entities"] == {"location": "Berlin", "minimum_power_kw": 150}
    assert result["provider"] == "openai"


def test_structured_output_extracts_bare_json_response():
    provider = _make_provider()
    content = '{"intent": "battery_status", "entities": {}}'
    with patch.object(
        provider._client.chat.completions, "create", return_value=_mock_completion(content)
    ):
        result = provider.structured_output("What's my battery level?")

    assert result["intent"] == "battery_status"


def test_structured_output_degrades_gracefully_on_non_json_response():
    """A malformed/non-JSON LLM response must not crash the pipeline —
    it should fall back to general_question with an explicit error field,
    not raise, so the agent can still produce a response.
    """
    provider = _make_provider()
    content = "I am not sure what you mean, sorry."
    with patch.object(
        provider._client.chat.completions, "create", return_value=_mock_completion(content)
    ):
        result = provider.structured_output("asdkjhasdkjh")

    assert result["intent"] == "general_question"
    assert result["confidence"] == 0.0
    assert "error" in result


def test_generate_passes_system_and_user_messages():
    provider = _make_provider()
    with patch.object(
        provider._client.chat.completions, "create", return_value=_mock_completion("Hello there")
    ) as mock_create:
        result = provider.generate("hi", system="You are helpful")

    assert result == "Hello there"
    _, kwargs = mock_create.call_args
    roles = [m["role"] for m in kwargs["messages"]]
    assert roles == ["system", "user"]


def test_get_llm_provider_selects_openai_when_key_configured(monkeypatch):
    from src.config.settings import get_settings
    from src.nlu.llm_provider import get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key")
    try:
        provider = get_llm_provider()
        assert provider.name == "openai"
    finally:
        get_settings.cache_clear()
        get_llm_provider.cache_clear()
