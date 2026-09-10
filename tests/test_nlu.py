import pytest

from src.nlu.entity_extractor import extract_entities
from src.nlu.intent_classifier import IntentClassifier
from src.nlu.llm_provider import MockLLMProvider
from src.nlu.router import NLURouter


@pytest.fixture(scope="module")
def classifier():
    clf = IntentClassifier()
    if not clf.is_ready:
        pytest.skip("Baseline model not trained. Run scripts/train_baseline.py first.")
    return clf


def test_intent_classifier_predicts_known_intents(classifier):
    intent, confidence = classifier.classify("Set the temperature to 22 degrees")
    assert intent == "set_temperature"
    assert 0.0 <= confidence <= 1.0


def test_intent_classifier_battery_status(classifier):
    intent, _ = classifier.classify("What's my battery level?")
    assert intent == "battery_status"


def test_entity_extraction_temperature():
    entities = extract_entities("Set the temperature to 22 degrees", intent="set_temperature")
    assert entities.get("temperature") == 22


def test_entity_extraction_location_and_power():
    entities = extract_entities(
        "Find a charging station near Berlin with at least 150 kW",
        intent="find_charging_station",
    )
    assert entities.get("location") == "Berlin"
    assert entities.get("minimum_power_kw") == 150


def test_entity_extraction_destination():
    entities = extract_entities("Navigate to Berlin Brandenburg Airport", intent="start_navigation")
    assert entities.get("destination") == "Berlin Brandenburg Airport"


def test_entity_extraction_music_genre():
    entities = extract_entities("Play some jazz", intent="play_music")
    assert entities.get("music_genre") == "jazz"


def test_entity_extraction_no_false_positive_outside_scope():
    # location should not be extracted for an intent that doesn't use it
    entities = extract_entities("Pause the music please, thanks Berlin", intent="pause_music")
    assert "location" not in entities


def test_mock_llm_provider_structured_output_has_required_keys():
    provider = MockLLMProvider()
    result = provider.structured_output("What's the weather like?")
    assert "intent" in result
    assert "entities" in result
    assert result["provider"] == "mock"


def test_mock_llm_generate_returns_nonempty_string():
    provider = MockLLMProvider()
    text = provider.generate("hello")
    assert isinstance(text, str) and len(text) > 0


def test_nlu_router_end_to_end():
    router = NLURouter()
    result = router.process("Find a charging station near Berlin with at least 150 kW")
    assert result.intent == "find_charging_station"
    assert result.entities.get("location") == "Berlin"
    assert result.entities.get("minimum_power_kw") == 150
