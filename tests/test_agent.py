import pytest

from src.agents.agent import AgentController
from src.tools.vehicle import VehicleSimulator


@pytest.fixture
def agent():
    vehicle = VehicleSimulator()  # fresh, non-shared instance per test
    controller = AgentController(vehicle=vehicle)
    if not controller.router._get_classifier().is_ready:
        pytest.skip("Baseline model not trained. Run scripts/train_baseline.py first.")
    return controller


def test_scenario_increase_temperature(agent):
    state = agent.handle("Make it warmer.")
    assert state.intent == "increase_temperature"
    assert state.selected_tool == "increase_temperature"
    assert state.safety_status["status"] == "passed"


def test_scenario_set_temperature_with_entity(agent):
    state = agent.handle("Set the temperature to 22 degrees.")
    assert state.intent == "set_temperature"
    assert state.entities.get("temperature") == 22
    assert state.tool_result["data"]["temperature_c"] == 22


def test_scenario_battery_status(agent):
    state = agent.handle("What's my battery level?")
    assert state.intent == "battery_status"
    assert "battery_pct" in state.tool_result["data"]


def test_scenario_charging_station_with_entities(agent):
    state = agent.handle("Find a charging station near Berlin with at least 150 kW.")
    assert state.intent == "find_charging_station"
    assert state.entities.get("location") == "Berlin"
    assert state.entities.get("minimum_power_kw") == 150
    assert state.tool_result["success"]


def test_scenario_start_navigation(agent):
    state = agent.handle("Navigate to Berlin Brandenburg Airport.")
    assert state.intent == "start_navigation"
    assert state.entities.get("destination") == "Berlin Brandenburg Airport"
    assert state.tool_result["data"]["navigation_active"] is True


def test_scenario_play_music_with_genre(agent):
    state = agent.handle("Play some jazz.")
    assert state.intent == "play_music"
    assert state.entities.get("music_genre") == "jazz"


def test_scenario_multi_step_battery_and_charging(agent):
    state = agent.handle("What's my battery level and find the nearest charging station.")
    assert len(state.steps) == 2
    intents = {s.intent for s in state.steps}
    assert "battery_status" in intents
    assert "find_charging_station" in intents


def test_scenario_safety_rejection_extreme_temperature(agent):
    state = agent.handle("Set temperature to 100 degrees.")
    assert state.intent == "set_temperature"
    assert state.safety_status["status"] == "rejected"
    assert "unsafe" not in state.final_response.lower()  # user-safe message, no internal jargon
    assert "couldn't" in state.final_response.lower()


def test_scenario_restaurant_search(agent):
    state = agent.handle("Find me an Italian restaurant nearby.")
    assert state.intent == "restaurant_search"


def test_scenario_range_query(agent):
    state = agent.handle("How far can I drive?")
    assert state.intent == "range_query"
    assert "range_km" in state.tool_result["data"]


def test_agent_never_raises_on_garbage_input(agent):
    state = agent.handle("asdkjhaskjdh 12903 !!!")
    assert state.final_response  # should still produce *some* response
