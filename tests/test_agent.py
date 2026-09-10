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


def test_scenario_climate_control_mutates_vehicle_state(agent):
    # Regression test: set_climate_mode was previously wired into the tool
    # registry with needs_vehicle=False despite requiring `vehicle` as its
    # first argument, so every climate_control request silently failed
    # with a "missing 1 required positional argument: 'vehicle'" tool
    # execution error. Also regression-covers the entity extractor
    # previously never producing a `mode` entity at all.
    state = agent.handle("Turn on the AC.")
    assert state.intent == "climate_control"
    assert state.entities.get("mode") == "ac_on"
    assert state.selected_tool == "set_climate_mode"
    assert state.safety_status["status"] == "passed"
    assert state.tool_result["success"] is True
    assert agent.vehicle.state.climate_mode == "ac_on"
    assert "couldn't" not in state.final_response.lower()


def test_tool_registry_needs_vehicle_matches_function_signature():
    """Every registered tool's `needs_vehicle` flag must agree with
    whether its callable actually takes `vehicle` as its first parameter
    — a mismatch means the tool crashes (or silently gets no vehicle
    access) the moment the agent actually calls it, exactly like the
    set_climate_mode bug this test guards against.
    """
    import inspect

    from src.agents.tool_registry import TOOL_REGISTRY

    mismatches = []
    for name, (func, needs_vehicle) in TOOL_REGISTRY.items():
        params = list(inspect.signature(func).parameters.keys())
        first_is_vehicle = bool(params) and params[0] == "vehicle"
        if first_is_vehicle != needs_vehicle:
            mismatches.append(name)
    assert not mismatches, f"needs_vehicle mismatch for: {mismatches}"


def test_all_response_templates_render_for_every_tool(agent):
    """Drive one representative, entity-complete utterance per tool
    through the real agent and assert the response is never the generic
    '{tool} succeeded but no template matched' fallback — catches
    template/tool-output key mismatches like set_climate_mode's
    {mode} vs. {climate_mode}, get_tire_pressure's two response shapes,
    and set_ambient_lighting's {color} vs. {ambient_light_color}.
    """
    utterances = [
        "Make it warmer",
        "Make it cooler",
        "Set the temperature to 22 degrees",
        "Turn on the AC",
        "What is my vehicle status",
        "What is my battery level",
        "How far can I drive",
        "What is my tire pressure",
        "Check the front left tire pressure",
        "Find a charging station near Berlin",
        "Find parking near Berlin",
        "Navigate to Berlin Brandenburg Airport",
        "Cancel navigation",
        "What is the traffic like",
        "Play some jazz",
        "Pause the music",
        "Next song",
        "Turn up the volume",
        "Call Mom",
        "Send a message to Sarah",
        "Open the front left window",
        "Close the front left window",
        "Adjust my driver seat",
        "Change the ambient lighting to blue",
        "Find me an Italian restaurant nearby",
        "What is the weather in Berlin",
    ]
    fallbacks = []
    for utterance in utterances:
        state = agent.handle(utterance)
        step = state.steps[0]
        if step.tool_result.get("success") and state.final_response.startswith("Done: {"):
            fallbacks.append((utterance, state.final_response))
    assert not fallbacks, f"Generic fallback response for: {fallbacks}"
