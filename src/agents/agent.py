"""Agent controller — the orchestration core of DriveMind AI.

    User message
      -> NLU (intent + entities)
      -> Planner (tool selection, multi-step split)
      -> Safety validation
      -> Tool execution
      -> Response generation

Every call to :meth:`AgentController.handle` returns a fully populated
:class:`AgentState`, which is both the API response payload and the trace
shown in the Streamlit debug panel.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.planner import split_into_subrequests, tool_for_intent
from src.agents.state import AgentState, StepTrace
from src.agents.tool_registry import execute_tool
from src.nlu.llm_provider import get_llm_provider
from src.nlu.router import NLURouter
from src.safety.validator import SafetyValidator
from src.tools.vehicle import VehicleSimulator, get_vehicle

logger = logging.getLogger("drivemind.agent")

_RESPONSE_TEMPLATES: dict[str, str] = {
    "get_vehicle_status": "Here's your vehicle status: battery {battery_pct}%, "
    "temperature {temperature_c}°C, speed {speed_kmh} km/h, range {range_km} km.",
    "get_battery_level": "Your battery is at {battery_pct}%.",
    "get_range": "You have {range_km} km of range remaining.",
    "set_temperature": "Temperature set to {temperature_c}°C.",
    "increase_temperature": "Increased the temperature to {temperature_c}°C.",
    "decrease_temperature": "Decreased the temperature to {temperature_c}°C.",
    "set_climate_mode": "Climate mode set to {climate_mode}.",
    "find_charging_stations": "Found {count} charging station(s) near {location}.",
    "find_parking": "Found {available_spots} parking spot(s) near {location} ({nearest_garage}).",
    "start_navigation": "Starting navigation to {destination} — ETA {eta_minutes} minutes "
    "({distance_km} km).",
    "cancel_navigation": "Navigation cancelled.",
    "search_navigation": "Navigation info retrieved.",
    "get_traffic": "Traffic near {location} is {traffic_level} (delay ~{delay_minutes} min).",
    "play_music": "Now playing {track}.",
    "pause_music": "Music paused.",
    "next_song": "Skipped to {track}.",
    "change_volume": "Volume set to {volume}.",
    "make_phone_call": "Calling {contact}...",
    "send_message": "Message sent to {contact}.",
    "open_window": "Opened the {window}.",
    "close_window": "Closed the {window}.",
    "adjust_seat": "Adjusted {seat}.",
    "set_ambient_lighting": "Ambient lighting set to {ambient_light_color}.",
    "search_restaurants": "Found {count} {cuisine} restaurant(s) near {location}.",
    "get_weather": "Weather in {location}: {condition}, {temperature_c}°C.",
}


class AgentController:
    def __init__(
        self,
        vehicle: VehicleSimulator | None = None,
        router: NLURouter | None = None,
        safety: SafetyValidator | None = None,
    ) -> None:
        self.vehicle = vehicle or get_vehicle()
        self.router = router or NLURouter()
        self.safety = safety or SafetyValidator()
        self.llm = get_llm_provider()

    def handle(self, user_message: str) -> AgentState:
        state = AgentState(user_message=user_message)
        sub_requests = split_into_subrequests(user_message)

        for sub_message in sub_requests:
            try:
                step = self._process_step(sub_message)
                state.steps.append(step)
            except Exception as exc:  # noqa: BLE001 - never let one bad step crash the request
                logger.exception("Error processing sub-request '%s'", sub_message)
                state.errors.append(f"Failed to process '{sub_message}': {exc}")

        state.final_response = self._compose_final_response(state)
        return state

    def _process_step(self, sub_message: str) -> StepTrace:
        nlu_result = self.router.process(sub_message)
        tool_name = tool_for_intent(nlu_result.intent)

        tool_result_dict: dict[str, Any] = {}
        safety_dict: dict[str, Any] = {"status": "not_applicable", "reason": None}
        arguments_used: dict[str, Any] = {}

        if tool_name:
            safety_result = self.safety.validate(
                tool_name, nlu_result.entities, current_speed_kmh=self.vehicle.state.speed_kmh
            )
            safety_dict = safety_result.to_dict()

            if safety_result.passed:
                tool_result, arguments_used = execute_tool(
                    tool_name, nlu_result.entities, self.vehicle
                )
                tool_result_dict = tool_result.to_dict()
            else:
                tool_result_dict = {"success": False, "data": {}, "error": safety_result.reason}

        return StepTrace(
            sub_message=sub_message,
            intent=nlu_result.intent,
            confidence=nlu_result.confidence,
            entities=nlu_result.entities,
            selected_tool=tool_name,
            tool_arguments=arguments_used,
            safety=safety_dict,
            tool_result=tool_result_dict,
            model_used=nlu_result.model_used,
        )

    def _compose_final_response(self, state: AgentState) -> str:
        sentences: list[str] = []
        for step in state.steps:
            sentences.append(self._response_for_step(step))
        if not sentences:
            sentences.append("I couldn't process your request.")
        return " ".join(sentences)

    def _response_for_step(self, step: StepTrace) -> str:
        if step.selected_tool is None:
            # No tool for this intent (e.g. general_question) -> LLM/templated reply.
            return self.llm.generate(step.sub_message)

        if not step.tool_result.get("success", False):
            reason = step.tool_result.get("error", "an unknown error")
            return f"Sorry, I couldn't do that: {reason}."

        data = step.tool_result.get("data", {})

        # get_tire_pressure returns one of two shapes depending on whether a
        # specific tire was requested (`tire`+`pressure_bar`) or all tires
        # were (`tire_pressure_bar`), so it can't use a single static
        # .format() template — handle it explicitly instead.
        if step.selected_tool == "get_tire_pressure":
            if "tire_pressure_bar" in data:
                parts = ", ".join(
                    f"{k.replace('_', ' ')}: {v} bar" for k, v in data["tire_pressure_bar"].items()
                )
                return f"Tire pressure — {parts}."
            return f"Tire pressure for {data.get('tire')}: {data.get('pressure_bar')} bar."

        template = _RESPONSE_TEMPLATES.get(step.selected_tool)
        if template is None:
            return f"Done: {data}."
        try:
            return template.format(**data)
        except (KeyError, IndexError):
            return f"Done: {data}."
