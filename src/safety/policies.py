"""Safety policy definitions.

These are plain data (not code) deliberately, so the rules are easy to
audit and change without touching validation logic. Loaded from
configs/config.yaml where possible (values that are also useful for the
UI/vehicle defaults), with additional structural policies defined here.
"""

from __future__ import annotations

from src.config.settings import load_yaml_config

_safety_cfg = load_yaml_config("config.yaml")["safety"]

TEMPERATURE_MIN_C = _safety_cfg["temperature_min_c"]
TEMPERATURE_MAX_C = _safety_cfg["temperature_max_c"]
VOLUME_MIN = _safety_cfg["volume_min"]
VOLUME_MAX = _safety_cfg["volume_max"]
MAX_SPEED_FOR_WINDOW_OPEN_KMH = _safety_cfg["max_speed_for_window_open_kmh"]

VALID_WINDOWS = {
    "front left window",
    "front right window",
    "rear left window",
    "rear right window",
    "all windows",
    "driver window",
    "passenger window",
}

VALID_WINDOW_POSITIONS = {"open", "closed"}

VALID_TIRES = {"front left", "front right", "rear left", "rear right", "all"}

# Every tool the agent is allowed to call, and the required argument names
# it must be invoked with. A tool call naming an unknown tool, or missing a
# required argument, is rejected before it ever reaches vehicle state.
REQUIRED_TOOL_ARGS: dict[str, list[str]] = {
    "get_vehicle_status": [],
    "get_battery_level": [],
    "get_range": [],
    "get_tire_pressure": [],
    "set_temperature": ["temperature"],
    "increase_temperature": [],
    "decrease_temperature": [],
    "set_climate_mode": ["mode"],
    "find_charging_stations": [],  # location defaults to "your current location"
    "find_parking": [],  # location defaults to "your current location"
    "start_navigation": ["destination"],
    "cancel_navigation": [],
    "search_navigation": [],
    "get_traffic": [],
    "play_music": [],
    "pause_music": [],
    "next_song": [],
    "change_volume": [],
    "make_phone_call": ["phone_contact"],
    "send_message": ["phone_contact"],
    "open_window": ["window"],
    "close_window": ["window"],
    "adjust_seat": ["seat"],
    "set_ambient_lighting": ["color"],
    "search_restaurants": [],
    "get_weather": [],
}
