"""Maps tool names (as produced by the NLU/intent layer) to callables.

Centralizing this mapping means the agent controller never needs to know
which tools are stateful (need the vehicle) vs. stateless (pure lookups
like weather/restaurants) — :func:`execute_tool` introspects each
function's signature and forwards only the entities it actually accepts.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from src.tools import (
    charging,
    climate,
    comfort,
    communication,
    media,
    navigation,
    restaurant,
    weather,
)
from src.tools.base import ToolResult
from src.tools.vehicle import VehicleSimulator

# tool_name -> (callable, needs_vehicle)
TOOL_REGISTRY: dict[str, tuple[Callable, bool]] = {
    "get_vehicle_status": (lambda vehicle: vehicle.get_status(), True),
    "get_battery_level": (lambda vehicle: vehicle.get_battery_level(), True),
    "get_range": (lambda vehicle: vehicle.get_range(), True),
    "get_tire_pressure": (lambda vehicle, tire="all": vehicle.get_tire_pressure(tire), True),
    "set_temperature": (climate.set_temperature, True),
    "increase_temperature": (climate.increase_temperature, True),
    "decrease_temperature": (climate.decrease_temperature, True),
    "set_climate_mode": (climate.set_climate_mode, True),
    "find_charging_stations": (charging.find_charging_stations, False),
    "find_parking": (navigation.find_parking, False),
    "start_navigation": (navigation.start_navigation, True),
    "cancel_navigation": (navigation.cancel_navigation, True),
    "search_navigation": (navigation.search_navigation, True),
    "get_traffic": (navigation.get_traffic, False),
    "play_music": (media.play_music, True),
    "pause_music": (media.pause_music, True),
    "next_song": (media.next_song, True),
    "change_volume": (media.change_volume, True),
    "make_phone_call": (communication.make_phone_call, False),
    "send_message": (communication.send_message, False),
    "open_window": (comfort.open_window, True),
    "close_window": (comfort.close_window, True),
    "adjust_seat": (comfort.adjust_seat, True),
    "set_ambient_lighting": (comfort.set_ambient_lighting, True),
    "search_restaurants": (restaurant.search_restaurants, False),
    "get_weather": (weather.get_weather, False),
}


def execute_tool(
    tool_name: str, entities: dict[str, Any], vehicle: VehicleSimulator
) -> tuple[ToolResult, dict[str, Any]]:
    """Execute a registered tool, forwarding only the entity keys the
    tool's signature accepts. Returns (result, arguments_actually_used).
    """
    if tool_name not in TOOL_REGISTRY:
        return ToolResult.fail(f"Unknown tool '{tool_name}'"), {}

    func, needs_vehicle = TOOL_REGISTRY[tool_name]
    sig_params = inspect.signature(func).parameters
    kwargs = {k: v for k, v in entities.items() if k in sig_params}

    try:
        result = func(vehicle, **kwargs) if needs_vehicle else func(**kwargs)
    except Exception as exc:  # noqa: BLE001 - tool errors must not crash the agent
        return ToolResult.fail(f"Tool execution error: {exc}"), kwargs

    return result, kwargs
