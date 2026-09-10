"""Cabin comfort tools: windows, seats, ambient lighting.

Kept separate from climate.py (temperature-only) since these map to
distinct intents/entities in the taxonomy (window/seat/color) even
though they are all "comfort" adjustments physically.
"""

from __future__ import annotations

from src.safety.policies import VALID_WINDOWS
from src.tools.base import ToolResult
from src.tools.vehicle import VehicleSimulator


def open_window(vehicle: VehicleSimulator, window: str) -> ToolResult:
    if window not in VALID_WINDOWS:
        return ToolResult.fail(f"Unknown window '{window}'")
    return vehicle.set_window(window, "open")


def close_window(vehicle: VehicleSimulator, window: str) -> ToolResult:
    if window not in VALID_WINDOWS:
        return ToolResult.fail(f"Unknown window '{window}'")
    return vehicle.set_window(window, "closed")


def adjust_seat(vehicle: VehicleSimulator, seat: str, adjustment: str = "adjusted") -> ToolResult:
    if not seat:
        return ToolResult.fail("seat must be provided")
    return vehicle.set_seat(seat, adjustment)


def set_ambient_lighting(vehicle: VehicleSimulator, color: str) -> ToolResult:
    if not color:
        return ToolResult.fail("color must be provided")
    return vehicle.set_ambient_light(color)
