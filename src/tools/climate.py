"""Climate control tools."""

from __future__ import annotations

from src.safety.policies import TEMPERATURE_MAX_C, TEMPERATURE_MIN_C
from src.tools.base import ToolResult
from src.tools.vehicle import VehicleSimulator

TEMP_STEP = 2


def set_temperature(vehicle: VehicleSimulator, temperature: int) -> ToolResult:
    if not isinstance(temperature, (int, float)):
        return ToolResult.fail("temperature must be a number")
    return vehicle.set_temperature(int(temperature))


def increase_temperature(vehicle: VehicleSimulator, amount: str | None = None) -> ToolResult:
    step = TEMP_STEP if amount != "a lot" else TEMP_STEP * 2
    # Clamp to the same safe range enforced for explicit set_temperature
    # calls — relative adjustments must not be able to sidestep the limit.
    new_value = min(vehicle.state.temperature_c + step, TEMPERATURE_MAX_C)
    return vehicle.set_temperature(new_value)


def decrease_temperature(vehicle: VehicleSimulator, amount: str | None = None) -> ToolResult:
    step = TEMP_STEP if amount != "a lot" else TEMP_STEP * 2
    new_value = max(vehicle.state.temperature_c - step, TEMPERATURE_MIN_C)
    return vehicle.set_temperature(new_value)


def set_climate_mode(vehicle: VehicleSimulator, mode: str) -> ToolResult:
    valid_modes = {"ac_on", "ac_off", "fan_on", "fan_off", "heater_on", "heater_off"}
    normalized = mode.lower().replace(" ", "_")
    if normalized not in valid_modes:
        # Accept freeform natural phrases mapped loosely; unknown -> fail.
        return ToolResult.fail(f"Unknown climate mode '{mode}'")
    return vehicle.set_climate_mode(normalized)
