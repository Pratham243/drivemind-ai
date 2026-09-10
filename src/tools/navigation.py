"""Navigation and traffic tools."""

from __future__ import annotations

import hashlib

from src.tools.base import ToolResult
from src.tools.vehicle import VehicleSimulator


def _deterministic_value(seed_text: str, lo: int, hi: int) -> int:
    """Map a string to a stable pseudo-random integer in [lo, hi].

    Deterministic (no external API, no real-time randomness) so tests and
    demos are reproducible, while still varying sensibly by input.
    """
    digest = hashlib.sha256(seed_text.encode()).hexdigest()
    return lo + (int(digest[:8], 16) % (hi - lo + 1))


def start_navigation(vehicle: VehicleSimulator, destination: str) -> ToolResult:
    if not destination or not isinstance(destination, str):
        return ToolResult.fail("destination must be a non-empty string")
    eta_minutes = _deterministic_value(destination, 5, 90)
    distance_km = _deterministic_value(destination + "_dist", 1, 400)
    result = vehicle.start_navigation(destination)
    result.data.update(eta_minutes=eta_minutes, distance_km=distance_km)
    return result


def cancel_navigation(vehicle: VehicleSimulator) -> ToolResult:
    return vehicle.cancel_navigation()


def search_navigation(vehicle: VehicleSimulator, destination: str | None = None) -> ToolResult:
    if not vehicle.state.navigation_active:
        return ToolResult.ok(navigation_active=False, message="No active navigation.")
    dest = vehicle.state.navigation_destination
    eta_minutes = _deterministic_value(dest or "current", 5, 90)
    return ToolResult.ok(navigation_active=True, destination=dest, eta_minutes=eta_minutes)


def get_traffic(location: str | None = None) -> ToolResult:
    location = location or "current route"
    level_options = ["light", "moderate", "heavy"]
    level = level_options[_deterministic_value(location + "_traffic", 0, 2)]
    delay_minutes = 0 if level == "light" else _deterministic_value(location + "_delay", 3, 25)
    return ToolResult.ok(location=location, traffic_level=level, delay_minutes=delay_minutes)


def find_parking(location: str | None = None) -> ToolResult:
    display_location = location or "your current location"
    n_spots = _deterministic_value(display_location + "_parking", 0, 40)
    garage_prefix = location or "Central"
    return ToolResult.ok(
        location=display_location,
        available_spots=n_spots,
        nearest_garage=f"{garage_prefix} Central Parking",
    )
