"""EV charging station search tool.

Backed by a small deterministic simulated directory (no external API
dependency) so the tool is fully testable offline.
"""

from __future__ import annotations

import hashlib

from src.tools.base import ToolResult

_STATION_NAME_POOLS = [
    "IONITY",
    "EnBW HyperNetz",
    "Tesla Supercharger",
    "Aral pulse",
    "EWE Go",
]


def _stations_for(location: str) -> list[dict]:
    stations = []
    for i, provider in enumerate(_STATION_NAME_POOLS):
        digest = hashlib.sha256(f"{location}_{provider}_{i}".encode()).hexdigest()
        power_kw = [50, 100, 150, 250, 350][int(digest[:2], 16) % 5]
        distance_km = round((int(digest[2:4], 16) % 200) / 10, 1)
        available = int(digest[4:6], 16) % 2 == 0
        stations.append(
            {
                "name": f"{provider} {location}",
                "location": location,
                "power_kw": power_kw,
                "distance_km": distance_km,
                "available": available,
            }
        )
    return sorted(stations, key=lambda s: s["distance_km"])


def find_charging_stations(
    location: str | None = None, minimum_power_kw: int | None = None
) -> ToolResult:
    location = location or "your current location"
    if minimum_power_kw is not None and (
        not isinstance(minimum_power_kw, (int, float)) or minimum_power_kw < 0
    ):
        return ToolResult.fail(f"Invalid minimum_power_kw: {minimum_power_kw}")

    stations = _stations_for(location)
    if minimum_power_kw:
        stations = [s for s in stations if s["power_kw"] >= minimum_power_kw]

    return ToolResult.ok(
        location=location,
        minimum_power_kw=minimum_power_kw,
        count=len(stations),
        stations=stations,
    )
