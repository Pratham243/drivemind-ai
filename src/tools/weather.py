"""Weather query tool (deterministic simulated backend)."""

from __future__ import annotations

import hashlib

from src.tools.base import ToolResult

_CONDITIONS = ["clear", "cloudy", "rainy", "windy", "snowy"]


def get_weather(location: str | None = None) -> ToolResult:
    location = location or "your current location"
    digest = hashlib.sha256(location.encode()).hexdigest()
    condition = _CONDITIONS[int(digest[:2], 16) % len(_CONDITIONS)]
    temperature_c = -5 + (int(digest[2:4], 16) % 35)
    return ToolResult.ok(location=location, condition=condition, temperature_c=temperature_c)
