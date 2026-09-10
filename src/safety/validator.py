"""Safety validation layer.

This module sits between the agent's tool selection and actual tool
execution:

    Agent -> structured tool call -> SafetyValidator.validate() -> Tool

No tool that mutates vehicle state may run unless it passes here. Every
rejection is logged (never silently dropped) and returned to the agent as
a user-safe error message rather than raising into the request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.safety.policies import (
    MAX_SPEED_FOR_WINDOW_OPEN_KMH,
    REQUIRED_TOOL_ARGS,
    TEMPERATURE_MAX_C,
    TEMPERATURE_MIN_C,
    VALID_TIRES,
    VALID_WINDOWS,
    VOLUME_MAX,
    VOLUME_MIN,
)

logger = logging.getLogger("drivemind.safety")


@dataclass
class SafetyResult:
    status: str  # "passed" | "rejected"
    reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason}


class SafetyValidator:
    """Validates a (tool_name, arguments) pair against safety policies."""

    def validate(
        self, tool_name: str, arguments: dict[str, Any], current_speed_kmh: int = 0
    ) -> SafetyResult:
        if tool_name not in REQUIRED_TOOL_ARGS:
            return self._reject(tool_name, arguments, f"Unknown tool '{tool_name}'")

        required = REQUIRED_TOOL_ARGS[tool_name]
        missing = [arg for arg in required if arg not in arguments or arguments[arg] is None]
        if missing:
            return self._reject(
                tool_name, arguments, f"Missing required parameter(s): {', '.join(missing)}"
            )

        # --- per-tool policy checks -----------------------------------
        if tool_name == "set_temperature":
            value = arguments.get("temperature")
            if not isinstance(value, (int, float)):
                return self._reject(tool_name, arguments, "temperature must be numeric")
            if not (TEMPERATURE_MIN_C <= value <= TEMPERATURE_MAX_C):
                return self._reject(
                    tool_name,
                    arguments,
                    f"temperature {value} outside safe range "
                    f"[{TEMPERATURE_MIN_C}, {TEMPERATURE_MAX_C}]°C",
                )

        if (
            tool_name == "change_volume"
            and "volume" in arguments
            and arguments["volume"] is not None
        ):
            value = arguments["volume"]
            if not isinstance(value, (int, float)):
                return self._reject(tool_name, arguments, "volume must be numeric")
            if not (VOLUME_MIN <= value <= VOLUME_MAX):
                return self._reject(
                    tool_name,
                    arguments,
                    f"volume {value} outside valid range [{VOLUME_MIN}, {VOLUME_MAX}]",
                )

        if tool_name in ("open_window",):
            window = arguments.get("window")
            if window is not None and window not in VALID_WINDOWS:
                return self._reject(tool_name, arguments, f"Unknown window '{window}'")
            if current_speed_kmh > MAX_SPEED_FOR_WINDOW_OPEN_KMH:
                return self._reject(
                    tool_name,
                    arguments,
                    f"Cannot open window above {MAX_SPEED_FOR_WINDOW_OPEN_KMH} km/h (current: {current_speed_kmh})",
                )

        if tool_name in ("open_window", "close_window"):
            window = arguments.get("window")
            if window is not None and window not in VALID_WINDOWS:
                return self._reject(tool_name, arguments, f"Unknown window '{window}'")

        if tool_name == "get_tire_pressure":
            tire = arguments.get("tire", "all")
            if tire not in VALID_TIRES:
                return self._reject(tool_name, arguments, f"Unknown tire position '{tire}'")

        if tool_name == "find_charging_stations":
            kw = arguments.get("minimum_power_kw")
            if kw is not None and (not isinstance(kw, (int, float)) or kw < 0 or kw > 400):
                return self._reject(tool_name, arguments, f"Invalid minimum_power_kw '{kw}'")

        return SafetyResult(status="passed")

    def _reject(self, tool_name: str, arguments: dict[str, Any], reason: str) -> SafetyResult:
        logger.warning("Safety rejection: tool=%s args=%s reason=%s", tool_name, arguments, reason)
        return SafetyResult(status="rejected", reason=reason)
