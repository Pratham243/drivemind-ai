"""Shared tool infrastructure: result envelope + validation error type.

Every tool function returns a :class:`ToolResult` so the agent layer has
one uniform shape to reason about regardless of which tool ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ToolValidationError(Exception):
    """Raised when tool arguments fail validation (missing/invalid/out of range)."""


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "data": self.data, "error": self.error}

    @classmethod
    def ok(cls, **data: Any) -> "ToolResult":
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        return cls(success=False, data={}, error=error)
