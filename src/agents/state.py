"""Agent execution state.

A single :class:`AgentState` is produced per user request and fully
describes what the agent understood and did — this is also the object
serialized as the "agent trace" surfaced in the API response and the
Streamlit debug panel (PHASE 33).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTrace:
    """One resolved (intent -> tool -> safety -> result) step. A simple
    request produces one step; a multi-step request produces several.
    """

    sub_message: str
    intent: str
    confidence: float
    entities: dict[str, Any]
    selected_tool: str | None
    tool_arguments: dict[str, Any]
    safety: dict[str, Any]
    tool_result: dict[str, Any]
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class AgentState:
    user_message: str
    steps: list[StepTrace] = field(default_factory=list)
    final_response: str = ""
    errors: list[str] = field(default_factory=list)

    # Convenience accessors mirroring the single-step fields requested in
    # the spec, populated from the first step for simple (non multi-step)
    # requests so API consumers don't have to special-case step count.
    @property
    def intent(self) -> str | None:
        return self.steps[0].intent if self.steps else None

    @property
    def entities(self) -> dict[str, Any]:
        return self.steps[0].entities if self.steps else {}

    @property
    def selected_tool(self) -> str | None:
        return self.steps[0].selected_tool if self.steps else None

    @property
    def tool_arguments(self) -> dict[str, Any]:
        return self.steps[0].tool_arguments if self.steps else {}

    @property
    def safety_status(self) -> dict[str, Any]:
        return self.steps[0].safety if self.steps else {}

    @property
    def tool_result(self) -> dict[str, Any]:
        return self.steps[0].tool_result if self.steps else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "steps": [s.to_dict() for s in self.steps],
            "final_response": self.final_response,
            "errors": self.errors,
            "intent": self.intent,
            "entities": self.entities,
            "selected_tool": self.selected_tool,
            "tool_arguments": self.tool_arguments,
            "safety": self.safety_status,
            "tool_result": self.tool_result,
        }
