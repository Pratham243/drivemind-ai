"""Agent-level evaluation.

Distinct from ML model evaluation (metrics.py): this measures the *whole*
pipeline (NLU -> planner -> safety -> tool execution) against a labeled
set of expected outcomes — tool selection accuracy, task success rate,
parameter (entity) extraction accuracy, safety rejection accuracy, and
latency. All numbers are computed from actually running the agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.agent import AgentController
from src.tools.vehicle import VehicleSimulator


@dataclass
class AgentEvalCase:
    message: str
    expected_intent: str
    expected_tool: str | None = None
    expected_entities: dict[str, Any] = field(default_factory=dict)
    expect_safety_rejection: bool = False


@dataclass
class AgentEvalReport:
    n_cases: int
    intent_accuracy: float
    tool_selection_accuracy: float
    entity_extraction_accuracy: float
    task_success_rate: float
    safety_rejection_accuracy: float
    fallback_rate: float
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    failures: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def _entities_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Entity extraction is correct if every expected key/value is present
    (extra, non-conflicting keys the extractor finds are not penalized).
    """
    return all(actual.get(k) == v for k, v in expected.items())


def evaluate_agent(cases: list[AgentEvalCase]) -> AgentEvalReport:
    agent = AgentController(vehicle=VehicleSimulator())  # isolated vehicle per eval run

    n = len(cases)
    intent_correct = 0
    tool_correct = 0
    entity_correct = 0
    task_success = 0
    safety_correct = 0
    n_safety_cases = 0
    n_fallback = 0
    n_errors = 0
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    for case in cases:
        start = time.perf_counter()
        state = agent.handle(case.message)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        step = state.steps[0] if state.steps else None
        if state.errors:
            n_errors += 1

        case_failures = []

        intent_ok = step is not None and step.intent == case.expected_intent
        intent_correct += int(intent_ok)
        if not intent_ok:
            case_failures.append(
                f"intent: expected {case.expected_intent}, got {step.intent if step else None}"
            )

        if step and step.model_used and "llm_fallback" in step.model_used:
            n_fallback += 1

        if case.expected_tool is not None:
            tool_ok = step is not None and step.selected_tool == case.expected_tool
            tool_correct += int(tool_ok)
            if not tool_ok:
                case_failures.append(
                    f"tool: expected {case.expected_tool}, got {step.selected_tool if step else None}"
                )
        else:
            tool_correct += 1  # not evaluated for this case

        if case.expected_entities:
            entity_ok = step is not None and _entities_match(case.expected_entities, step.entities)
            entity_correct += int(entity_ok)
            if not entity_ok:
                case_failures.append(
                    f"entities: expected {case.expected_entities}, got {step.entities if step else None}"
                )
        else:
            entity_correct += 1

        if case.expect_safety_rejection:
            n_safety_cases += 1
            safety_ok = step is not None and step.safety.get("status") == "rejected"
            safety_correct += int(safety_ok)
            if not safety_ok:
                case_failures.append("safety: expected rejection, but call passed")
            task_success += int(safety_ok)  # "success" here = correctly rejected
        else:
            success_ok = (
                step is not None
                and step.safety.get("status") in ("passed", "not_applicable")
                and (step.selected_tool is None or step.tool_result.get("success", False))
            )
            task_success += int(success_ok)
            if not success_ok:
                case_failures.append("task did not complete successfully")

        if case_failures:
            failures.append({"message": case.message, "problems": case_failures})

    latencies.sort()
    p95_idx = min(len(latencies) - 1, int(0.95 * len(latencies)))

    return AgentEvalReport(
        n_cases=n,
        intent_accuracy=round(intent_correct / n, 4),
        tool_selection_accuracy=round(tool_correct / n, 4),
        entity_extraction_accuracy=round(entity_correct / n, 4),
        task_success_rate=round(task_success / n, 4),
        safety_rejection_accuracy=(
            round(safety_correct / n_safety_cases, 4) if n_safety_cases else 1.0
        ),
        fallback_rate=round(n_fallback / n, 4),
        error_rate=round(n_errors / n, 4),
        avg_latency_ms=round(sum(latencies) / n, 2),
        p95_latency_ms=round(latencies[p95_idx], 2),
        failures=failures,
    )
