"""API route handlers.

Kept separate from ``main.py`` (app creation/wiring) so the routing logic
is independently testable and readable.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Query

from src.agents.agent import AgentController
from src.api.schemas import (
    AgentRequest,
    ChatRequest,
    ChatResponse,
    ClimateRequest,
    HealthResponse,
    IntentRequest,
    IntentResponse,
    VehicleStatusResponse,
)
from src.config.settings import get_settings
from src.nlu.llm_provider import get_llm_provider
from src.nlu.router import NLURouter
from src.safety.validator import SafetyValidator
from src.tools.charging import find_charging_stations
from src.tools.climate import decrease_temperature, increase_temperature, set_temperature
from src.tools.vehicle import get_vehicle

logger = logging.getLogger("drivemind.api")
router = APIRouter()

_agent = AgentController()
_nlu_router = NLURouter()
_safety = SafetyValidator()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    llm = get_llm_provider()
    return HealthResponse(status="ok", nlu_model=settings.nlu_model, llm_provider=llm.name)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    state = _agent.handle(request.message)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "chat request=%r intent=%s tool=%s latency_ms=%s",
        request.message,
        state.intent,
        state.selected_tool,
        latency_ms,
    )
    return ChatResponse(
        message=request.message,
        intent=state.intent,
        entities=state.entities,
        tool=state.selected_tool,
        tool_arguments=state.tool_arguments,
        safety=state.safety_status,
        tool_result=state.tool_result,
        response=state.final_response,
        steps=[s.to_dict() for s in state.steps],
        errors=state.errors,
    )


@router.post("/intent", response_model=IntentResponse)
def classify_intent(request: IntentRequest) -> IntentResponse:
    result = _nlu_router.process(request.text)
    return IntentResponse(
        text=result.text,
        intent=result.intent,
        confidence=result.confidence,
        entities=result.entities,
        model_used=result.model_used,
    )


@router.post("/agent", response_model=ChatResponse)
def run_agent(request: AgentRequest) -> ChatResponse:
    state = _agent.handle(request.message)
    return ChatResponse(
        message=request.message,
        intent=state.intent,
        entities=state.entities,
        tool=state.selected_tool,
        tool_arguments=state.tool_arguments,
        safety=state.safety_status,
        tool_result=state.tool_result,
        response=state.final_response,
        steps=[s.to_dict() for s in state.steps],
        errors=state.errors,
    )


@router.get("/vehicle/status", response_model=VehicleStatusResponse)
def vehicle_status() -> VehicleStatusResponse:
    vehicle = get_vehicle()
    return VehicleStatusResponse(**vehicle.state.to_dict())


@router.get("/vehicle/battery")
def vehicle_battery() -> dict:
    vehicle = get_vehicle()
    return vehicle.get_battery_level().to_dict()


@router.post("/vehicle/climate")
def vehicle_climate(request: ClimateRequest) -> dict:
    vehicle = get_vehicle()

    if request.temperature is not None:
        # Every state-mutating request — REST or agent-driven — passes
        # through the same safety validator before touching vehicle state.
        safety_result = _safety.validate("set_temperature", {"temperature": request.temperature})
        if not safety_result.passed:
            raise HTTPException(status_code=400, detail=safety_result.reason)
        result = set_temperature(vehicle, request.temperature)
    elif request.direction == "increase":
        result = increase_temperature(vehicle, request.amount)
    elif request.direction == "decrease":
        result = decrease_temperature(vehicle, request.amount)
    else:
        raise HTTPException(status_code=400, detail="Provide 'temperature' or 'direction'.")

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return result.to_dict()


@router.get("/charging-stations")
def charging_stations(
    location: str | None = Query(default=None),
    minimum_power_kw: int | None = Query(default=None),
) -> dict:
    result = find_charging_stations(location=location, minimum_power_kw=minimum_power_kw)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    return result.to_dict()
