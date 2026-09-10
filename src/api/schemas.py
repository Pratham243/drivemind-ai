"""Pydantic request/response schemas for the DriveMind API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, examples=["Find a charging station near Berlin"])


class ChatResponse(BaseModel):
    message: str
    intent: str | None
    entities: dict[str, Any]
    tool: str | None
    tool_arguments: dict[str, Any]
    safety: dict[str, Any]
    tool_result: dict[str, Any]
    response: str
    steps: list[dict[str, Any]]
    errors: list[str]


class IntentRequest(BaseModel):
    text: str = Field(min_length=1)


class IntentResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    entities: dict[str, Any]
    model_used: str


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)


class ClimateRequest(BaseModel):
    temperature: int | None = None
    amount: str | None = None
    direction: str | None = Field(
        default=None, description="'increase' or 'decrease' when temperature is not given"
    )


class VehicleStatusResponse(BaseModel):
    battery_pct: int
    temperature_c: int
    speed_kmh: int
    range_km: int
    windows: dict[str, str]
    tire_pressure_bar: dict[str, float]
    seats: dict[str, str]
    ambient_light_color: str
    climate_mode: str
    navigation_active: bool
    navigation_destination: str | None
    media: dict[str, Any]


class ChargingStationsQuery(BaseModel):
    location: str | None = None
    minimum_power_kw: int | None = None


class HealthResponse(BaseModel):
    status: str
    nlu_model: str
    llm_provider: str
    version: str = "0.1.0"
