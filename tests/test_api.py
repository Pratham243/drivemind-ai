import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.model_manager import get_model_manager

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def _require_trained_model():
    if not get_model_manager().is_ready:
        pytest.skip("Baseline model not trained. Run scripts/train_baseline.py first.")


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "nlu_model" in body


def test_chat_endpoint_returns_structured_response():
    response = client.post("/chat", json={"message": "What's my battery level?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "battery_status"
    assert body["tool"] == "get_battery_level"
    assert body["safety"]["status"] == "passed"


def test_chat_endpoint_rejects_empty_message():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_intent_endpoint():
    response = client.post("/intent", json={"text": "Play some jazz"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "play_music"
    assert body["entities"].get("music_genre") == "jazz"


def test_vehicle_status_endpoint():
    response = client.get("/vehicle/status")
    assert response.status_code == 200
    body = response.json()
    assert "battery_pct" in body
    assert "temperature_c" in body


def test_vehicle_status_schema_matches_full_vehicle_state():
    """Regression test: VehicleStatusResponse previously didn't declare
    `climate_mode`, so pydantic's default extra='ignore' behavior silently
    dropped it from every /vehicle/status response with no error anywhere
    — the field just vanished. Assert the two stay in sync so a future
    VehicleState field addition fails loudly instead of silently.
    """
    from src.tools.vehicle import VehicleSimulator

    state_keys = set(VehicleSimulator().state.to_dict().keys())
    response = client.get("/vehicle/status")
    response_keys = set(response.json().keys())
    assert state_keys == response_keys


def test_vehicle_climate_endpoint_valid_temperature():
    response = client.post("/vehicle/climate", json={"temperature": 21})
    assert response.status_code == 200
    assert response.json()["data"]["temperature_c"] == 21


def test_vehicle_climate_endpoint_rejects_unsafe_temperature():
    response = client.post("/vehicle/climate", json={"temperature": 100})
    assert response.status_code == 400
    assert "outside safe range" in response.json()["detail"]


def test_vehicle_climate_endpoint_requires_input():
    response = client.post("/vehicle/climate", json={})
    assert response.status_code == 400


def test_charging_stations_endpoint():
    response = client.get(
        "/charging-stations", params={"location": "Berlin", "minimum_power_kw": 150}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["count"] >= 0
    for station in body["data"]["stations"]:
        assert station["power_kw"] >= 150


def test_agent_endpoint_multi_step():
    response = client.post(
        "/agent", json={"message": "What's my battery level and find a charger."}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["steps"]) == 2
