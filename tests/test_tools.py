import pytest

from src.tools.charging import find_charging_stations
from src.tools.climate import (
    decrease_temperature,
    increase_temperature,
    set_climate_mode,
    set_temperature,
)
from src.tools.comfort import adjust_seat, close_window, open_window, set_ambient_lighting
from src.tools.communication import make_phone_call, send_message
from src.tools.media import change_volume, next_song, pause_music, play_music
from src.tools.navigation import (
    cancel_navigation,
    find_parking,
    get_traffic,
    search_navigation,
    start_navigation,
)
from src.tools.restaurant import search_restaurants
from src.tools.vehicle import VehicleSimulator
from src.tools.weather import get_weather


@pytest.fixture
def vehicle():
    return VehicleSimulator()


def test_set_temperature_updates_state(vehicle):
    result = set_temperature(vehicle, 24)
    assert result.success
    assert vehicle.state.temperature_c == 24


def test_set_temperature_rejects_non_numeric(vehicle):
    result = set_temperature(vehicle, "warm")
    assert not result.success


def test_increase_temperature_clamped_to_safe_max(vehicle):
    vehicle.state.temperature_c = 29
    for _ in range(5):
        increase_temperature(vehicle)
    assert vehicle.state.temperature_c <= 30


def test_decrease_temperature_clamped_to_safe_min(vehicle):
    vehicle.state.temperature_c = 17
    for _ in range(5):
        decrease_temperature(vehicle)
    assert vehicle.state.temperature_c >= 16


def test_open_close_window(vehicle):
    result = open_window(vehicle, "front left window")
    assert result.success
    assert vehicle.state.windows["front_left"] == "open"
    result = close_window(vehicle, "front left window")
    assert vehicle.state.windows["front_left"] == "closed"


def test_open_window_rejects_unknown_window(vehicle):
    result = open_window(vehicle, "sunroof")
    assert not result.success


def test_find_charging_stations_filters_by_power():
    result = find_charging_stations(location="Berlin", minimum_power_kw=200)
    assert result.success
    for station in result.data["stations"]:
        assert station["power_kw"] >= 200


def test_find_charging_stations_rejects_invalid_power():
    result = find_charging_stations(location="Berlin", minimum_power_kw=-10)
    assert not result.success


def test_play_pause_next_music(vehicle):
    result = play_music(vehicle, music_genre="jazz")
    assert result.success and vehicle.state.media.playing
    pause_music(vehicle)
    assert not vehicle.state.media.playing
    next_song(vehicle)
    assert vehicle.state.media.playing


def test_change_volume_by_direction(vehicle):
    vehicle.state.media.volume = 50
    change_volume(vehicle, direction="up")
    assert vehicle.state.media.volume > 50


def test_change_volume_requires_argument(vehicle):
    result = change_volume(vehicle)
    assert not result.success


def test_start_and_cancel_navigation(vehicle):
    result = start_navigation(vehicle, "Berlin Brandenburg Airport")
    assert result.success
    assert vehicle.state.navigation_active
    cancel_navigation(vehicle)
    assert not vehicle.state.navigation_active


def test_start_navigation_requires_destination(vehicle):
    result = start_navigation(vehicle, "")
    assert not result.success


def test_get_traffic_returns_structured_output():
    result = get_traffic("Berlin")
    assert result.success
    assert "traffic_level" in result.data


def test_find_parking_defaults_location():
    result = find_parking()
    assert result.success
    assert "available_spots" in result.data


def test_search_restaurants_by_cuisine():
    result = search_restaurants(location="Munich", restaurant_cuisine="Italian")
    assert result.success
    assert result.data["cuisine"] == "Italian"
    assert result.data["count"] > 0


def test_get_weather_deterministic():
    r1 = get_weather("Berlin")
    r2 = get_weather("Berlin")
    assert r1.data == r2.data  # deterministic simulated backend


def test_make_phone_call_and_send_message():
    result = make_phone_call("Mom")
    assert result.success and result.data["contact"] == "Mom"
    result = send_message("Mom", "On my way")
    assert result.success and result.data["message_body"] == "On my way"


def test_make_phone_call_requires_contact():
    result = make_phone_call("")
    assert not result.success


def test_adjust_seat_and_ambient_lighting(vehicle):
    result = adjust_seat(vehicle, "driver seat", "heated")
    assert result.success
    assert vehicle.state.seats["driver seat"] == "heated"
    result = set_ambient_lighting(vehicle, "blue")
    assert result.success
    assert vehicle.state.ambient_light_color == "blue"


def test_get_vehicle_status_reflects_current_state(vehicle):
    vehicle.state.temperature_c = 25
    result = vehicle.get_status()
    assert result.success
    assert result.data["temperature_c"] == 25
    assert "battery_pct" in result.data
    assert "range_km" in result.data


def test_get_battery_level_returns_current_value(vehicle):
    vehicle.state.battery_pct = 42
    result = vehicle.get_battery_level()
    assert result.success
    assert result.data["battery_pct"] == 42


def test_get_range_returns_current_value(vehicle):
    vehicle.state.range_km = 150
    result = vehicle.get_range()
    assert result.success
    assert result.data["range_km"] == 150


def test_get_tire_pressure_all(vehicle):
    result = vehicle.get_tire_pressure("all")
    assert result.success
    assert set(result.data["tire_pressure_bar"].keys()) == {
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    }


def test_get_tire_pressure_single_tire(vehicle):
    result = vehicle.get_tire_pressure("front left")
    assert result.success
    assert result.data["tire"] == "front left"
    assert isinstance(result.data["pressure_bar"], float)


def test_get_tire_pressure_rejects_unknown_tire(vehicle):
    result = vehicle.get_tire_pressure("sunroof")
    assert not result.success


def test_search_navigation_reports_no_active_route(vehicle):
    result = search_navigation(vehicle)
    assert result.success
    assert result.data["navigation_active"] is False


def test_search_navigation_reports_active_route(vehicle):
    start_navigation(vehicle, "Berlin Brandenburg Airport")
    result = search_navigation(vehicle)
    assert result.success
    assert result.data["navigation_active"] is True
    assert result.data["destination"] == "Berlin Brandenburg Airport"


def test_set_climate_mode_accepts_known_mode_and_updates_state(vehicle):
    result = set_climate_mode(vehicle, mode="ac_on")
    assert result.success
    assert result.data["climate_mode"] == "ac_on"
    assert vehicle.state.climate_mode == "ac_on"


def test_set_climate_mode_rejects_unknown_mode(vehicle):
    result = set_climate_mode(vehicle, mode="warp_drive")
    assert not result.success
