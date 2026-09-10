from src.safety.validator import SafetyValidator


def make_validator():
    return SafetyValidator()


def test_temperature_within_range_passes():
    result = make_validator().validate("set_temperature", {"temperature": 22})
    assert result.passed


def test_temperature_above_max_rejected():
    result = make_validator().validate("set_temperature", {"temperature": 100})
    assert not result.passed
    assert "outside safe range" in result.reason


def test_temperature_below_min_rejected():
    result = make_validator().validate("set_temperature", {"temperature": 5})
    assert not result.passed


def test_temperature_missing_parameter_rejected():
    result = make_validator().validate("set_temperature", {})
    assert not result.passed
    assert "Missing required parameter" in result.reason


def test_temperature_wrong_type_rejected():
    result = make_validator().validate("set_temperature", {"temperature": "warm"})
    assert not result.passed


def test_unknown_tool_rejected():
    result = make_validator().validate("launch_missiles", {})
    assert not result.passed
    assert "Unknown tool" in result.reason


def test_volume_out_of_range_rejected():
    result = make_validator().validate("change_volume", {"volume": 500})
    assert not result.passed


def test_volume_in_range_passes():
    result = make_validator().validate("change_volume", {"volume": 50})
    assert result.passed


def test_unknown_window_rejected():
    result = make_validator().validate("open_window", {"window": "sunroof"})
    assert not result.passed


def test_known_window_passes():
    result = make_validator().validate("open_window", {"window": "front left window"})
    assert result.passed


def test_window_blocked_at_high_speed():
    result = make_validator().validate(
        "open_window", {"window": "front left window"}, current_speed_kmh=250
    )
    assert not result.passed
    assert "km/h" in result.reason


def test_start_navigation_missing_destination_rejected():
    result = make_validator().validate("start_navigation", {})
    assert not result.passed


def test_find_charging_stations_invalid_power_rejected():
    result = make_validator().validate("find_charging_stations", {"minimum_power_kw": -50})
    assert not result.passed


def test_find_charging_stations_no_args_passes():
    # location defaults inside the tool; safety layer should not require it.
    result = make_validator().validate("find_charging_stations", {})
    assert result.passed
