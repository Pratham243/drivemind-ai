"""Simulated in-car vehicle backend.

Holds the single source of truth for vehicle state. Tools in this package
mutate state via :class:`VehicleSimulator` methods only — the agent/LLM
never touches state directly (see src/safety for the enforcement layer
that sits in front of every mutation).

A process-wide singleton (:func:`get_vehicle`) is used so the FastAPI app,
Streamlit dashboard, and agent all observe the same simulated car during a
run, matching how a real vehicle has exactly one state.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

from src.config.settings import load_yaml_config
from src.tools.base import ToolResult


@dataclass
class MediaState:
    playing: bool = False
    track: str | None = None
    genre: str | None = None
    volume: int = 50


@dataclass
class VehicleState:
    battery_pct: int = 68
    temperature_c: int = 20
    speed_kmh: int = 0
    range_km: int = 320
    windows: dict[str, str] = field(
        default_factory=lambda: {
            "front_left": "closed",
            "front_right": "closed",
            "rear_left": "closed",
            "rear_right": "closed",
        }
    )
    tire_pressure_bar: dict[str, float] = field(
        default_factory=lambda: {
            "front_left": 2.3,
            "front_right": 2.3,
            "rear_left": 2.3,
            "rear_right": 2.3,
        }
    )
    seats: dict[str, str] = field(
        default_factory=lambda: {
            "driver seat": "normal",
            "passenger seat": "normal",
        }
    )
    ambient_light_color: str = "white"
    climate_mode: str = "off"  # e.g. "ac_on", "fan_off", "heater_on"
    navigation_active: bool = False
    navigation_destination: str | None = None
    media: MediaState = field(default_factory=MediaState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_pct": self.battery_pct,
            "temperature_c": self.temperature_c,
            "speed_kmh": self.speed_kmh,
            "range_km": self.range_km,
            "windows": dict(self.windows),
            "tire_pressure_bar": dict(self.tire_pressure_bar),
            "seats": dict(self.seats),
            "ambient_light_color": self.ambient_light_color,
            "climate_mode": self.climate_mode,
            "navigation_active": self.navigation_active,
            "navigation_destination": self.navigation_destination,
            "media": {
                "playing": self.media.playing,
                "track": self.media.track,
                "genre": self.media.genre,
                "volume": self.media.volume,
            },
        }


class VehicleSimulator:
    """In-memory simulated vehicle. Not thread-safe by design — this is a
    single-driver demo simulator, not a production fleet backend.
    """

    def __init__(self) -> None:
        cfg = load_yaml_config("config.yaml")["vehicle"]
        self.state = VehicleState(
            battery_pct=cfg["initial_battery_pct"],
            temperature_c=cfg["initial_temperature_c"],
            range_km=cfg["initial_range_km"],
            speed_kmh=cfg["initial_speed_kmh"],
        )

    def reset(self) -> None:
        self.__init__()  # noqa: PLW0108 - deliberate full reinitialization

    # --- read-only queries -------------------------------------------------
    def get_status(self) -> ToolResult:
        return ToolResult.ok(**self.state.to_dict())

    def get_battery_level(self) -> ToolResult:
        return ToolResult.ok(battery_pct=self.state.battery_pct)

    def get_range(self) -> ToolResult:
        return ToolResult.ok(range_km=self.state.range_km)

    def get_tire_pressure(self, tire: str = "all") -> ToolResult:
        if tire == "all":
            return ToolResult.ok(tire_pressure_bar=dict(self.state.tire_pressure_bar))
        key = tire.replace(" ", "_").lower()
        if key not in self.state.tire_pressure_bar:
            return ToolResult.fail(f"Unknown tire position: '{tire}'")
        return ToolResult.ok(tire=tire, pressure_bar=self.state.tire_pressure_bar[key])

    # --- mutations (called only after safety validation) --------------------
    def set_temperature(self, value: int) -> ToolResult:
        self.state.temperature_c = value
        return ToolResult.ok(temperature_c=self.state.temperature_c)

    def set_window(self, window: str, position: str) -> ToolResult:
        if window == "all windows":
            for key in self.state.windows:
                self.state.windows[key] = position
            return ToolResult.ok(windows=dict(self.state.windows))
        key = window.replace(" window", "").replace(" ", "_").lower()
        key_map = {"driver": "front_left", "passenger": "front_right"}
        key = key_map.get(key, key)
        if key not in self.state.windows:
            return ToolResult.fail(f"Unknown window: '{window}'")
        self.state.windows[key] = position
        return ToolResult.ok(window=window, position=position)

    def set_seat(self, seat: str, adjustment: str) -> ToolResult:
        key = seat.lower()
        key_map = {"my seat": "driver seat", "the back seat": "passenger seat"}
        key = key_map.get(key, key)
        if key not in self.state.seats:
            self.state.seats[key] = adjustment
        else:
            self.state.seats[key] = adjustment
        return ToolResult.ok(seat=seat, adjustment=adjustment)

    def set_ambient_light(self, color: str) -> ToolResult:
        self.state.ambient_light_color = color
        return ToolResult.ok(ambient_light_color=color)

    def set_climate_mode(self, mode: str) -> ToolResult:
        self.state.climate_mode = mode
        return ToolResult.ok(climate_mode=mode)

    def start_navigation(self, destination: str) -> ToolResult:
        self.state.navigation_active = True
        self.state.navigation_destination = destination
        return ToolResult.ok(navigation_active=True, destination=destination)

    def cancel_navigation(self) -> ToolResult:
        self.state.navigation_active = False
        self.state.navigation_destination = None
        return ToolResult.ok(navigation_active=False)

    def play_music(self, genre: str | None = None, artist: str | None = None) -> ToolResult:
        self.state.media.playing = True
        self.state.media.genre = genre
        self.state.media.track = artist or (f"{genre.title()} Mix" if genre else "Shuffle Mix")
        return ToolResult.ok(
            playing=True, track=self.state.media.track, genre=self.state.media.genre
        )

    def pause_music(self) -> ToolResult:
        self.state.media.playing = False
        return ToolResult.ok(playing=False)

    def next_song(self) -> ToolResult:
        if not self.state.media.playing:
            self.state.media.playing = True
        self.state.media.track = "Next Track"
        return ToolResult.ok(track=self.state.media.track)

    def set_volume(self, volume: int) -> ToolResult:
        self.state.media.volume = volume
        return ToolResult.ok(volume=volume)


@functools.lru_cache
def get_vehicle() -> VehicleSimulator:
    """Process-wide singleton vehicle instance."""
    return VehicleSimulator()
