"""Media playback tools (play/pause/skip/volume)."""

from __future__ import annotations

from src.tools.base import ToolResult
from src.tools.vehicle import VehicleSimulator

VOLUME_STEP = 15


def play_music(
    vehicle: VehicleSimulator, music_genre: str | None = None, artist: str | None = None
) -> ToolResult:
    return vehicle.play_music(genre=music_genre, artist=artist)


def pause_music(vehicle: VehicleSimulator) -> ToolResult:
    return vehicle.pause_music()


def next_song(vehicle: VehicleSimulator) -> ToolResult:
    return vehicle.next_song()


def change_volume(
    vehicle: VehicleSimulator, volume: int | None = None, direction: str | None = None
) -> ToolResult:
    from src.safety.policies import VOLUME_MAX, VOLUME_MIN

    if volume is not None:
        if not isinstance(volume, (int, float)):
            return ToolResult.fail("volume must be numeric")
        return vehicle.set_volume(int(volume))

    current = vehicle.state.media.volume
    if direction == "up":
        new_value = min(current + VOLUME_STEP, VOLUME_MAX)
    elif direction == "down":
        new_value = max(current - VOLUME_STEP, VOLUME_MIN)
    else:
        return ToolResult.fail("Either 'volume' or 'direction' must be provided")
    return vehicle.set_volume(new_value)
