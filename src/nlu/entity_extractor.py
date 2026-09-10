"""Deterministic, rule-based entity extraction.

Design decision (documented for interview discussion): entity extraction
here uses regex + gazetteer lookups rather than a trained NER model.
For a closed automotive domain with a bounded vocabulary (city names,
music genres, window positions, ...) deterministic extraction is more
reliable and dramatically cheaper than a statistical NER model, and its
behavior is fully predictable and unit-testable. A statistical/transformer
NER model would be a natural extension for open-vocabulary entities (e.g.
arbitrary destinations); see README "Future Improvements".
"""

from __future__ import annotations

import re
from typing import Any

_LOCATIONS = [
    "Berlin",
    "Munich",
    "Hamburg",
    "Frankfurt",
    "Cologne",
    "Stuttgart",
    "Dusseldorf",
    "Duesseldorf",
    "Leipzig",
    "Dresden",
    "Nuremberg",
    "Bremen",
    "Hannover",
    "Vienna",
    "Zurich",
    "Amsterdam",
    "Paris",
]

_DESTINATIONS_EXTRA = [
    "Berlin Brandenburg Airport",
    "Munich Airport",
    "Frankfurt Airport",
    "the nearest gas station",
    "my office",
    "the train station",
    "home",
]

_GENRES = [
    "jazz",
    "rock",
    "pop",
    "classical",
    "hip hop",
    "techno",
    "blues",
    "country",
    "reggae",
    "edm",
    "metal",
    "r&b",
    "r & b",
]

_ARTISTS = [
    "Taylor Swift",
    "Daft Punk",
    "Mozart",
    "Beethoven",
    "Ed Sheeran",
    "Beyonce",
    "Coldplay",
    "The Beatles",
]

_WINDOWS = [
    "front left window",
    "front right window",
    "rear left window",
    "rear right window",
    "all windows",
    "driver window",
    "passenger window",
    "window",
]

_SEATS = ["driver seat", "passenger seat", "my seat", "the back seat", "seat"]

_COLORS = ["blue", "red", "green", "purple", "white", "warm white", "orange"]

_CUISINES = [
    "italian",
    "chinese",
    "indian",
    "turkish",
    "mexican",
    "vegan",
    "sushi",
    "thai",
    "greek",
    "vietnamese",
]

_TIRES = ["front left", "front right", "rear left", "rear right", "all"]

_CONTACTS = [
    "mom",
    "dad",
    "john",
    "sarah",
    "my wife",
    "my husband",
    "my boss",
    "alex",
    "julia",
    "peter",
    "anna",
]

_TEMP_RE = re.compile(r"(-?\d{1,3})\s*(?:degrees?|°)?", re.IGNORECASE)
_KW_RE = re.compile(r"(\d{2,3})\s*kw", re.IGNORECASE)
_VOLUME_RE = re.compile(r"\bvolume\D{0,10}?(\d{1,3})\b", re.IGNORECASE)
_MESSAGE_BODY_RE = re.compile(
    r"\b(?:that|saying|telling (?:them|him|her))\s+(.+?)[.!?]*$", re.IGNORECASE
)


def _find_first(text: str, candidates: list[str]) -> str | None:
    lowered = text.lower()
    # Prefer longer matches first (e.g. "front left window" before "window").
    for cand in sorted(candidates, key=len, reverse=True):
        if cand.lower() in lowered:
            return cand
    return None


def extract_entities(text: str, intent: str | None = None) -> dict[str, Any]:
    """Extract structured entities from raw text.

    If ``intent`` is provided, extraction is scoped to the entity keys
    relevant to that intent (see configs/intents.yaml), reducing false
    positives. Without an intent, all entity types are attempted.
    """
    from src.config.settings import load_intents  # local import avoids cycle

    entities: dict[str, Any] = {}
    allowed = None
    if intent is not None:
        intents_cfg = load_intents()
        allowed = set(intents_cfg.get(intent, {}).get("entities", []))

    def wants(key: str) -> bool:
        return allowed is None or key in allowed

    lowered = text.lower()

    if wants("temperature") or wants("amount"):
        match = _TEMP_RE.search(text)
        if match and ("degree" in lowered or "°" in text or wants("temperature")):
            value = int(match.group(1))
            # Deliberately wide bound here: extraction must surface
            # out-of-range values (e.g. 100 degrees) so the safety layer
            # can reject them with a meaningful reason. Narrowing this
            # range would silently hide unsafe requests as "missing"
            # rather than "rejected".
            if wants("temperature") and -50 <= value <= 200:
                entities["temperature"] = value

    if wants("minimum_power_kw"):
        match = _KW_RE.search(text)
        if match:
            entities["minimum_power_kw"] = int(match.group(1))

    if wants("volume"):
        match = _VOLUME_RE.search(text)
        if match:
            entities["volume"] = int(match.group(1))

    if wants("location"):
        loc = _find_first(text, _LOCATIONS)
        if loc:
            entities["location"] = loc

    if wants("destination"):
        dest = _find_first(text, _LOCATIONS + _DESTINATIONS_EXTRA)
        if dest:
            entities["destination"] = dest

    if wants("music_genre"):
        genre = _find_first(text, _GENRES)
        if genre:
            entities["music_genre"] = genre

    if wants("artist"):
        artist = _find_first(text, _ARTISTS)
        if artist:
            entities["artist"] = artist

    if wants("window"):
        win = _find_first(text, _WINDOWS)
        if win:
            entities["window"] = win

    if wants("seat"):
        seat = _find_first(text, _SEATS)
        if seat:
            entities["seat"] = seat

    if wants("color"):
        color = _find_first(text, _COLORS)
        if color:
            entities["color"] = color

    if wants("restaurant_cuisine"):
        cuisine = _find_first(text, _CUISINES)
        if cuisine:
            entities["restaurant_cuisine"] = cuisine

    if wants("tire"):
        tire = _find_first(text, _TIRES)
        if tire:
            entities["tire"] = tire

    if wants("phone_contact"):
        contact = _find_first(text, _CONTACTS)
        if contact:
            entities["phone_contact"] = contact.title() if contact.islower() else contact

    if wants("message_body"):
        match = _MESSAGE_BODY_RE.search(text)
        if match:
            entities["message_body"] = match.group(1).strip()

    if wants("direction"):
        if any(w in lowered for w in ["up", "louder", "increase", "raise"]):
            entities["direction"] = "up"
        elif any(w in lowered for w in ["down", "lower", "quieter", "decrease"]):
            entities["direction"] = "down"

    if wants("amount"):
        for phrase in ["a bit", "a little", "a lot", "slightly", "a few degrees"]:
            if phrase in lowered:
                entities["amount"] = phrase
                break

    if wants("mode"):
        is_on = bool(re.search(r"\bon\b", lowered))
        is_off = bool(re.search(r"\boff\b", lowered))
        if "fan" in lowered:
            system = "fan"
        elif "heat" in lowered:
            system = "heater"
        elif re.search(r"\bac\b", lowered) or "air condition" in lowered or "climate" in lowered:
            system = "ac"
        else:
            system = None
        if system and is_on and not is_off:
            entities["mode"] = f"{system}_on"
        elif system and is_off and not is_on:
            entities["mode"] = f"{system}_off"

    return entities
