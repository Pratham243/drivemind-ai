"""Generate the synthetic DriveMind automotive NLU dataset.

Produces natural-language variation per intent (short/long, informal,
polite, ambiguous, paraphrased, misspelled) with ground-truth intent +
entity annotations, then writes:

  - data/raw/dataset.csv        (text, intent, split — no entities)
  - data/annotated/dataset.jsonl (full annotation: id, text, intent,
    entities, split, source, difficulty)

Deterministic given the configured random seed (configs/config.yaml).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import load_yaml_config  # noqa: E402

# ---------------------------------------------------------------------------
# Entity value pools
# ---------------------------------------------------------------------------

LOCATIONS = [
    "Berlin",
    "Munich",
    "Hamburg",
    "Frankfurt",
    "Cologne",
    "Stuttgart",
    "Dusseldorf",
    "Leipzig",
    "Dresden",
    "Nuremberg",
    "Bremen",
    "Hannover",
    "Vienna",
    "Zurich",
    "Amsterdam",
    "Paris",
    "downtown",
    "the city center",
]

DESTINATIONS = LOCATIONS + [
    "Berlin Brandenburg Airport",
    "Munich Airport",
    "the nearest gas station",
    "my office",
    "the train station",
    "Frankfurt Airport",
    "home",
]

GENRES = [
    "jazz",
    "rock",
    "pop",
    "classical",
    "hip hop",
    "techno",
    "blues",
    "country",
    "reggae",
    "EDM",
    "metal",
    "R&B",
]

ARTISTS = [
    "Taylor Swift",
    "Daft Punk",
    "Mozart",
    "Beethoven",
    "Ed Sheeran",
    "Beyonce",
    "Coldplay",
    "The Beatles",
]

CONTACTS = [
    "Mom",
    "Dad",
    "John",
    "Sarah",
    "my wife",
    "my husband",
    "my boss",
    "Alex",
    "Julia",
    "Peter",
    "Anna",
]

WINDOWS = [
    "front left window",
    "front right window",
    "rear left window",
    "rear right window",
    "all windows",
    "driver window",
    "passenger window",
]

SEATS = ["driver seat", "passenger seat", "my seat", "the back seat"]

COLORS = ["blue", "red", "green", "purple", "white", "warm white", "orange"]

CUISINES = [
    "Italian",
    "Chinese",
    "Indian",
    "Turkish",
    "Mexican",
    "vegan",
    "sushi",
    "Thai",
    "Greek",
    "Vietnamese",
]

TIRES = ["front left", "front right", "rear left", "rear right", "all"]

KW_VALUES = [50, 100, 150, 250, 350]

AMOUNTS = ["a bit", "a little", "a lot", "slightly", "a few degrees"]

TEMPERATURES = list(range(17, 29))

VOLUMES = list(range(10, 100, 10))

POLITE_PREFIXES = ["Could you please ", "Would you mind ", "Can you ", "Please ", ""]
INFORMAL_PREFIXES = ["Hey, ", "Yo, ", "Uh, ", "So, ", ""]


def _typo(text: str, rng: random.Random) -> str:
    """Introduce a light, realistic spelling variation into text."""
    if len(text) < 6:
        return text
    idx = rng.randint(1, len(text) - 2)
    chars = list(text)
    swap_type = rng.choice(["drop", "double", "swap"])
    if swap_type == "drop":
        del chars[idx]
    elif swap_type == "double":
        chars.insert(idx, chars[idx])
    elif swap_type == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Templates: intent -> list of (template, difficulty, entity_fillers)
# entity_fillers is a dict of placeholder -> (entity_key, pool) used to
# fill and record ground-truth entities for that template instance.
# ---------------------------------------------------------------------------


def build_templates(rng: random.Random) -> dict[str, list[tuple[str, str, dict]]]:
    return {
        "increase_temperature": [
            ("Make it warmer", "easy", {}),
            ("It's cold in here", "hard", {}),
            ("Turn up the heat", "easy", {}),
            (
                "Can you increase the temperature {amount}?",
                "medium",
                {"amount": ("amount", AMOUNTS)},
            ),
            ("I'm feeling cold, warm it up please", "medium", {}),
            ("Bump the temperature up", "medium", {}),
            ("A little warmer would be nice", "hard", {}),
            ("Heat it up in here", "easy", {}),
        ],
        "decrease_temperature": [
            ("Make it cooler", "easy", {}),
            ("It's too hot in here", "hard", {}),
            ("Turn down the heat", "easy", {}),
            (
                "Can you decrease the temperature {amount}?",
                "medium",
                {"amount": ("amount", AMOUNTS)},
            ),
            ("I'm feeling hot, cool it down please", "medium", {}),
            ("Lower the temperature a bit", "medium", {}),
            ("Cool it down in here", "easy", {}),
        ],
        "set_temperature": [
            (
                "Set the temperature to {temperature} degrees",
                "easy",
                {"temperature": ("temperature", TEMPERATURES)},
            ),
            (
                "Set the cabin temperature to {temperature}",
                "medium",
                {"temperature": ("temperature", TEMPERATURES)},
            ),
            (
                "Change the temperature to {temperature} degrees please",
                "medium",
                {"temperature": ("temperature", TEMPERATURES)},
            ),
            (
                "I want {temperature} degrees in here",
                "hard",
                {"temperature": ("temperature", TEMPERATURES)},
            ),
            (
                "Adjust temperature to exactly {temperature}",
                "medium",
                {"temperature": ("temperature", TEMPERATURES)},
            ),
        ],
        "climate_control": [
            ("Turn on the AC", "easy", {}),
            ("Turn off the air conditioning", "easy", {}),
            ("Switch on climate control", "medium", {}),
            ("Turn on the fan", "easy", {}),
            ("Can you turn off the heater", "medium", {}),
        ],
        "vehicle_status": [
            ("What's my vehicle status?", "easy", {}),
            ("How is my car doing?", "medium", {}),
            ("Give me a status update on the car", "medium", {}),
            ("Show me the car's current status", "easy", {}),
            ("Is everything okay with the vehicle?", "hard", {}),
        ],
        "battery_status": [
            ("What's my battery level?", "easy", {}),
            ("How much battery do I have?", "easy", {}),
            ("Check the battery please", "medium", {}),
            ("How much charge is left?", "medium", {}),
            ("Battery status", "easy", {}),
            ("Tell me the current battery percentage", "medium", {}),
        ],
        "range_query": [
            ("How far can I drive?", "easy", {}),
            ("What's my remaining range?", "easy", {}),
            ("How many kilometers can I go?", "medium", {}),
            ("Do I have enough range to get home?", "hard", {}),
            ("What's my driving range right now?", "medium", {}),
        ],
        "tire_pressure": [
            ("What's my tire pressure?", "easy", {}),
            ("Check the {tire} tire pressure", "medium", {"tire": ("tire", TIRES)}),
            ("Are my tires okay?", "hard", {}),
            ("Show tire pressure for {tire}", "medium", {"tire": ("tire", TIRES)}),
        ],
        "find_charging_station": [
            ("Where is the nearest charger?", "easy", {}),
            (
                "Find a charging station near {location}",
                "medium",
                {"location": ("location", LOCATIONS)},
            ),
            (
                "Find a charging station near {location} with at least {kw} kW",
                "hard",
                {"location": ("location", LOCATIONS), "kw": ("minimum_power_kw", KW_VALUES)},
            ),
            (
                "I need to charge, find a station near {location}",
                "medium",
                {"location": ("location", LOCATIONS)},
            ),
            (
                "Look for a {kw} kW charger nearby",
                "medium",
                {"kw": ("minimum_power_kw", KW_VALUES)},
            ),
            (
                "Are there any charging points around {location}?",
                "hard",
                {"location": ("location", LOCATIONS)},
            ),
        ],
        "find_parking": [
            ("Find parking near {location}", "medium", {"location": ("location", LOCATIONS)}),
            ("Where can I park around here?", "easy", {}),
            (
                "Is there a parking spot near {location}?",
                "medium",
                {"location": ("location", LOCATIONS)},
            ),
            ("Find me a parking garage", "easy", {}),
        ],
        "start_navigation": [
            ("Navigate to {destination}", "easy", {"destination": ("destination", DESTINATIONS)}),
            ("Take me to {destination}", "easy", {"destination": ("destination", DESTINATIONS)}),
            (
                "Start navigation to {destination}",
                "easy",
                {"destination": ("destination", DESTINATIONS)},
            ),
            (
                "I need directions to {destination}",
                "medium",
                {"destination": ("destination", DESTINATIONS)},
            ),
            (
                "Can you route me to {destination}?",
                "medium",
                {"destination": ("destination", DESTINATIONS)},
            ),
            ("Let's go to {destination}", "hard", {"destination": ("destination", DESTINATIONS)}),
        ],
        "cancel_navigation": [
            ("Cancel navigation", "easy", {}),
            ("Stop the directions", "medium", {}),
            ("I don't need directions anymore", "hard", {}),
            ("End the route guidance", "medium", {}),
        ],
        "navigation": [
            ("What's the fastest route?", "medium", {}),
            ("How long until we arrive?", "medium", {}),
            ("What's my ETA?", "easy", {}),
            ("Which way should I go?", "hard", {}),
        ],
        "traffic_query": [
            ("How's the traffic?", "easy", {}),
            (
                "What's the traffic like near {location}?",
                "medium",
                {"location": ("location", LOCATIONS)},
            ),
            ("Is there a traffic jam ahead?", "medium", {}),
            ("Any delays on my route?", "hard", {}),
        ],
        "play_music": [
            ("Play some {genre}", "easy", {"genre": ("music_genre", GENRES)}),
            ("Play music", "easy", {}),
            ("Play {artist}", "medium", {"artist": ("artist", ARTISTS)}),
            ("Put on some {genre} music", "medium", {"genre": ("music_genre", GENRES)}),
            ("I feel like listening to {genre}", "hard", {"genre": ("music_genre", GENRES)}),
            ("Can you play something by {artist}?", "medium", {"artist": ("artist", ARTISTS)}),
        ],
        "pause_music": [
            ("Pause the music", "easy", {}),
            ("Stop playing music", "easy", {}),
            ("Pause playback", "medium", {}),
            ("Can you pause that?", "hard", {}),
        ],
        "next_song": [
            ("Next song", "easy", {}),
            ("Skip this track", "easy", {}),
            ("Play the next one", "medium", {}),
            ("I don't like this song, skip it", "hard", {}),
        ],
        "change_volume": [
            ("Turn up the volume", "easy", {}),
            ("Turn down the volume", "easy", {}),
            ("Set volume to {volume}", "medium", {"volume": ("volume", VOLUMES)}),
            ("Make it louder", "medium", {}),
            ("Lower the volume a bit", "medium", {}),
            ("It's too loud", "hard", {}),
        ],
        "make_phone_call": [
            ("Call {contact}", "easy", {"contact": ("phone_contact", CONTACTS)}),
            ("Phone {contact}", "easy", {"contact": ("phone_contact", CONTACTS)}),
            ("Can you call {contact} for me?", "medium", {"contact": ("phone_contact", CONTACTS)}),
            ("Dial {contact}'s number", "medium", {"contact": ("phone_contact", CONTACTS)}),
            (
                "I need to talk to {contact}, call them",
                "hard",
                {"contact": ("phone_contact", CONTACTS)},
            ),
        ],
        "send_message": [
            ("Send a message to {contact}", "easy", {"contact": ("phone_contact", CONTACTS)}),
            (
                "Text {contact} that I'm on my way",
                "medium",
                {"contact": ("phone_contact", CONTACTS)},
            ),
            ("Send {contact} a text", "easy", {"contact": ("phone_contact", CONTACTS)}),
            (
                "Can you message {contact} for me?",
                "medium",
                {"contact": ("phone_contact", CONTACTS)},
            ),
        ],
        "open_window": [
            ("Open the {window}", "easy", {"window": ("window", WINDOWS)}),
            ("Roll down the {window}", "medium", {"window": ("window", WINDOWS)}),
            ("Can you open the {window}?", "medium", {"window": ("window", WINDOWS)}),
            ("It's stuffy, open a window", "hard", {}),
        ],
        "close_window": [
            ("Close the {window}", "easy", {"window": ("window", WINDOWS)}),
            ("Roll up the {window}", "medium", {"window": ("window", WINDOWS)}),
            ("Can you close the {window}?", "medium", {"window": ("window", WINDOWS)}),
            ("It's windy, close the windows", "hard", {}),
        ],
        "seat_adjustment": [
            ("Adjust my {seat}", "medium", {"seat": ("seat", SEATS)}),
            ("Heat up the {seat}", "medium", {"seat": ("seat", SEATS)}),
            ("Move the {seat} back", "medium", {"seat": ("seat", SEATS)}),
            ("Turn on seat heating", "easy", {}),
        ],
        "ambient_lighting": [
            ("Change the ambient lighting to {color}", "medium", {"color": ("color", COLORS)}),
            ("Set the interior lights to {color}", "medium", {"color": ("color", COLORS)}),
            ("Can you make the lights {color}?", "medium", {"color": ("color", COLORS)}),
            ("Dim the ambient lighting", "hard", {}),
        ],
        "restaurant_search": [
            (
                "Find me an {cuisine} restaurant nearby",
                "medium",
                {"cuisine": ("restaurant_cuisine", CUISINES)},
            ),
            ("I'm hungry, find a restaurant", "easy", {}),
            (
                "Search for {cuisine} food near {location}",
                "hard",
                {"cuisine": ("restaurant_cuisine", CUISINES), "location": ("location", LOCATIONS)},
            ),
            ("Any good restaurants around here?", "easy", {}),
            ("Find a place to eat", "easy", {}),
        ],
        "weather_query": [
            ("What's the weather like?", "easy", {}),
            ("What's the weather in {location}?", "medium", {"location": ("location", LOCATIONS)}),
            ("Is it going to rain today?", "hard", {}),
            ("Do I need an umbrella?", "hard", {}),
        ],
        "general_question": [
            ("What can you do?", "medium", {}),
            ("Tell me a joke", "hard", {}),
            ("What time is it?", "medium", {}),
            ("How are you today?", "hard", {}),
            ("Who made you?", "hard", {}),
            ("What's the meaning of life?", "hard", {}),
        ],
    }


def _apply_style(text: str, rng: random.Random) -> str:
    """Apply a random stylistic wrapper: polite / informal / plain / typo."""
    style = rng.choice(["plain", "polite", "informal", "typo", "plain"])
    if style == "polite":
        prefix = rng.choice(POLITE_PREFIXES)
        if prefix:
            lowered = text[0].lower() + text[1:]
            text = f"{prefix}{lowered}"
            if not text.endswith("?") and prefix.strip() in {
                "Could you please",
                "Can you",
                "Would you mind",
            }:
                text = text.rstrip(".") + "?"
    elif style == "informal":
        prefix = rng.choice(INFORMAL_PREFIXES)
        text = f"{prefix}{text}"
    elif style == "typo":
        text = _typo(text, rng)
    return text


def generate_examples(rng: random.Random, per_intent_target: dict[str, int]) -> list[dict]:
    templates = build_templates(rng)
    examples: list[dict] = []
    ex_id = 0

    for intent, target_count in per_intent_target.items():
        intent_templates = templates[intent]
        generated = 0
        attempts = 0
        seen: set[str] = set()
        while generated < target_count and attempts < target_count * 20:
            attempts += 1
            template, difficulty, fillers = rng.choice(intent_templates)
            entities: dict = {}
            text = template
            for placeholder, (entity_key, pool) in fillers.items():
                value = rng.choice(pool)
                text = text.replace(f"{{{placeholder}}}", str(value))
                entities[entity_key] = value
            text = _apply_style(text, rng)

            key = text.lower()
            if key in seen and rng.random() < 0.9:
                continue  # mostly avoid exact duplicates, allow a few (realistic noise)
            seen.add(key)

            ex_id += 1
            examples.append(
                {
                    "id": f"ex_{ex_id:05d}",
                    "text": text,
                    "intent": intent,
                    "entities": entities,
                    "source": "synthetic_template",
                    "difficulty": difficulty,
                }
            )
            generated += 1

    return examples


def assign_splits(examples: list[dict], rng: random.Random, train: float, val: float) -> None:
    """Stratified split assignment, mutating examples in place."""
    by_intent: dict[str, list[dict]] = {}
    for ex in examples:
        by_intent.setdefault(ex["intent"], []).append(ex)

    for intent_examples in by_intent.values():
        rng.shuffle(intent_examples)
        n = len(intent_examples)
        n_train = max(1, int(n * train))
        n_val = max(1, int(n * val)) if n > 2 else 0
        for i, ex in enumerate(intent_examples):
            if i < n_train:
                ex["split"] = "train"
            elif i < n_train + n_val:
                ex["split"] = "val"
            else:
                ex["split"] = "test"


def main() -> None:
    cfg = load_yaml_config("config.yaml")["dataset"]
    seed = cfg["seed"]
    rng = random.Random(seed)

    # Mildly imbalanced per-intent target counts (realistic, not uniform).
    intent_names = list(build_templates(rng).keys())
    base_counts = {
        "battery_status": 160,
        "set_temperature": 150,
        "find_charging_station": 150,
        "start_navigation": 140,
        "play_music": 140,
        "vehicle_status": 120,
        "increase_temperature": 110,
        "decrease_temperature": 110,
        "range_query": 110,
        "change_volume": 110,
        "make_phone_call": 100,
        "send_message": 100,
        "open_window": 95,
        "close_window": 95,
        "restaurant_search": 100,
        "weather_query": 95,
        "traffic_query": 85,
        "find_parking": 85,
        "cancel_navigation": 70,
        "navigation": 80,
        "pause_music": 75,
        "next_song": 75,
        "tire_pressure": 70,
        "seat_adjustment": 70,
        "ambient_lighting": 70,
        "climate_control": 90,
        "general_question": 60,
    }
    per_intent_target = {name: base_counts.get(name, 80) for name in intent_names}

    examples = generate_examples(rng, per_intent_target)
    assign_splits(examples, rng, cfg["train_split"], cfg["val_split"])

    raw_path = PROJECT_ROOT / cfg["raw_path"]
    annotated_path = PROJECT_ROOT / cfg["annotated_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(examples)
    df_raw = df[["id", "text", "intent", "split"]]
    df_raw.to_csv(raw_path, index=False)

    with open(annotated_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} examples across {len(per_intent_target)} intents.")
    print(f"Raw dataset:       {raw_path}")
    print(f"Annotated dataset: {annotated_path}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
