"""Tests for the annotation validator and data-quality analyzer, run
against small synthetic fixtures (not the full generated dataset) so
they run fast and test edge cases precisely.
"""

import pandas as pd

from src.annotation.quality import analyze_dataset
from src.annotation.validator import validate_dataset


def test_validator_accepts_well_formed_records():
    records = [
        {
            "id": "1",
            "text": "Set the temperature to 22 degrees",
            "intent": "set_temperature",
            "entities": {"temperature": 22},
            "split": "train",
            "source": "synthetic_template",
            "difficulty": "easy",
        },
    ]
    report = validate_dataset(records)
    assert report.valid == 1
    assert report.invalid == 0


def test_validator_flags_unknown_intent():
    records = [
        {
            "id": "1",
            "text": "Do a barrel roll",
            "intent": "not_a_real_intent",
            "entities": {},
            "split": "train",
            "source": "x",
            "difficulty": "easy",
        },
    ]
    report = validate_dataset(records)
    assert report.invalid == 1
    assert "invalid_intent" in report.issue_counts()


def test_validator_flags_malformed_entity_type():
    records = [
        {
            "id": "1",
            "text": "Set the temperature to 22 degrees",
            "intent": "set_temperature",
            "entities": {"temperature": "warm"},
            "split": "train",
            "source": "x",
            "difficulty": "easy",
        },
    ]
    report = validate_dataset(records)
    assert report.invalid == 1
    assert "malformed_entity" in report.issue_counts()


def test_validator_flags_out_of_range_entity():
    records = [
        {
            "id": "1",
            "text": "Set the temperature to 100 degrees",
            "intent": "set_temperature",
            "entities": {"temperature": 100},
            "split": "train",
            "source": "x",
            "difficulty": "easy",
        },
    ]
    report = validate_dataset(records)
    assert report.invalid == 1


def test_validator_flags_duplicates():
    records = [
        {
            "id": "1",
            "text": "What's my battery level?",
            "intent": "battery_status",
            "entities": {},
            "split": "train",
            "source": "x",
            "difficulty": "easy",
        },
        {
            "id": "2",
            "text": "what's my battery level?",
            "intent": "battery_status",
            "entities": {},
            "split": "train",
            "source": "x",
            "difficulty": "easy",
        },
    ]
    report = validate_dataset(records)
    assert "duplicate" in report.issue_counts()


def test_validator_flags_missing_text():
    records = [
        {
            "id": "1",
            "text": "",
            "intent": "battery_status",
            "entities": {},
            "split": "train",
            "source": "x",
            "difficulty": "easy",
        },
    ]
    report = validate_dataset(records)
    assert report.invalid == 1


def test_data_quality_report_computes_real_stats():
    df = pd.DataFrame(
        [
            {
                "id": "1",
                "text": "Play some jazz",
                "intent": "play_music",
                "entities": {"music_genre": "jazz"},
            },
            {
                "id": "2",
                "text": "Play some rock",
                "intent": "play_music",
                "entities": {"music_genre": "rock"},
            },
            {
                "id": "3",
                "text": "What's my battery level?",
                "intent": "battery_status",
                "entities": {},
            },
        ]
    )
    report = analyze_dataset(df)
    assert report.total_samples == 3
    assert report.class_distribution["play_music"] == 2
    assert report.class_distribution["battery_status"] == 1
    assert report.duplicate_samples == 0


def test_data_quality_report_detects_duplicates_and_imbalance():
    df = pd.DataFrame(
        [
            {"id": "1", "text": "Play music", "intent": "play_music", "entities": {}},
            {"id": "2", "text": "Play music", "intent": "play_music", "entities": {}},
            {"id": "3", "text": "Battery?", "intent": "battery_status", "entities": {}},
        ]
    )
    report = analyze_dataset(df)
    assert report.duplicate_samples == 1
    assert report.class_imbalance_ratio == 2.0
