"""Validation of annotated examples against the DriveMind schema.

Catches the failure modes explicitly called out in the spec: missing
labels, invalid intents, malformed entities, duplicates, inconsistent
annotation, and invalid parameter values.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from src.annotation.schema import ENTITY_TYPES, AnnotatedExample
from src.config.settings import load_intents

# Reasonable value ranges for numeric entities. Anything outside is flagged
# as an invalid parameter value (not silently accepted).
ENTITY_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (16, 30),
    "minimum_power_kw": (0, 400),
    "volume": (0, 100),
}


@dataclass
class ValidationIssue:
    example_id: str
    kind: str  # e.g. "invalid_intent", "malformed_entity", "duplicate"
    detail: str


@dataclass
class ValidationReport:
    total: int = 0
    valid: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def invalid(self) -> int:
        return self.total - self.valid

    def issue_counts(self) -> Counter:
        return Counter(issue.kind for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "issue_counts": dict(self.issue_counts()),
            "issues": [issue.__dict__ for issue in self.issues],
        }


def _validate_entities(intent: str, entities: dict[str, Any], intents_cfg: dict) -> list[str]:
    """Return a list of problem descriptions for one example's entities."""
    problems: list[str] = []
    allowed = set(intents_cfg.get(intent, {}).get("entities", []))

    for key, value in entities.items():
        if key not in ENTITY_TYPES:
            problems.append(f"unknown entity key '{key}'")
            continue
        if key not in allowed:
            problems.append(f"entity '{key}' is not valid for intent '{intent}'")
            continue
        expected_type = ENTITY_TYPES[key]
        if not isinstance(value, expected_type):
            problems.append(
                f"entity '{key}' expected type {expected_type.__name__}, got {type(value).__name__}"
            )
            continue
        if key in ENTITY_RANGES:
            lo, hi = ENTITY_RANGES[key]
            if not (lo <= value <= hi):
                problems.append(f"entity '{key}'={value} out of valid range [{lo}, {hi}]")
    return problems


def validate_dataset(records: list[dict[str, Any]]) -> ValidationReport:
    """Validate a list of raw dict records (as loaded from CSV/JSONL).

    This performs schema validation (via pydantic) plus dataset-level
    checks (duplicates) that only make sense across the whole collection.
    """
    intents_cfg = load_intents()
    valid_intents = set(intents_cfg.keys())

    report = ValidationReport(total=len(records))
    seen_texts: Counter = Counter()

    for rec in records:
        example_id = str(rec.get("id", "<unknown>"))
        problems: list[str] = []

        # 1. Schema-level validation (missing fields, empty text, bad split).
        try:
            example = AnnotatedExample.model_validate(rec)
        except ValidationError as exc:
            for err in exc.errors():
                report.issues.append(
                    ValidationIssue(example_id, "schema_error", f"{err['loc']}: {err['msg']}")
                )
            continue

        # 2. Intent must be part of the known taxonomy.
        if example.intent not in valid_intents:
            problems.append(f"unknown intent '{example.intent}'")

        # 3. Entity validation (type + allowed-for-intent + range).
        if example.intent in valid_intents:
            problems.extend(_validate_entities(example.intent, example.entities, intents_cfg))

        # 4. Duplicate detection (exact text match).
        normalized = example.text.strip().lower()
        seen_texts[normalized] += 1
        if seen_texts[normalized] > 1:
            problems.append(f"duplicate text (seen {seen_texts[normalized]}x)")

        if problems:
            for p in problems:
                kind = (
                    "malformed_entity"
                    if "entity" in p
                    else (
                        "invalid_intent"
                        if "intent" in p
                        else ("duplicate" if "duplicate" in p else "other")
                    )
                )
                report.issues.append(ValidationIssue(example_id, kind, p))
        else:
            report.valid += 1

    return report
