"""Agent planning: intent -> tool selection, and multi-step decomposition.

Multi-step handling (PHASE 15) is deliberately simple and rule-based
rather than LLM-driven: split the utterance on coordinating conjunctions
("and", ",", ";") into sub-requests, run NLU on each independently, and
keep the ones that resolve to a distinct actionable (tool-bearing) intent.
This is transparent, fast, free, and covers the example scenarios in the
spec ("What's my battery level and find the nearest charger") without
depending on an LLM being configured.
"""

from __future__ import annotations

import re

from src.config.settings import load_intents

_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b)\s*", re.IGNORECASE)


def tool_for_intent(intent: str) -> str | None:
    intents_cfg = load_intents()
    return intents_cfg.get(intent, {}).get("tool")


def split_into_subrequests(text: str) -> list[str]:
    """Split a compound utterance into candidate sub-requests.

    Falls back to returning the whole text as a single sub-request if
    splitting would produce fragments too short to be meaningful.
    """
    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 2]
    if len(parts) < 2:
        return [text]
    return parts
