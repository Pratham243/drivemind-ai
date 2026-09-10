"""Restaurant search tool (deterministic simulated directory)."""

from __future__ import annotations

import hashlib

from src.tools.base import ToolResult

_NAME_TEMPLATES = [
    "{cuisine} House",
    "The {cuisine} Kitchen",
    "{cuisine} Corner",
    "Bella {cuisine}",
]


def search_restaurants(
    location: str | None = None, restaurant_cuisine: str | None = None
) -> ToolResult:
    location = location or "your current location"
    cuisine = (restaurant_cuisine or "local").title()

    results = []
    for i, template in enumerate(_NAME_TEMPLATES):
        digest = hashlib.sha256(f"{location}_{cuisine}_{i}".encode()).hexdigest()
        rating = round(3.0 + (int(digest[:2], 16) % 20) / 10, 1)
        distance_km = round((int(digest[2:4], 16) % 50) / 10, 1)
        results.append(
            {
                "name": template.format(cuisine=cuisine),
                "cuisine": cuisine,
                "location": location,
                "rating": rating,
                "distance_km": distance_km,
            }
        )

    results.sort(key=lambda r: r["distance_km"])
    return ToolResult.ok(
        location=location, cuisine=cuisine, count=len(results), restaurants=results
    )
