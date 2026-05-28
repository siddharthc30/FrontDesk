"""Deterministic deictic-reference resolution.

Runs BEFORE the router LLM. If the user's question contains a deictic
city reference ("my city", "here", ...), we either substitute the
user's selected city literally into the text, or — if no city was
supplied — signal an honest decline. Proximity phrases ("near me",
"within X km", ...) are left untouched so the router still routes them
through lat/lng + radius_km.

This module is transport-agnostic (no FastAPI / Streamlit imports) per
BUILD_PLAN §2 #1.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Order matters: longer phrases first, so "in my city" matches before "my city".
_CITY_DEICTIC_PHRASES: tuple[str, ...] = (
    "in this city",
    "in my city",
    "in my town",
    "in town",
    "around here",
    "near here",
    "this city",
    "my city",
    "here",
)

# Phrases that mean PROXIMITY, not city-scope. If any of these appear, do not
# treat "here"-style words as city deictics (e.g. "near me here in Soho").
_PROXIMITY_PHRASES: tuple[str, ...] = (
    "near me",
    "close to me",
    "walking distance",
    "nearby",
    "around me",
)
_WITHIN_DISTANCE_RE = re.compile(
    r"\bwithin\s+\d+(?:\.\d+)?\s*(?:km|kilometers?|mi|miles?|m)\b",
    re.IGNORECASE,
)

# Cities the corpus actually covers. If the question already names one of
# these literally, the deictic phrase is redundant — pass the question through
# unchanged. This is what makes the voice-mode case ("...in my city Paris?")
# behave the same as a typed question that already names a city.
KNOWN_CITIES: tuple[str, ...] = (
    "London", "Paris", "Barcelona", "Milan", "Vienna", "Amsterdam",
)
_KNOWN_CITY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(c) for c in KNOWN_CITIES) + r")\b",
    re.IGNORECASE,
)


def _compile_phrase(phrase: str) -> re.Pattern[str]:
    # \b only anchors on word chars; phrases here are all word-bounded.
    return re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)


_CITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (p, _compile_phrase(p)) for p in _CITY_DEICTIC_PHRASES
)
_PROXIMITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    _compile_phrase(p) for p in _PROXIMITY_PHRASES
)


class DeicticResolution(NamedTuple):
    """Result of resolving deictic references in a question.

    - `question`: the (possibly rewritten) question text to hand to the router.
    - `decline_reason`: if non-None, the pipeline should short-circuit to an
      honest decline instead of calling the router.
    - `rewritten`: True iff we replaced a city deictic with the user's city.
    """
    question: str
    decline_reason: str | None
    rewritten: bool


def _has_proximity_intent(text: str) -> bool:
    if _WITHIN_DISTANCE_RE.search(text):
        return True
    return any(p.search(text) for p in _PROXIMITY_PATTERNS)


def find_city_deictic(text: str) -> tuple[str, re.Pattern[str]] | None:
    """Return (phrase, pattern) for the first city-deictic match, else None.

    Skips matches where the only deictic is the bare word "here" if a
    proximity phrase is also present — that combination is proximity, not
    city-scope (e.g. "near me here").
    """
    has_proximity = _has_proximity_intent(text)
    for phrase, pattern in _CITY_PATTERNS:
        if not pattern.search(text):
            continue
        if phrase == "here" and has_proximity:
            continue
        return phrase, pattern
    return None


def resolve_deictic_city(question: str, user_city: str | None) -> DeicticResolution:
    """Resolve deictic city references in `question` against `user_city`.

    Behaviour:
      - No city deictic present → return question unchanged, no decline.
      - City deictic present + user_city provided → substitute the matched
        phrase with "in {user_city}" so the router sees a named city.
      - City deictic present + user_city missing → return a decline reason;
        the pipeline should not call the router.
      - Proximity phrases ("near me", "within X km", ...) are never rewritten.
    """
    if not question:
        return DeicticResolution(question, None, False)

    match = find_city_deictic(question)
    if match is None:
        return DeicticResolution(question, None, False)

    # If the question already names a known city literally, the deictic is
    # redundant. Let it through unchanged so the router routes on the named
    # city (this is what makes voice transcripts like "in my city Paris" work).
    if _KNOWN_CITY_RE.search(question):
        return DeicticResolution(question, None, False)

    if not user_city:
        return DeicticResolution(
            question,
            "Please select a city to use 'my city' (or similar). "
            "I don't know which city you mean.",
            False,
        )

    _, pattern = match
    rewritten = pattern.sub(f"in {user_city}", question, count=1)
    return DeicticResolution(rewritten, None, True)
