"""Evaluation dataset for the hotel NL search pipeline.

Each EvalItem captures one test question, the expected router path, optional
user coordinates, machine-checkable assertions on the result, and human notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalItem:
    id: str                           # unique identifier, e.g. "param_01"
    question: str                     # the natural language question
    expected_path: str                # "parameterized" | "semantic" | "review_search" | "fallback" | "declined"
    user_lat: Optional[float] = None  # for geo queries
    user_lng: Optional[float] = None
    assertions: Optional[dict] = None # machine-checkable assertions on the result
    notes: str = ""                   # human-readable explanation of what this tests


# ── Assertion key reference ───────────────────────────────────────────────────
#
# Result-level checks:
#   "min_hotels": N         — expect at least N hotels returned
#   "max_hotels": N         — expect at most N hotels
#   "exact_hotels": N       — expect exactly N hotels/rows
#
# Column-value checks on every returned hotel:
#   "all_city": "Paris"     — every hotel must have this city (case-insensitive)
#   "all_min_score": 8.0    — every hotel must have avg_score >= this
#   "all_has_amenity": "pool" — every hotel must have has_pool == 1
#
# Checks on the answer text:
#   "answer_contains": [..] — all substrings must appear (case-insensitive)
#   "answer_not_contains": [..] — none of the substrings may appear
#
# Decline checks:
#   "should_decline": True  — expects declined=True
#
# ─────────────────────────────────────────────────────────────────────────────

EVAL_DATASET: list[EvalItem] = [

    # === PARAMETERIZED PATH ===
    EvalItem(
        id="param_01",
        question="Show me the top 5 hotels in Paris",
        expected_path="parameterized",
        assertions={
            "min_hotels": 1,
            "max_hotels": 5,
            "all_city": "Paris",
        },
        notes="Basic city filter + limit. Most common query shape.",
    ),
    EvalItem(
        id="param_02",
        question="Best rated hotels in London with score above 9",
        expected_path="parameterized",
        assertions={
            "min_hotels": 1,
            "all_city": "London",
            "all_min_score": 9.0,
        },
        notes="City filter + min rating + sort by score.",
    ),
    EvalItem(
        id="param_03",
        question="Hotels near me within 5km",
        expected_path="parameterized",
        user_lat=48.8566,
        user_lng=2.3522,
        assertions={
            "min_hotels": 1,
        },
        notes="Geo query with user coordinates (central Paris).",
    ),
    EvalItem(
        id="param_04",
        question="Top 3 cheapest hotels in Barcelona",
        expected_path="parameterized",
        assertions={
            "max_hotels": 3,
            "all_city": "Barcelona",
        },
        notes="City + price sort + limit.",
    ),
    EvalItem(
        id="param_05",
        question="Hotels in Amsterdam with a pool",
        expected_path="parameterized",
        assertions={
            "min_hotels": 1,
            "all_city": "Amsterdam",
            "all_has_amenity": "pool",
        },
        notes="City + amenity filter.",
    ),

    # === SEMANTIC PATH ===
    # Note: semantic queries return aggregate rows (counts, averages) that never
    # become HotelRow objects in the pipeline. Don't use min_hotels assertions here;
    # routing_correct + has_query_ran are the meaningful checks for this path.
    EvalItem(
        id="sem_01",
        question="How many hotels are in each city?",
        expected_path="semantic",
        assertions={
            "answer_contains": ["London", "Paris"],
        },
        notes="Count + group_by city. Answer must mention at least two cities.",
    ),
    EvalItem(
        id="sem_02",
        question="What is the average review score in Barcelona?",
        expected_path="semantic",
        assertions={
            "answer_contains": ["Barcelona"],
        },
        notes="Single aggregation with city filter.",
    ),
    EvalItem(
        id="sem_03",
        question="Which city has the highest average hotel rating?",
        expected_path="semantic",
        assertions={},
        notes="Group-by + avg + sort descending.",
    ),
    EvalItem(
        id="sem_04",
        question="What is the average price per night by city?",
        expected_path="semantic",
        assertions={},
        notes="Avg price grouped by city.",
    ),
    EvalItem(
        id="sem_05",
        question="How many hotels have a pool?",
        expected_path="semantic",
        assertions={},
        notes="Simple count with amenity filter.",
    ),

    # === REVIEW SEARCH PATH ===
    EvalItem(
        id="review_01",
        question="Hotels where guests mention a rooftop terrace",
        expected_path="review_search",
        assertions={
            "min_hotels": 0,
        },
        notes="Novel aspect not in precomputed taxonomy. Tests FTS5 review search path.",
    ),
    EvalItem(
        id="review_02",
        question="Which hotels have good views?",
        expected_path="review_search",
        assertions={},
        notes="Qualitative aspect search through review text.",
    ),

    # === FALLBACK PATH (text_to_sql) ===
    EvalItem(
        id="fall_01",
        question="Show me hotels with 'Grand' in the name",
        expected_path="fallback",
        assertions={
            "min_hotels": 1,
        },
        notes="LIKE query on hotel name — not a standard filter, needs raw SQL.",
    ),
    EvalItem(
        id="fall_02",
        question="Which hotel has the most reviews?",
        expected_path="fallback",
        assertions={
            "min_hotels": 1,
            "max_hotels": 1,
        },
        notes="ORDER BY total_reviews DESC LIMIT 1.",
    ),
    EvalItem(
        id="fall_03",
        question="Hotels in Paris or Barcelona with score above 9",
        expected_path="fallback",
        assertions={
            "min_hotels": 1,
            "all_min_score": 9.0,
        },
        notes="Multi-city OR + score filter. Too complex for parameterized.",
    ),
    EvalItem(
        id="fall_04",
        question="What is the most common reviewer nationality for Paris hotels?",
        expected_path="fallback",
        assertions={},
        notes="Needs JOIN + GROUP BY + COUNT on reviewer_nationality from reviews table.",
    ),

    # === DECLINE ===
    EvalItem(
        id="decline_01",
        question="What's the weather like in London?",
        expected_path="declined",
        assertions={
            "should_decline": True,
        },
        notes="Out of scope — not about hotel data.",
    ),
    EvalItem(
        id="decline_02",
        question="Book me a room at the Holiday Inn",
        expected_path="declined",
        assertions={
            "should_decline": True,
        },
        notes="Action request — system is read-only, can't book.",
    ),
    EvalItem(
        id="decline_03",
        question="What's the revenue of the Hilton?",
        expected_path="declined",
        assertions={
            "should_decline": True,
        },
        notes="Data ceiling — revenue isn't in the dataset.",
    ),
    EvalItem(
        id="decline_04",
        question="Tell me a joke",
        expected_path="declined",
        assertions={
            "should_decline": True,
        },
        notes="Not about hotels at all.",
    ),

    # === EDGE CASES ===
    EvalItem(
        id="edge_01",
        question="Hotels in Tokyo",
        expected_path="declined",  # router declines: Tokyo not in DB (London/Paris/etc only)
        assertions={
            "should_decline": True,
        },
        notes="City not in data. Router consistently declines rather than returning 0 results. "
              "Either outcome is acceptable but declined is what the system actually does.",
    ),
    EvalItem(
        id="edge_02",
        question="",
        expected_path="declined",
        assertions={
            "should_decline": True,
        },
        notes="Empty input. Should decline or error gracefully.",
    ),
]
