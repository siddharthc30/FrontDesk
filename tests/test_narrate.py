"""Narration grounding tests — narration must not assert a scope the rows don't satisfy."""
from __future__ import annotations

import asyncio
import json

from core import narrate as narrate_mod


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _multi_city_rows() -> list[dict]:
    return [
        {"hotel_id": 1, "name": "A", "city": "London",    "has_pool": 1, "avg_score": 8.6},
        {"hotel_id": 2, "name": "B", "city": "Paris",     "has_pool": 1, "avg_score": 8.5},
        {"hotel_id": 3, "name": "C", "city": "Barcelona", "has_pool": 1, "avg_score": 8.4},
        {"hotel_id": 4, "name": "D", "city": "Milan",     "has_pool": 1, "avg_score": 8.3},
        {"hotel_id": 5, "name": "E", "city": "Vienna",    "has_pool": 1, "avg_score": 8.2},
        {"hotel_id": 6, "name": "F", "city": "Amsterdam", "has_pool": 1, "avg_score": 8.1},
    ]


def test_narration_prompt_carries_grounding_rule():
    """The system prompt must explicitly forbid asserting unverified scope."""
    sp = narrate_mod.NARRATION_SYSTEM_PROMPT
    assert "GROUNDING RULE" in sp
    assert "EVERY row" in sp
    assert "in your city" in sp  # spelled out as a negative example


def test_narration_does_not_assert_single_city_scope_for_multi_city_rows(monkeypatch):
    """Spy on the prompt: it must (a) include row cities and (b) carry the rule."""
    seen = {}

    async def _fake_chat(messages, system=None, model_tier="main",
                         response_json=False, json_schema=None, temperature=0.0):
        seen["system"] = system
        seen["prompt"] = messages[-1]["content"]
        # Simulate a well-behaved model that obeys the grounding rule.
        return json.dumps({
            "answer": "The results span several cities — London, Paris, Barcelona, "
                      "Milan, Vienna, and Amsterdam — all with pools.",
            "insights": "Hotels with pools are distributed across all six cities.",
        })

    import core.llm as llm
    monkeypatch.setattr(llm, "chat_completion", _fake_chat)

    rows = _multi_city_rows()
    answer, insights = _run(narrate_mod.narrate_results(
        question="What are the hotels with pool in my city?",
        rows=rows,
        path="parameterized",
        query_ran='{"required_amenities":["has_pool"]}',
    ))

    # Sanity: rule was forwarded to the model.
    assert "GROUNDING RULE" in seen["system"]
    assert "Ground every claim in the rows above" in seen["prompt"]
    # All distinct cities appear in the prompt's row dump.
    for c in ("London", "Paris", "Barcelona", "Milan", "Vienna", "Amsterdam"):
        assert c in seen["prompt"]

    # The (simulated) narration does not assert a single-city scope.
    lowered = answer.lower()
    assert "in your city" not in lowered
    assert "in my city" not in lowered
    # And it acknowledges multiple cities.
    assert ("several cities" in lowered) or ("multiple cities" in lowered) \
        or sum(c.lower() in lowered for c in ("london", "paris", "barcelona", "milan", "vienna", "amsterdam")) >= 2


def test_narration_truncates_to_50_rows_and_surfaces_total(monkeypatch):
    """When >50 rows, narration must receive only 50 rows AND the true total."""
    seen = {}

    async def _fake_chat(messages, system=None, model_tier="main",
                         response_json=False, json_schema=None, temperature=0.0):
        seen["prompt"] = messages[-1]["content"]
        return json.dumps({
            "answer": "Showing a sample of 50 of 292 matches in Paris.",
            "insights": "",
        })

    import core.llm as llm
    monkeypatch.setattr(llm, "chat_completion", _fake_chat)

    rows = [
        {"hotel_id": i, "name": f"H{i}", "city": "Paris", "avg_score": 8.0}
        for i in range(292)
    ]
    answer, _ = _run(narrate_mod.narrate_results(
        question="all hotels in Paris",
        rows=rows,
        path="parameterized",
        query_ran='{"city":"Paris"}',
    ))

    # Prompt must surface the true total count and the sample size.
    assert "292" in seen["prompt"]
    assert "sample of 50" in seen["prompt"]
    # The dump must not leak rows past the 50-row ceiling (e.g. H50..H291).
    assert "H0" in seen["prompt"]
    assert "H291" not in seen["prompt"]
    assert "H100" not in seen["prompt"]
    # Answer surfaces the total.
    assert "292" in answer
