"""End-to-end pipeline tests for deictic city resolution.

These exercise the full ask() path including search_hotels against the real DB,
but mock the LLM (router + narration) to keep the test deterministic.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "hotels.db")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _patch_llm(monkeypatch, captured: dict):
    """Patch the router + narration LLM hooks.

    Router behaviour mirrors the deployed model: when a NAMED city appears
    literally in the question, it calls search_hotels(city=<that city>, ...);
    when only 'my city' appears (deictic, unresolved), it silently drops the
    constraint — which is the bug we're guarding against.
    """
    import re
    import core.llm as llm

    async def fake_function_call(messages, tools, system, model_tier):
        text = messages[-1]["content"]
        captured["router_question"] = text

        amenities = []
        if re.search(r"\bpool\b", text, re.IGNORECASE):
            amenities.append("has_pool")

        m = re.search(
            r"\b(Paris|London|Barcelona|Milan|Vienna|Amsterdam)\b",
            text,
            re.IGNORECASE,
        )
        if m:
            city = m.group(1).capitalize()
            args = {"city": city, "limit": 10}
            if amenities:
                args["required_amenities"] = amenities
            return "search_hotels", args

        # No named city found → bug-mirroring behaviour: drop the constraint.
        args = {"limit": 10}
        if amenities:
            args["required_amenities"] = amenities
        return "search_hotels", args

    async def fake_chat_completion(messages, system=None, model_tier="main",
                                   response_json=False, json_schema=None, temperature=0.0):
        captured["narrate_prompt"] = messages[-1]["content"]
        return json.dumps({"answer": "ok", "insights": ""})

    monkeypatch.setattr(llm, "function_call", fake_function_call)
    monkeypatch.setattr(llm, "chat_completion", fake_chat_completion)


def test_text_mode_my_city_returns_only_paris(monkeypatch):
    """The original failing case: 'hotels with pool in my city' + Paris → only Paris."""
    captured: dict = {}
    _patch_llm(monkeypatch, captured)

    from core.pipeline import ask
    result = _run(ask(
        "What are the hotels with pool in my city?",
        user_lat=48.8566, user_lng=2.3522, user_city="Paris",
        db_path=DB_PATH,
    ))

    assert result.declined is False
    assert len(result.hotels) > 0
    assert all(h.city == "Paris" for h in result.hotels), \
        f"Expected only Paris hotels, got cities: {sorted({h.city for h in result.hotels})}"
    assert all(h.has_pool == 1 for h in result.hotels)
    # The router saw a literally-named Paris in the question, not a bare deictic.
    assert "Paris" in captured["router_question"]
    assert "my city" not in captured["router_question"].lower()


def test_voice_style_question_with_named_city_also_works(monkeypatch):
    """Regression: voice-style transcript ('...in my city Paris?') still routes correctly."""
    captured: dict = {}
    _patch_llm(monkeypatch, captured)

    from core.pipeline import ask
    result = _run(ask(
        "What are the hotels with pool in my city Paris?",
        user_lat=None, user_lng=None, user_city=None,  # voice path: no dropdown
        db_path=DB_PATH,
    ))
    assert result.declined is False
    assert len(result.hotels) > 0
    assert all(h.city == "Paris" for h in result.hotels)


def test_text_mode_my_city_without_user_city_declines(monkeypatch):
    """Deictic phrase + no user_city → honest decline before the router LLM."""
    captured: dict = {}
    _patch_llm(monkeypatch, captured)

    from core.pipeline import ask
    result = _run(ask(
        "What are the hotels with pool in my city?",
        user_lat=None, user_lng=None, user_city=None,
        db_path=DB_PATH,
    ))
    assert result.declined is True
    assert result.path == "declined"
    assert "city" in (result.decline_reason or "").lower()
    # Critically: router LLM was NEVER called.
    assert "router_question" not in captured


def test_near_me_still_uses_geo_not_city_substitution(monkeypatch):
    """Proximity phrases must remain proximity; helper must not rewrite them."""
    captured: dict = {}
    _patch_llm(monkeypatch, captured)

    from core.pipeline import ask
    _run(ask(
        "hotels near me",
        user_lat=48.8566, user_lng=2.3522, user_city="Paris",
        db_path=DB_PATH,
    ))
    # The router question still says "near me" — no city substitution.
    assert "near me" in captured["router_question"]
    assert "in Paris" not in captured["router_question"]
