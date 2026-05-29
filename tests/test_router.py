"""Router tests for deictic-city resolution and honest decline."""
from __future__ import annotations

import asyncio

import pytest

from core import router as router_mod
from core.models import SearchParams


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _Recorder:
    """Records the system prompt + user message handed to the LLM."""

    def __init__(self, fn_name: str, fn_args: dict):
        self.fn_name = fn_name
        self.fn_args = fn_args
        self.seen_system: str | None = None
        self.seen_user: str | None = None

    async def __call__(self, messages, tools, system, model_tier):
        self.seen_system = system
        self.seen_user = messages[-1]["content"]
        return self.fn_name, self.fn_args


def _patch_llm(monkeypatch, recorder: _Recorder):
    """Patch core.llm.function_call (the deferred import inside route_question)."""
    import core.llm as llm
    monkeypatch.setattr(llm, "function_call", recorder)


def test_my_city_resolves_when_user_city_provided(monkeypatch):
    rec = _Recorder(
        "search_hotels",
        {"city": "Paris", "required_amenities": ["has_pool"]},
    )
    _patch_llm(monkeypatch, rec)

    path, params = _run(router_mod.route_question(
        "hotels with pool in my city",
        user_lat=48.8566, user_lng=2.3522, user_city="Paris",
    ))

    assert path == "parameterized"
    assert isinstance(params, SearchParams)
    assert params.city == "Paris"
    assert params.radius_km is None
    # Confirm the context line was forwarded verbatim to the LLM.
    assert 'city="Paris"' in rec.seen_user


def test_my_city_declines_when_user_city_missing(monkeypatch):
    rec = _Recorder(
        "decline",
        {"reason": "Please select a city to use 'my city'."},
    )
    _patch_llm(monkeypatch, rec)

    path, reason = _run(router_mod.route_question(
        "hotels with pool in my city",
        user_lat=None, user_lng=None, user_city=None,
    ))

    assert path == "declined"
    assert isinstance(reason, str)
    assert "city" in reason.lower()
    # No user context line should be appended when nothing is provided.
    assert "[User context:" not in rec.seen_user


def test_near_me_uses_geo_not_city(monkeypatch):
    rec = _Recorder(
        "search_hotels",
        {"user_lat": 48.8566, "user_lng": 2.3522, "radius_km": 5.0},
    )
    _patch_llm(monkeypatch, rec)

    path, params = _run(router_mod.route_question(
        "hotels near me",
        user_lat=48.8566, user_lng=2.3522, user_city="Paris",
    ))

    assert path == "parameterized"
    assert isinstance(params, SearchParams)
    assert params.city is None
    assert params.radius_km == 5.0
    assert params.user_lat == 48.8566
    # Context line must include both city and geo so the model can choose.
    assert 'city="Paris"' in rec.seen_user
    assert "lat=48.8566" in rec.seen_user


def test_all_hotels_yields_no_limit(monkeypatch):
    """'all hotels in Paris' must produce limit=None — not a reflexive 10."""
    rec = _Recorder("search_hotels", {"city": "Paris"})
    _patch_llm(monkeypatch, rec)

    path, params = _run(router_mod.route_question("all hotels in Paris"))

    assert path == "parameterized"
    assert isinstance(params, SearchParams)
    assert params.city == "Paris"
    assert params.limit is None


def test_top_n_yields_explicit_limit(monkeypatch):
    """'top 5 hotels in Paris' must produce limit=5."""
    rec = _Recorder("search_hotels", {"city": "Paris", "limit": 5})
    _patch_llm(monkeypatch, rec)

    path, params = _run(router_mod.route_question("top 5 hotels in Paris"))

    assert path == "parameterized"
    assert isinstance(params, SearchParams)
    assert params.city == "Paris"
    assert params.limit == 5


def test_router_prompt_contains_limit_rule():
    prompt = router_mod.ROUTER_SYSTEM_PROMPT
    assert "LIMIT RULE" in prompt
    assert "all hotels in Paris" in prompt
    assert "top 5 hotels in Paris" in prompt


def test_router_prompt_contains_deictic_resolution_rules():
    """Smoke-check the prompt actually carries the new instructions."""
    prompt = router_mod.ROUTER_SYSTEM_PROMPT
    assert "my city" in prompt
    assert "near me" in prompt
    assert "decline" in prompt.lower()
    assert "User context" in prompt
