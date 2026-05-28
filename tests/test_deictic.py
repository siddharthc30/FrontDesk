"""Unit tests for core.deictic — deterministic deictic resolution."""
from __future__ import annotations

import pytest

from core.deictic import resolve_deictic_city


# ── city deictics with user_city ─────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_substring", [
    ("hotels with pool in my city",      "in Paris"),
    ("What are the hotels with pool in my city?", "in Paris"),
    ("hotels in this city",              "in Paris"),
    ("best hotels here",                 "in Paris"),
    ("anything around here with a gym",  "in Paris"),
    ("show hotels in town",              "in Paris"),
    ("MY CITY top hotels",               "in Paris"),  # case-insensitive
])
def test_city_deictic_substituted_when_user_city_provided(question, expected_substring):
    res = resolve_deictic_city(question, user_city="Paris")
    assert res.rewritten is True
    assert res.decline_reason is None
    assert expected_substring in res.question
    # Original deictic phrase should be gone (case-insensitive).
    assert "my city" not in res.question.lower()
    assert "this city" not in res.question.lower()


def test_voice_style_question_unchanged_when_city_already_named():
    """Voice case: speaker named the city literally. Deictic is redundant — pass through."""
    q = "What are the hotels with pool in my city Paris?"
    res = resolve_deictic_city(q, user_city="Paris")
    assert res.rewritten is False
    assert res.decline_reason is None
    assert res.question == q


def test_named_city_wins_over_missing_user_city():
    """Even with no user_city, a literally-named city short-circuits the decline."""
    q = "hotels in my city Paris"
    res = resolve_deictic_city(q, user_city=None)
    assert res.rewritten is False
    assert res.decline_reason is None
    assert res.question == q


def test_no_deictic_phrase_returns_unchanged():
    q = "top 5 hotels in Barcelona"
    res = resolve_deictic_city(q, user_city="Paris")
    assert res.rewritten is False
    assert res.decline_reason is None
    assert res.question == q


# ── decline when user_city missing ───────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "hotels with pool in my city",
    "anything good here?",
    "hotels in this city",
    "around here",
])
def test_decline_when_deictic_but_no_user_city(question):
    res = resolve_deictic_city(question, user_city=None)
    assert res.rewritten is False
    assert res.decline_reason is not None
    assert "city" in res.decline_reason.lower()


def test_empty_user_city_treated_as_missing():
    res = resolve_deictic_city("hotels in my city", user_city="")
    assert res.decline_reason is not None


# ── proximity phrases left untouched ─────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "hotels near me",
    "hotels close to me",
    "hotels within 5 km",
    "hotels within 2.5 miles",
    "hotels within 500m",
    "any nearby hotels",
    "hotels around me",
    "walking distance hotels",
])
def test_proximity_phrases_not_rewritten(question):
    res = resolve_deictic_city(question, user_city="Paris")
    assert res.rewritten is False
    assert res.decline_reason is None
    assert res.question == question


def test_near_me_here_combo_is_proximity_not_city_scope():
    """'near me here' contains 'here' but the proximity phrase dominates."""
    q = "hotels near me here"
    res = resolve_deictic_city(q, user_city="Paris")
    assert res.rewritten is False
    assert res.question == q


# ── word boundary safety ─────────────────────────────────────────────────────

def test_no_false_match_on_substring():
    # "hereafter" contains "here" but should not match.
    q = "any hereafter plans for hotels in Rome"
    res = resolve_deictic_city(q, user_city="Paris")
    assert res.rewritten is False
    assert res.question == q


def test_empty_question_safe():
    res = resolve_deictic_city("", user_city="Paris")
    assert res.rewritten is False
    assert res.decline_reason is None
    assert res.question == ""
