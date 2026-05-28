"""Deterministic and LLM-based scoring functions for the hotel NL search pipeline.

All deterministic evaluators are pure Python — no LLM calls, no network.
score_narration_faithfulness is the only async function and makes one LLM call.

Scoring functions return langfuse.Evaluation objects so they can be attached
directly to Langfuse traces via run_experiment evaluators.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langfuse import Evaluation

from core.models import HotelRow, PipelineResponse


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first attribute/key that exists on obj."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def _hotel_city(hotel: Any) -> Optional[str]:
    v = _get_field(hotel, "city")
    return v.strip() if isinstance(v, str) else None


def _hotel_score(hotel: Any) -> Optional[float]:
    # HotelRow uses avg_score; guard against any legacy 'rating' key
    v = _get_field(hotel, "avg_score", "rating")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _hotel_has_amenity(hotel: Any, amenity: str) -> bool:
    key = f"has_{amenity}"
    v = _get_field(hotel, key, default=0)
    return bool(v)


# ── Deterministic evaluators ──────────────────────────────────────────────────

def score_routing(result: PipelineResponse, expected_path: str) -> Evaluation:
    """BOOLEAN: does result.path match expected_path?"""
    return Evaluation(
        name="routing_correct",
        value=result.path == expected_path,
        data_type="BOOLEAN",
        comment=f"expected={expected_path}, got={result.path}",
    )


def score_result_grounding(
    result: PipelineResponse,
    assertions: dict,
) -> list[Evaluation]:
    """BOOLEAN scores for each assertion key present in the assertions dict.

    Supported keys:
      min_hotels, max_hotels, exact_hotels,
      all_city, all_min_score, all_has_amenity,
      answer_contains, answer_not_contains,
      should_decline
    """
    scores: list[Evaluation] = []
    hotels = result.hotels or []
    n = len(hotels)
    answer_lower = (result.answer or "").lower()

    if "min_hotels" in assertions:
        target = assertions["min_hotels"]
        scores.append(Evaluation(
            name="grounding_min_hotels",
            value=n >= target,
            data_type="BOOLEAN",
            comment=f"expected >= {target}, got {n}",
        ))

    if "max_hotels" in assertions:
        target = assertions["max_hotels"]
        scores.append(Evaluation(
            name="grounding_max_hotels",
            value=n <= target,
            data_type="BOOLEAN",
            comment=f"expected <= {target}, got {n}",
        ))

    if "exact_hotels" in assertions:
        target = assertions["exact_hotels"]
        scores.append(Evaluation(
            name="grounding_exact_hotels",
            value=n == target,
            data_type="BOOLEAN",
            comment=f"expected == {target}, got {n}",
        ))

    if "all_city" in assertions:
        expected_city = assertions["all_city"].lower()
        failures = [
            _hotel_city(h) for h in hotels
            if (_hotel_city(h) or "").lower() != expected_city
        ]
        passed = len(failures) == 0
        scores.append(Evaluation(
            name="grounding_all_city",
            value=passed,
            data_type="BOOLEAN",
            comment=(
                "all hotels match city" if passed
                else f"wrong cities: {failures[:3]}"
            ),
        ))

    if "all_min_score" in assertions:
        min_score = float(assertions["all_min_score"])
        failing = []
        for h in hotels:
            s = _hotel_score(h)
            if s is None or s < min_score:
                failing.append(_get_field(h, "name", default="?"))
        passed = len(failing) == 0
        scores.append(Evaluation(
            name="grounding_all_min_score",
            value=passed,
            data_type="BOOLEAN",
            comment=(
                f"all scores >= {min_score}" if passed
                else f"failing hotels: {failing[:3]}"
            ),
        ))

    if "all_has_amenity" in assertions:
        amenity = assertions["all_has_amenity"]
        failing = [
            _get_field(h, "name", default="?")
            for h in hotels
            if not _hotel_has_amenity(h, amenity)
        ]
        passed = len(failing) == 0
        scores.append(Evaluation(
            name="grounding_all_has_amenity",
            value=passed,
            data_type="BOOLEAN",
            comment=(
                f"all hotels have {amenity}" if passed
                else f"missing amenity: {failing[:3]}"
            ),
        ))

    if "answer_contains" in assertions:
        substrings = assertions["answer_contains"]
        missing = [s for s in substrings if s.lower() not in answer_lower]
        passed = len(missing) == 0
        scores.append(Evaluation(
            name="grounding_answer_contains",
            value=passed,
            data_type="BOOLEAN",
            comment=(
                "all substrings found" if passed
                else f"missing: {missing}"
            ),
        ))

    if "answer_not_contains" in assertions:
        substrings = assertions["answer_not_contains"]
        found = [s for s in substrings if s.lower() in answer_lower]
        passed = len(found) == 0
        scores.append(Evaluation(
            name="grounding_answer_not_contains",
            value=passed,
            data_type="BOOLEAN",
            comment=(
                "no forbidden substrings found" if passed
                else f"found forbidden: {found}"
            ),
        ))

    if "should_decline" in assertions:
        expected_decline = bool(assertions["should_decline"])
        passed = result.declined == expected_decline
        scores.append(Evaluation(
            name="grounding_should_decline",
            value=passed,
            data_type="BOOLEAN",
            comment=f"expected declined={expected_decline}, got {result.declined}",
        ))

    return scores


def score_empty_result_honesty(result: PipelineResponse) -> Optional[Evaluation]:
    """BOOLEAN: when hotels is empty and not declined, does the answer admit it?

    Returns None when the check doesn't apply (hotels non-empty, declined, or a
    semantic/review_search path where empty hotels is normal — aggregate rows and
    review rows never become HotelRow objects in the pipeline).
    """
    if result.declined or result.hotels or result.path in ("semantic", "review_search"):
        return None

    honest_phrases = [
        "no hotel", "no result", "no match", "couldn't find", "could not find",
        "none", "0 hotel", "didn't find", "did not find", "not found",
        "no data", "nothing",
    ]
    answer_lower = (result.answer or "").lower()
    is_honest = any(p in answer_lower for p in honest_phrases)

    return Evaluation(
        name="empty_result_honest",
        value=is_honest,
        data_type="BOOLEAN",
        comment=(
            "answer acknowledges empty results" if is_honest
            else f"answer may be inventing hotels (no honest phrase found)"
        ),
    )


def score_transparency(result: PipelineResponse) -> Evaluation:
    """BOOLEAN: non-declined responses must have a non-empty query_ran field."""
    if result.declined:
        return Evaluation(
            name="has_query_ran",
            value=True,
            data_type="BOOLEAN",
            comment="declined — query_ran not required",
        )
    has_query = bool(result.query_ran and result.query_ran.strip())
    return Evaluation(
        name="has_query_ran",
        value=has_query,
        data_type="BOOLEAN",
        comment=(
            f"query_ran present ({len(result.query_ran or '')} chars)" if has_query
            else "query_ran is empty"
        ),
    )


# ── LLM-as-judge evaluator ────────────────────────────────────────────────────

_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Your task is to determine whether an AI assistant's \
answer is FAITHFUL to the data it was given.

## The user asked:
{question}

## The system returned these rows from the database:
{rows_as_json}

## The assistant's answer:
{answer}

## The assistant's insights:
{insights}

## Your task:
Score the faithfulness of the answer and insights on a scale from 0.0 to 1.0:
- 1.0: Every claim in the answer and insights is directly supported by the data rows.
- 0.5: Some claims are supported, but some are embellished or slightly inaccurate.
- 0.0: The answer contains fabricated hotels, invented statistics, or claims with \
no basis in the returned data.

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}\
"""


async def score_narration_faithfulness(
    result: PipelineResponse,
    question: str,
) -> Evaluation:
    """NUMERIC (0.0–1.0): does the narration stay grounded in the returned rows?

    Uses the same LLM abstraction as the pipeline (core/llm.py).
    Skip calling this for declined results or empty-results-with-honest-answer.
    """
    from core.llm import chat_completion  # deferred import to avoid circular deps

    rows_json = json.dumps(
        [h.model_dump(exclude_none=True) if isinstance(h, HotelRow) else h
         for h in (result.hotels or [])],
        indent=2,
    )

    prompt = _FAITHFULNESS_PROMPT.format(
        question=question,
        rows_as_json=rows_json or "[]",
        answer=result.answer or "",
        insights=result.insights or "",
    )

    try:
        raw = await chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model_tier="router",   # cheap/fast tier is sufficient for judging
            response_json=True,
            temperature=0.0,
        )
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]
        reason = str(parsed.get("reason", ""))
    except json.JSONDecodeError:
        score = 0.5
        reason = f"JSON parse failed; raw response: {str(raw)[:200]}"
    except Exception as exc:  # noqa: BLE001
        score = 0.5
        reason = f"Judge call failed: {exc}"

    return Evaluation(
        name="narration_faithful",
        value=score,
        data_type="NUMERIC",
        comment=reason,
    )
