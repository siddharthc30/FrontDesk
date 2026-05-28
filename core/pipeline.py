"""
Phase 4.5 / 5 — Pipeline orchestrator.

ask()        → non-streaming, returns PipelineResponse (used by test_interactive.py & tests)
ask_stream() → async generator yielding (event_type, data) tuples for SSE (used by FastAPI)

Both call the same internal helpers; no business logic is duplicated.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from core.db import get_connection
from core.deictic import resolve_deictic_city
from core.fallback import FallbackError, execute_fallback
from core.models import HotelRow, PipelineResponse, QuerySpec, SearchParams
from core.narrate import narrate_results, path_to_confidence
from core.observability import observe as _observe
from core.review_search import search_reviews_conn
from core.router import ReviewSearchParams, route_question
from core.search import search_hotels
from core.semantic import compile_query_spec, execute_semantic_query

logger = logging.getLogger(__name__)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_final_hotels(path: str, hotel_rows: list[HotelRow], rows: list[dict]) -> list[HotelRow]:
    """Return HotelRow list from whichever representation we have."""
    if path == "parameterized":
        return hotel_rows
    # fallback / semantic: try to cast rows that have the required fields
    final: list[HotelRow] = []
    for r in rows:
        if "hotel_id" in r and "latitude" in r and "longitude" in r:
            try:
                final.append(HotelRow(**r))
            except Exception:  # noqa: BLE001
                pass
    return final


def _path_label(path: str) -> str:
    labels = {
        "parameterized": "parameterized (filter/sort)",
        "semantic":      "semantic (aggregate/count)",
        "review_search": "review_search (guest review text)",
        "fallback":      "fallback (SQL generation)",
    }
    return labels.get(path, path)


# ── Streaming pipeline ─────────────────────────────────────────────────────────

@_observe(name="pipeline_ask")
async def ask_stream(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
    user_city: str | None = None,
    db_path: str = "data/hotels.db",
) -> AsyncIterator[tuple[str, Any]]:
    """Async generator yielding (event_type, data) tuples.

    event_type values:
      "step"   → data is a dict {"step": str, "message": str}
      "result" → data is a PipelineResponse
      "error"  → data is a dict {"message": str}  (only on unexpected exceptions)
    """
    conn = get_connection(db_path)
    try:
        # ── 1. Resolve deictic city references deterministically ─────────────
        # ("my city", "this city", "here", ...). Mirrors the working voice-mode
        # case (where the speaker names the city literally) by substituting the
        # selected city into the text before the router LLM sees it. Done in
        # code, not in the prompt, because LLM-side resolution is unreliable.
        resolution = resolve_deictic_city(question, user_city)
        if resolution.decline_reason is not None:
            yield ("step", {"step": "declined", "message": f"❌ Cannot answer: {resolution.decline_reason}"})
            yield ("result", PipelineResponse(
                answer=f"I can't answer that: {resolution.decline_reason}",
                insights="",
                hotels=[],
                path="declined",
                declined=True,
                decline_reason=resolution.decline_reason,
            ))
            return
        effective_question = resolution.question
        if resolution.rewritten:
            yield ("step", {
                "step": "resolving",
                "message": f"📍 Resolved 'my city' → **{user_city}**",
            })

        # ── 2. Route ──────────────────────────────────────────────────────────
        yield ("step", {"step": "routing", "message": "🔍 Analyzing your question..."})

        path, path_input = await route_question(effective_question, user_lat, user_lng, user_city)

        if path == "declined":
            reason = str(path_input or "Cannot answer from available data.")
            yield ("step", {"step": "declined", "message": f"❌ Cannot answer: {reason}"})
            yield ("result", PipelineResponse(
                answer=f"I can't answer that: {reason}",
                insights="",
                hotels=[],
                path="declined",
                declined=True,
                decline_reason=reason,
            ))
            return

        yield ("step", {"step": "routing", "message": f"📋 Matched to **{_path_label(path)}** path"})

        # ── 2. Execute ────────────────────────────────────────────────────────
        if path == "fallback":
            yield ("step", {"step": "executing", "message": "⚡ Generating and validating SQL..."})
        else:
            yield ("step", {"step": "executing", "message": f"⚡ Running query via {path} path..."})

        rows: list[dict] = []
        query_ran: str | None = None
        hotel_rows: list[HotelRow] = []

        if path == "parameterized":
            assert isinstance(path_input, SearchParams)
            hotel_rows = search_hotels(path_input, conn)
            rows = [h.model_dump() for h in hotel_rows]
            query_ran = path_input.model_dump_json(exclude_none=True)

        elif path == "semantic":
            assert isinstance(path_input, QuerySpec)
            sql, _params = compile_query_spec(path_input)
            rows = execute_semantic_query(path_input, conn)
            query_ran = sql

        elif path == "review_search":
            assert isinstance(path_input, ReviewSearchParams)
            rows = search_reviews_conn(
                path_input.search_terms,
                conn,
                city=path_input.city,
                min_rating=path_input.min_rating,
                limit=path_input.limit,
            )
            query_ran = f"review_search terms={path_input.search_terms}"

        elif path == "fallback":
            rows, query_ran = await execute_fallback(effective_question, conn)

        else:
            yield ("result", PipelineResponse(
                answer="I encountered an unexpected routing error.",
                insights="",
                hotels=[],
                path="declined",
                declined=True,
                decline_reason=f"Unknown path: {path}",
            ))
            return

        # ── 3. Narrate ────────────────────────────────────────────────────────
        n = len(rows)
        yield ("step", {"step": "narrating", "message": f"✍️ Generating answer from {n} result(s)..."})

        answer, insights = await narrate_results(effective_question, rows, path, query_ran)

        # ── 4. Yield final result ─────────────────────────────────────────────
        final_hotels = _build_final_hotels(path, hotel_rows, rows)
        yield ("result", PipelineResponse(
            answer=answer,
            insights=insights,
            hotels=final_hotels,
            path=path,
            confidence=path_to_confidence(path),
            query_ran=query_ran,
        ))

    except FallbackError as exc:
        logger.warning("FallbackError: %s", exc)
        yield ("result", PipelineResponse(
            answer="I couldn't generate a valid query for that question. Try rephrasing it.",
            insights="",
            hotels=[],
            path="fallback",
            confidence="guarded",
            declined=True,
            decline_reason="The generated query could not be validated against the database schema.",
        ))

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline error: %s", exc)
        yield ("error", {"message": "An unexpected error occurred. Please try again."})

    finally:
        conn.close()


# ── Non-streaming pipeline (test_interactive.py, unit tests) ──────────────────
# Note: the @observe(name="pipeline_ask") trace root lives on ask_stream() so it
# covers both the streaming and non-streaming entry points.

async def ask(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
    user_city: str | None = None,
    db_path: str = "data/hotels.db",
) -> PipelineResponse:
    """Full pipeline: route → execute → narrate.

    Always returns a PipelineResponse; never raises to the caller.
    Collects all events from ask_stream() and returns the final result.
    """
    last_result: PipelineResponse | None = None

    try:
        async for event_type, data in ask_stream(question, user_lat, user_lng, user_city, db_path):
            if event_type == "result":
                last_result = data
            elif event_type == "error":
                return PipelineResponse(
                    answer="An unexpected error occurred. Please try again.",
                    insights="",
                    hotels=[],
                    path="declined",
                    declined=True,
                    decline_reason="Internal error.",
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline error in ask(): %s", exc)
        return PipelineResponse(
            answer="An unexpected error occurred. Please try again.",
            insights="",
            hotels=[],
            path="declined",
            declined=True,
            decline_reason="Internal error.",
        )

    if last_result is None:
        return PipelineResponse(
            answer="No result was produced.",
            insights="",
            hotels=[],
            path="declined",
            declined=True,
            decline_reason="Pipeline produced no result event.",
        )

    return last_result
