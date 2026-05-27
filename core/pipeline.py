"""
Phase 4.5 — Pipeline orchestrator.

ask() wires: router → path execution → narration → PipelineResponse.
This is the single entry-point for any question; FastAPI and tests call this.
"""

from __future__ import annotations

import logging

from core.db import get_connection
from core.fallback import FallbackError, execute_fallback
from core.models import HotelRow, PipelineResponse, QuerySpec, SearchParams
from core.narrate import narrate_results
from core.router import route_question
from core.search import search_hotels
from core.semantic import compile_query_spec, execute_semantic_query

logger = logging.getLogger(__name__)

# ── Langfuse @observe decorator (optional) ────────────────────────────────────
try:
    from langfuse import observe as _observe
except ImportError:
    def _observe(name: str | None = None, **_kw):  # type: ignore[misc]
        def _decorator(fn):
            return fn
        return _decorator


@_observe(name="pipeline_ask")
async def ask(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
    db_path: str = "data/hotels.db",
) -> PipelineResponse:
    """Full pipeline: route → execute → narrate.

    Always returns a PipelineResponse; never raises to the caller.
    Unexpected exceptions are caught and returned as a decline response.
    """
    conn = get_connection(db_path)
    try:
        # ── 1. Route ──────────────────────────────────────────────────────────
        path, path_input = await route_question(question, user_lat, user_lng)

        # ── 2a. Declined ─────────────────────────────────────────────────────
        if path == "declined":
            reason = str(path_input or "Cannot answer from available data.")
            return PipelineResponse(
                answer=f"I can't answer that: {reason}",
                insights="",
                hotels=[],
                path="declined",
                declined=True,
                decline_reason=reason,
            )

        # ── 2b. Execute the chosen path ───────────────────────────────────────
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

        elif path == "fallback":
            rows, query_ran = await execute_fallback(question, conn)

        else:
            # Unknown path — treat as decline
            return PipelineResponse(
                answer="I encountered an unexpected routing error.",
                insights="",
                hotels=[],
                path="declined",
                declined=True,
                decline_reason=f"Unknown path: {path}",
            )

        # ── 3. Narrate ────────────────────────────────────────────────────────
        answer, insights = await narrate_results(question, rows, path, query_ran)

        # ── 4. Build PipelineResponse ─────────────────────────────────────────
        # Convert raw dicts to HotelRow only for the parameterized path
        # (semantic/fallback rows may not have all HotelRow fields)
        if path == "parameterized":
            final_hotels = hotel_rows
        else:
            # Try to cast fallback/semantic rows to HotelRow; skip rows that
            # don't have the required fields (aggregation rows, etc.)
            final_hotels = []
            for r in rows:
                if "id" in r and "latitude" in r and "longitude" in r:
                    try:
                        final_hotels.append(HotelRow(**r))
                    except Exception:  # noqa: BLE001
                        pass

        return PipelineResponse(
            answer=answer,
            insights=insights,
            hotels=final_hotels,
            path=path,
            query_ran=query_ran,
        )

    except FallbackError as exc:
        logger.warning("FallbackError: %s", exc)
        return PipelineResponse(
            answer=f"I couldn't generate a valid query for that question: {exc}",
            insights="",
            hotels=[],
            path="fallback",
            declined=True,
            decline_reason=str(exc),
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pipeline error: %s", exc)
        return PipelineResponse(
            answer="An unexpected error occurred. Please try again.",
            insights="",
            hotels=[],
            path="declined",
            declined=True,
            decline_reason=str(exc),
        )

    finally:
        conn.close()
