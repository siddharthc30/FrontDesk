"""
Phase 4 — Grounded narration.

narrate_results() turns raw query rows into a human-readable answer and
1–3 insight sentences.  The model is fed ONLY the result rows — never the
full database, schema, or raw question-to-SQL mapping.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# ── Langfuse @observe decorator (optional) ────────────────────────────────────
try:
    from langfuse import observe as _observe
except ImportError:  # Langfuse not installed or not configured
    def _observe(name: str | None = None, **_kw):  # type: ignore[misc]
        """No-op decorator when Langfuse is unavailable."""
        def _decorator(fn):
            return fn
        return _decorator


def _format_rows(rows: list[dict]) -> str:
    """Produce a compact text representation of query rows for the prompt."""
    if not rows:
        return "(no rows returned)"

    # For aggregation results that have a single value key
    if len(rows) == 1 and list(rows[0].keys()) == ["value"]:
        return f"Result: {rows[0]['value']}"

    # Truncate to first 15 rows when there are many
    display = rows[:15]
    suffix = f"\n... and {len(rows) - 15} more rows" if len(rows) > 15 else ""

    # Build a plain text table
    if not display:
        return "(no rows)"

    keys = list(display[0].keys())
    header = " | ".join(str(k) for k in keys)
    separator = "-" * len(header)
    lines = [header, separator]
    for row in display:
        lines.append(" | ".join(
            "—" if v is None else str(v)
            for v in (row[k] for k in keys)
        ))
    return "\n".join(lines) + suffix


NARRATION_SYSTEM_PROMPT = (
    "You are a helpful assistant that summarises hotel search results. "
    "Answer the user's question using ONLY the data provided below. "
    "Do not invent, guess, or add any information not present in the results. "
    "Be concise and specific."
)


@_observe(name="narrate_results")
async def narrate_results(
    question: str,
    rows: list[dict],
    path: str,
    query_ran: str | None,
) -> tuple[str, str]:
    """Generate a grounded answer and insight sentences from query rows.

    Returns (answer, insights).
    Falls back to a plain-text answer if JSON parsing fails.
    """
    from google.genai import types

    from core.llm import get_client, get_model

    n = len(rows)
    formatted = _format_rows(rows)

    if n == 0:
        # Honest empty-result handling — still call the model for a polished message
        empty_note = "No rows were returned for this query."
    else:
        empty_note = ""

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"QUERY RESULTS ({n} rows):\n{formatted}\n"
        + (f"\nNOTE: {empty_note}\n" if empty_note else "")
        + "\nRespond in JSON with exactly two fields:\n"
        '- "answer": A direct, concise answer to the user\'s question (1–3 sentences).\n'
        '- "insights": 1–3 additional observations from the data that might interest '
        "the user. Each insight must reference specific values from the results. "
        "If there are no results, set insights to an empty string.\n\n"
        "If no results were returned, say so honestly. Do not make up hotels or scores."
    )

    client = get_client()
    model = get_model()

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=NARRATION_SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type="OBJECT",
                    properties={
                        "answer":   types.Schema(type="STRING"),
                        "insights": types.Schema(type="STRING"),
                    },
                    required=["answer", "insights"],
                ),
            ),
        )
        result = json.loads(response.text)
        answer   = str(result.get("answer", "")).strip()
        insights = str(result.get("insights", "")).strip()
        if not answer:
            answer = f"Found {n} result(s) for your query."
        return answer, insights

    except Exception as exc:  # noqa: BLE001
        logger.warning("Narration JSON parse/call failed: %s", exc)
        # Graceful fallback — never crash
        if n == 0:
            return "No hotels matched your criteria.", ""
        return f"Found {n} result(s) for your query.", ""
