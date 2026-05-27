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

from core.observability import observe as _observe


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

# JSON schema for the narration response (OpenAI-style; Gemini converts internally).
_NARRATION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer":   {"type": "string"},
        "insights": {"type": "string"},
    },
    "required": ["answer", "insights"],
}


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
    from core.llm import chat_completion  # deferred import keeps module importable without LLM

    n = len(rows)
    formatted = _format_rows(rows)

    empty_note = "No rows were returned for this query." if n == 0 else ""

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

    try:
        raw = await chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system=NARRATION_SYSTEM_PROMPT,
            model_tier="main",
            response_json=True,
            json_schema=_NARRATION_JSON_SCHEMA,
            temperature=0.2,
        )
        result = json.loads(raw)
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
