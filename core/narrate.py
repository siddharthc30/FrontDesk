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

# Typical aspect sentiment ranges — used to calibrate framing so "0.85 staff" isn't
# called exceptional when the average is 0.82.
ASPECT_TYPICAL_RANGES = {
    "location":     0.90,
    "staff":        0.82,
    "cleanliness":  0.78,
    "breakfast":    0.58,
    "room_comfort": 0.56,
    "noise":        0.51,
    "value":        0.37,
}

_CONFIDENCE_LABELS = {
    "parameterized": "high",
    "semantic":      "high",
    "review_search": "medium",
    "fallback":      "guarded",
    "declined":      "guarded",
}


def path_to_confidence(path: str) -> str:
    return _CONFIDENCE_LABELS.get(path, "high")


def _format_rows(rows: list[dict]) -> str:
    """Produce a compact text representation of query rows for the prompt."""
    if not rows:
        return "(no rows returned)"

    if len(rows) == 1 and list(rows[0].keys()) == ["value"]:
        return f"Result: {rows[0]['value']}"

    display = rows[:15]
    suffix = f"\n... and {len(rows) - 15} more rows" if len(rows) > 15 else ""

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


def _build_context_note(path: str, query_ran: str | None) -> str:
    """Build path-specific context note for the narration prompt."""
    if path == "review_search":
        terms = ""
        if query_ran and "terms=" in query_ran:
            terms = query_ran.split("terms=")[-1]
        return (
            f"NOTE: These results come from searching guest review text for {terms}. "
            "This is an ad-hoc search — sentiment values are computed from this search, "
            "not from precomputed scores. Mention counts indicate how many guests wrote "
            "about this topic. Do NOT present these as authoritative scores. "
            "Use phrasing like 'Based on searching guest reviews for ...' "
            "and cite mention counts (e.g. 'N guests mentioned this')."
        )
    if path == "fallback":
        return (
            "NOTE: This answer comes from a generated SQL query. "
            "It may be less reliable than structured queries. "
            "If results seem unexpected, acknowledge uncertainty."
        )
    if path in ("parameterized", "semantic"):
        aspect_note = ""
        aspect_ranges_text = ", ".join(
            f"{a}≈{v:.0%}" for a, v in ASPECT_TYPICAL_RANGES.items()
        )
        aspect_note = (
            f"NOTE: Typical aspect sentiment benchmarks: {aspect_ranges_text}. "
            "Calibrate your language accordingly — don't call 0.85 staff sentiment "
            "'exceptional' when the typical range is 0.82."
        )
        return aspect_note
    return ""


NARRATION_SYSTEM_PROMPT = (
    "You are a helpful assistant that summarises hotel search results. "
    "Answer the user's question using ONLY the data provided below. "
    "Do not invent, guess, or add any information not present in the results. "
    "Be concise and specific."
)

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
    context_note = _build_context_note(path, query_ran)

    empty_note = "No rows were returned for this query." if n == 0 else ""

    prompt = (
        f"USER QUESTION: {question}\n\n"
        f"QUERY RESULTS ({n} rows):\n{formatted}\n"
        + (f"\nNOTE: {empty_note}\n" if empty_note else "")
        + (f"\n{context_note}\n" if context_note else "")
        + "\nRespond in JSON with exactly two fields:\n"
        '- "answer": A direct, concise answer to the user\'s question (1–3 sentences). '
        "Synthesize and interpret — don't list hotel names and scores, since the user "
        "already sees the full results table below your answer. Focus on what the results "
        "MEAN for their question.\n"
        '- "insights": 1–3 observations that go BEYOND what\'s obvious from scanning the '
        "results table. Good insights include: patterns or trends across results "
        "(e.g. 'all three are clustered in the Gothic Quarter — this neighbourhood dominates "
        "for high-rated stays'); standout review highlights when review data is present "
        "(e.g. 'guests specifically praise the rooftop pool views and natural light'); "
        "trade-offs worth noting (e.g. 'Hotel X is closest but also the priciest — Hotel Y "
        "offers similar amenities for £20 less per night'); or honest caveats about what the "
        "data can or can't confirm. "
        "Do NOT restate name, score, price, distance, or amenity values that are already "
        "visible in the results table — the user can read those themselves. Never just echo "
        "column values. If there are no results, set insights to an empty string.\n\n"
        "If review text or review summaries are included in the results, prioritise weaving "
        "specific guest feedback into both the answer and insights. Review content is the "
        "most valuable thing you can surface because the user cannot easily scan hundreds of "
        "review snippets the way they can scan a table of scores and prices.\n\n"
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
        if n == 0:
            return "No hotels matched your criteria.", ""
        return f"Found {n} result(s) for your query.", ""
