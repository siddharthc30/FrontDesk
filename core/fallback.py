"""
Guarded text-to-SQL fallback path.
LLM writes SQL only here. sqlglot validates before execution.
"""

from __future__ import annotations

import re
import sqlite3

import sqlglot
import sqlglot.expressions as sqlglot_exp

from core.db import get_data_dictionary

# ── constants ────────────────────────────────────────────────────────────────

ALLOWED_TABLES: frozenset[str] = frozenset({
    "hotels", "reviews", "hotel_aspect_sentiment",
    "pragma_table_info",  # SQLite table-valued function for schema introspection
})

ALLOWED_COLUMNS: frozenset[str] = frozenset({
    # hotels table
    "hotel_id", "name", "address", "city", "country",
    "latitude", "longitude", "avg_score", "total_reviews",
    "price_per_night",
    "has_wifi", "has_pool", "has_gym", "has_sauna",
    "has_restaurant", "has_room_service", "has_lounge", "has_event_space",
    # reviews table
    "review_id", "review_date", "reviewer_nationality",
    "positive_review", "positive_word_count",
    "negative_review", "negative_word_count",
    "reviewer_score", "tags", "days_since_review",
    # hotel_aspect_sentiment table
    "aspect", "sentiment", "mention_count", "pos_count", "neg_count", "is_facility",
    # computed aliases used in SELECT/ORDER BY expressions
    "distance_km", "value", "count", "n", "avg_sentiment",
    "avg_score", "avg_price", "avg_reviews", "total",
    "hotel_avg_score", "reviewer_avg_score",
    # pragma_table_info output columns
    "cid", "type", "notnull", "dflt_value", "pk",
})

# Hardcoded few-shot examples — NOT generated; teach the model table/column names.
FEW_SHOT_EXAMPLES = """
Examples of correct SQL for this database:

Q: "Which hotel has the highest review score?"
SQL: SELECT * FROM hotels ORDER BY avg_score DESC LIMIT 1

Q: "How many hotels are in Paris?"
SQL: SELECT COUNT(*) AS count FROM hotels WHERE LOWER(city) = LOWER('Paris')

Q: "Show me hotels within 10km of latitude 51.5, longitude -0.12"
SQL: SELECT *, haversine(latitude, longitude, 51.5, -0.12) AS distance_km FROM hotels WHERE haversine(latitude, longitude, 51.5, -0.12) <= 10 ORDER BY distance_km ASC

Q: "What is the average score of hotels in each country?"
SQL: SELECT country, ROUND(AVG(avg_score), 2) AS avg_score FROM hotels GROUP BY country ORDER BY avg_score DESC

Q: "List hotels whose name contains 'Palace'"
SQL: SELECT * FROM hotels WHERE name LIKE '%Palace%' ORDER BY avg_score DESC

Q: "What is the cheapest hotel in Barcelona?"
SQL: SELECT * FROM hotels WHERE LOWER(city) = LOWER('Barcelona') ORDER BY price_per_night ASC LIMIT 1

Q: "Show hotels with a pool that cost less than 200 per night"
SQL: SELECT * FROM hotels WHERE has_pool = 1 AND price_per_night < 200 ORDER BY avg_score DESC

Q: "What is the average nightly price by city?"
SQL: SELECT city, ROUND(AVG(price_per_night), 0) AS avg_price FROM hotels GROUP BY city ORDER BY avg_price ASC

Q: "Which hotels have both a gym and a sauna?"
SQL: SELECT * FROM hotels WHERE has_gym = 1 AND has_sauna = 1 ORDER BY avg_score DESC

Q: "Which hotels have the most reviews?"
SQL: SELECT hotel_id, name, city, total_reviews FROM hotels ORDER BY total_reviews DESC LIMIT 10

Q: "What is the average reviewer score vs hotel avg_score by city?"
SQL: SELECT h.city, ROUND(AVG(h.avg_score), 2) AS hotel_avg_score, ROUND(AVG(r.reviewer_score), 2) AS reviewer_avg_score FROM hotels h JOIN reviews r ON r.hotel_id = h.hotel_id GROUP BY h.city ORDER BY hotel_avg_score DESC

Q: "Which aspect has the highest average sentiment?"
SQL: SELECT aspect, ROUND(AVG(sentiment), 3) AS avg_sentiment FROM hotel_aspect_sentiment GROUP BY aspect ORDER BY avg_sentiment DESC

Q: "How many columns are there in the hotel table?"
SQL: SELECT COUNT(*) AS count FROM pragma_table_info('hotels')

Q: "What columns does the hotels table have?"
SQL: SELECT name FROM pragma_table_info('hotels')
""".strip()

SYSTEM_INSTRUCTION = (
    "You are a SQL expert. Write a single SELECT statement against the hotel database. "
    "Available tables: hotels, reviews, hotel_aspect_sentiment. "
    "Use only the columns listed in the schema below. "
    "Join hotels and reviews on hotel_id. Join hotels and hotel_aspect_sentiment on hotel_id. "
    "For questions about the database schema or table structure, use pragma_table_info('hotels') "
    "— e.g. SELECT name, type FROM pragma_table_info('hotels'). "
    "Do not use INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, or PRAGMA. "
    "Return ONLY the SQL — no markdown fences, no explanation."
)


def _strip_fences(text: str) -> str:
    """Remove ```sql ... ``` or ``` ... ``` markdown fences."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_generated_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL with sqlglot. Returns (is_valid, error_reason)."""
    try:
        parsed = sqlglot.parse(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return False, f"Parse error: {e}"

    if len(parsed) != 1:
        return False, "Must be exactly one SQL statement"

    stmt = parsed[0]

    if not isinstance(stmt, sqlglot_exp.Select):
        return False, "Only SELECT statements are allowed"

    BLOCKED_TYPES = (
        sqlglot_exp.Insert, sqlglot_exp.Update, sqlglot_exp.Delete,
        sqlglot_exp.Drop, sqlglot_exp.Create, sqlglot_exp.Alter,
    )
    for node in stmt.walk():
        if isinstance(node, BLOCKED_TYPES):
            return False, f"Blocked statement type: {type(node).__name__}"

    for table in stmt.find_all(sqlglot_exp.Table):
        # Table-valued functions (e.g. pragma_table_info('hotels')) parse as
        # Table(this=Anonymous(...)) — extract the function name in that case.
        if isinstance(table.this, sqlglot_exp.Anonymous):
            tbl_name = table.this.name.lower()
        else:
            tbl_name = table.name.lower()
        if tbl_name not in ALLOWED_TABLES:
            return False, f"Unknown table: {tbl_name}"

    for col in stmt.find_all(sqlglot_exp.Column):
        if col.name.lower() not in ALLOWED_COLUMNS:
            return False, f"Unknown column: {col.name}"

    return True, ""


def _build_prompt(question: str, error_context: str | None = None) -> str:
    data_dict = get_data_dictionary()
    parts = [SYSTEM_INSTRUCTION, "", data_dict, "", FEW_SHOT_EXAMPLES, ""]
    if error_context:
        parts.append(error_context)
        parts.append("")
    parts.append(f'Question: "{question}"')
    parts.append("SQL:")
    return "\n".join(parts)


from core.observability import observe as _observe


class FallbackError(Exception):
    pass


@_observe(name="execute_fallback")
async def execute_fallback(
    question: str,
    conn: sqlite3.Connection,
) -> tuple[list[dict], str]:
    """Generate SQL via LLM, validate with sqlglot, execute on read-only connection.
    Returns (rows, sql_that_ran). Raises FallbackError on unrecoverable failure."""
    from core.llm import call_llm  # imported here to keep core/fallback importable without LLM setup

    # ── first attempt ─────────────────────────────────────────────────────────
    prompt = _build_prompt(question)
    raw = await call_llm(prompt, temperature=0.0)
    sql = _strip_fences(raw)

    valid, reason = validate_generated_sql(sql)
    if not valid:
        raise FallbackError(f"Generated SQL failed validation: {reason}\nSQL: {sql}")

    try:
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows], sql
    except sqlite3.Error as e:
        # ── one self-correction retry on execution error ──────────────────────
        error_context = (
            f'The following SQL failed with error "{e}":\n{sql}\n'
            f'Please write a corrected SELECT statement for the question above.'
        )
        retry_prompt = _build_prompt(question, error_context)
        raw2 = await call_llm(retry_prompt, temperature=0.0)
        sql2 = _strip_fences(raw2)

        valid2, reason2 = validate_generated_sql(sql2)
        if not valid2:
            raise FallbackError(
                f"Retry SQL also failed validation: {reason2}\nSQL: {sql2}"
            )

        try:
            rows2 = conn.execute(sql2).fetchall()
            return [dict(r) for r in rows2], sql2
        except sqlite3.Error as e2:
            raise FallbackError(f"Retry SQL execution failed: {e2}\nSQL: {sql2}")
