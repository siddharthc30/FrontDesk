"""
Semantic path: deterministic SQL compiler from QuerySpec.
The LLM emits a structured spec; this module turns it into SQL deterministically.
Column names in generated SQL come only from hardcoded allowlists, never from user input.
"""

from __future__ import annotations

import sqlite3

from core.models import QuerySpec
from core.observability import observe

# ── allowlists ───────────────────────────────────────────────────────────────

ALLOWED_METRICS: dict[str, str] = {
    "count":             "COUNT(*)",
    "avg_score":         "ROUND(AVG(avg_score), 2)",
    "avg_reviews":       "ROUND(AVG(total_reviews), 0)",
    "min_score":         "MIN(avg_score)",
    "max_score":         "MAX(avg_score)",
    "min_reviews":       "MIN(total_reviews)",
    "max_reviews":       "MAX(total_reviews)",
    "total_reviews_sum": "SUM(total_reviews)",
    "avg_price":         "ROUND(AVG(price_per_night), 0)",
    "min_price":         "MIN(price_per_night)",
    "max_price":         "MAX(price_per_night)",
}

ALLOWED_GROUP_BY: frozenset[str] = frozenset({
    "city",
    "country",
})

# filter_key -> (sql_fragment_with_placeholder, python_type)
ALLOWED_FILTERS: dict[str, tuple[str, type]] = {
    "city":          ("LOWER(city) = LOWER(?)",      str),
    "country":       ("LOWER(country) = LOWER(?)",   str),
    "min_rating":    ("avg_score >= ?",               float),
    "max_rating":    ("avg_score <= ?",               float),
    "min_reviews":   ("total_reviews >= ?",           int),
    "price_min":     ("price_per_night >= ?",         int),
    "price_max":     ("price_per_night <= ?",         int),
    # amenity flags — value must be 0 or 1
    "has_wifi":         ("has_wifi = ?",         int),
    "has_pool":         ("has_pool = ?",         int),
    "has_gym":          ("has_gym = ?",          int),
    "has_sauna":        ("has_sauna = ?",        int),
    "has_restaurant":   ("has_restaurant = ?",   int),
    "has_room_service": ("has_room_service = ?", int),
    "has_lounge":       ("has_lounge = ?",       int),
    "has_event_space":  ("has_event_space = ?",  int),
}


def compile_query_spec(spec: QuerySpec) -> tuple[str, list]:
    """Compile a QuerySpec into (sql_string, param_values).
    Raises ValueError if any field is not in the allowlists."""
    if spec.metric not in ALLOWED_METRICS:
        raise ValueError(
            f"Unknown metric '{spec.metric}'. Allowed: {sorted(ALLOWED_METRICS)}"
        )

    if spec.group_by is not None and spec.group_by not in ALLOWED_GROUP_BY:
        raise ValueError(
            f"Unknown group_by '{spec.group_by}'. Allowed: {sorted(ALLOWED_GROUP_BY)}"
        )

    metric_expr = ALLOWED_METRICS[spec.metric]
    params: list = []

    # ── SELECT ────────────────────────────────────────────────────────────────
    if spec.group_by:
        sql = f"SELECT {spec.group_by}, {metric_expr} AS value FROM hotels"
    else:
        sql = f"SELECT {metric_expr} AS value FROM hotels"

    # ── WHERE ─────────────────────────────────────────────────────────────────
    if spec.filters:
        clauses = []
        for key, val in spec.filters.items():
            if key not in ALLOWED_FILTERS:
                raise ValueError(
                    f"Unknown filter '{key}'. Allowed: {sorted(ALLOWED_FILTERS)}"
                )
            fragment, cast = ALLOWED_FILTERS[key]
            clauses.append(fragment)
            params.append(cast(val))
        sql += " WHERE " + " AND ".join(clauses)

    # ── GROUP BY ──────────────────────────────────────────────────────────────
    if spec.group_by:
        sql += f" GROUP BY {spec.group_by}"

    # ── ORDER BY ──────────────────────────────────────────────────────────────
    order = (spec.sort_order or "desc").upper()
    if order not in ("ASC", "DESC"):
        order = "DESC"
    sql += f" ORDER BY value {order}"

    # ── LIMIT ─────────────────────────────────────────────────────────────────
    if spec.limit is not None:
        sql += " LIMIT ?"
        params.append(int(spec.limit))

    return sql, params


@observe(name="execute_semantic")
def execute_semantic_query(spec: QuerySpec, conn: sqlite3.Connection) -> list[dict]:
    """Compile and execute a QuerySpec. Returns list of result dicts."""
    sql, params = compile_query_spec(spec)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]
