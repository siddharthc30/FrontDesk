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
    # aspect-sentiment metric — requires spec.aspect to be set
    "avg_sentiment":     "ROUND(AVG(a.sentiment), 3)",
}

ALLOWED_GROUP_BY: frozenset[str] = frozenset({
    "city",
    "country",
    "aspect",  # group across all aspects of a hotel set
})

# filter_key -> (sql_fragment_with_placeholder, python_type)
# Filters on the hotels table (h alias when joining aspect table)
ALLOWED_FILTERS: dict[str, tuple[str, type]] = {
    "city":          ("LOWER(h.city) = LOWER(?)",      str),
    "country":       ("LOWER(h.country) = LOWER(?)",   str),
    "min_rating":    ("h.avg_score >= ?",               float),
    "max_rating":    ("h.avg_score <= ?",               float),
    "min_reviews":   ("h.total_reviews >= ?",           int),
    "price_min":     ("h.price_per_night >= ?",         int),
    "price_max":     ("h.price_per_night <= ?",         int),
    "has_wifi":         ("h.has_wifi = ?",         int),
    "has_pool":         ("h.has_pool = ?",         int),
    "has_gym":          ("h.has_gym = ?",          int),
    "has_sauna":        ("h.has_sauna = ?",        int),
    "has_restaurant":   ("h.has_restaurant = ?",   int),
    "has_room_service": ("h.has_room_service = ?", int),
    "has_lounge":       ("h.has_lounge = ?",       int),
    "has_event_space":  ("h.has_event_space = ?",  int),
}

# Separate filter map for the non-aspect (hotels-only) path with unaliased columns
ALLOWED_FILTERS_PLAIN: dict[str, tuple[str, type]] = {
    k: (v[0].replace("h.", ""), v[1])
    for k, v in ALLOWED_FILTERS.items()
}

KNOWN_ASPECTS: frozenset[str] = frozenset({
    "wifi", "pool", "gym", "sauna", "restaurant", "room_service", "lounge",
    "staff", "cleanliness", "location", "room_comfort", "value", "noise", "breakfast",
})


def compile_query_spec(spec: QuerySpec) -> tuple[str, list]:
    """Compile a QuerySpec into (sql_string, param_values).
    Raises ValueError if any field is not in the allowlists."""
    if spec.metric not in ALLOWED_METRICS:
        raise ValueError(
            f"Unknown metric '{spec.metric}'. Allowed: {sorted(ALLOWED_METRICS)}"
        )

    if spec.metric == "avg_sentiment" and not spec.aspect and spec.group_by != "aspect":
        raise ValueError("metric 'avg_sentiment' requires an 'aspect' to be specified (or group_by='aspect')")

    if spec.aspect and spec.aspect not in KNOWN_ASPECTS:
        raise ValueError(
            f"Unknown aspect '{spec.aspect}'. Allowed: {sorted(KNOWN_ASPECTS)}"
        )

    if spec.group_by is not None and spec.group_by not in ALLOWED_GROUP_BY:
        raise ValueError(
            f"Unknown group_by '{spec.group_by}'. Allowed: {sorted(ALLOWED_GROUP_BY)}"
        )

    uses_aspect_table = spec.metric == "avg_sentiment" or spec.group_by == "aspect"
    metric_expr = ALLOWED_METRICS[spec.metric]
    params: list = []

    if uses_aspect_table:
        # ── SELECT with aspect JOIN ───────────────────────────────────────────
        group_col = spec.group_by if spec.group_by in ("city", "country", "aspect") else None

        if group_col == "aspect":
            sql = f"SELECT a.aspect, {metric_expr} AS value FROM hotels h JOIN hotel_aspect_sentiment a ON a.hotel_id = h.hotel_id"
        elif group_col:
            sql = f"SELECT h.{group_col}, {metric_expr} AS value FROM hotels h JOIN hotel_aspect_sentiment a ON a.hotel_id = h.hotel_id"
        else:
            sql = f"SELECT {metric_expr} AS value FROM hotels h JOIN hotel_aspect_sentiment a ON a.hotel_id = h.hotel_id"

        # ── WHERE ─────────────────────────────────────────────────────────────
        clauses: list[str] = []
        if spec.aspect:
            clauses.append("a.aspect = ?")
            params.append(spec.aspect)

        if spec.filters:
            for key, val in spec.filters.items():
                if key not in ALLOWED_FILTERS:
                    raise ValueError(
                        f"Unknown filter '{key}'. Allowed: {sorted(ALLOWED_FILTERS)}"
                    )
                fragment, cast = ALLOWED_FILTERS[key]
                clauses.append(fragment)
                params.append(cast(val))

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        # ── GROUP BY ──────────────────────────────────────────────────────────
        if group_col == "aspect":
            sql += " GROUP BY a.aspect"
        elif group_col:
            sql += f" GROUP BY h.{group_col}"

    else:
        # ── SELECT hotels only (original path) ───────────────────────────────
        if spec.group_by:
            sql = f"SELECT {spec.group_by}, {metric_expr} AS value FROM hotels"
        else:
            sql = f"SELECT {metric_expr} AS value FROM hotels"

        # ── WHERE ─────────────────────────────────────────────────────────────
        if spec.filters:
            clauses = []
            for key, val in spec.filters.items():
                if key not in ALLOWED_FILTERS_PLAIN:
                    raise ValueError(
                        f"Unknown filter '{key}'. Allowed: {sorted(ALLOWED_FILTERS_PLAIN)}"
                    )
                fragment, cast = ALLOWED_FILTERS_PLAIN[key]
                clauses.append(fragment)
                params.append(cast(val))
            sql += " WHERE " + " AND ".join(clauses)

        # ── GROUP BY ──────────────────────────────────────────────────────────
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
