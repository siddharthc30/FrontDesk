"""
Parameterized path: deterministic SQL builder from SearchParams.
The LLM never touches SQL here — it only fills a SearchParams form.
"""

from __future__ import annotations

import sqlite3

from core.models import HotelRow, SearchParams

AMENITY_ALLOWLIST: frozenset[str] = frozenset({
    "has_wifi",
    "has_pool",
    "has_gym",
    "has_sauna",
    "has_restaurant",
    "has_room_service",
    "has_lounge",
    "has_event_space",
})

SORT_COLUMN_MAP: dict[str, str] = {
    "avg_score":     "avg_score",
    "total_reviews": "total_reviews",
    "distance":      "distance_km",
    "price":         "price_per_night",
}


def search_hotels(params: SearchParams, conn: sqlite3.Connection) -> list[HotelRow]:
    """Build and execute a parameterized SELECT from SearchParams.
    All user-supplied values go through ? placeholders; column names come from allowlists."""
    has_geo = params.user_lat is not None and params.user_lng is not None

    if has_geo:
        select = (
            "SELECT *, haversine(latitude, longitude, ?, ?) AS distance_km "
            "FROM hotels WHERE 1=1"
        )
        bind: list = [params.user_lat, params.user_lng]
    else:
        select = "SELECT *, NULL AS distance_km FROM hotels WHERE 1=1"
        bind = []

    # ── scalar filters ──────────────────────────────────────────────────────
    if params.city is not None:
        select += " AND LOWER(city) = LOWER(?)"
        bind.append(params.city)

    if params.country is not None:
        select += " AND LOWER(country) = LOWER(?)"
        bind.append(params.country)

    if params.min_rating is not None:
        select += " AND avg_score >= ?"
        bind.append(params.min_rating)

    if params.max_rating is not None:
        select += " AND avg_score <= ?"
        bind.append(params.max_rating)

    if params.min_reviews is not None:
        select += " AND total_reviews >= ?"
        bind.append(params.min_reviews)

    if params.price_min is not None:
        select += " AND price_per_night >= ?"
        bind.append(params.price_min)

    if params.price_max is not None:
        select += " AND price_per_night <= ?"
        bind.append(params.price_max)

    # ── geo radius ───────────────────────────────────────────────────────────
    if params.radius_km is not None and has_geo:
        select += " AND haversine(latitude, longitude, ?, ?) <= ?"
        bind.extend([params.user_lat, params.user_lng, params.radius_km])

    # ── amenity flags (column names from allowlist, never from user input) ──
    if params.required_amenities:
        bad = [a for a in params.required_amenities if a not in AMENITY_ALLOWLIST]
        if bad:
            raise ValueError(f"Unknown amenity column(s): {bad}")
        for amenity in params.required_amenities:
            select += f" AND {amenity} = 1"  # column name from allowlist

    # ── ORDER BY ─────────────────────────────────────────────────────────────
    sort_col = SORT_COLUMN_MAP.get(params.sort_by or "") if params.sort_by else None
    if sort_col == "distance_km" and not has_geo:
        sort_col = None  # can't sort by distance without coordinates

    order = (params.sort_order or "desc").upper()
    if order not in ("ASC", "DESC"):
        order = "DESC"

    if sort_col:
        select += f" ORDER BY {sort_col} {order}"
    else:
        select += " ORDER BY avg_score DESC"

    # ── LIMIT ────────────────────────────────────────────────────────────────
    limit = params.limit if params.limit is not None else 10
    select += " LIMIT ?"
    bind.append(limit)

    rows = conn.execute(select, bind).fetchall()
    return [HotelRow(**dict(r)) for r in rows]
