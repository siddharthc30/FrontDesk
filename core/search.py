"""
Parameterized path: deterministic SQL builder from SearchParams.
The LLM never touches SQL here — it only fills a SearchParams form.
"""

from __future__ import annotations

import sqlite3

from core.models import HotelRow, SearchParams
from core.observability import observe

AMENITY_ALLOWLIST: frozenset[str] = frozenset({
    "has_wifi",
    "has_pool",
    "has_gym",
    "has_sauna",
    "has_restaurant",
    "has_room_service",
    "has_lounge",
    # has_event_space excluded: unreliable (only 6/1000 detected)
})

KNOWN_ASPECTS: frozenset[str] = frozenset({
    "wifi", "pool", "gym", "sauna", "restaurant", "room_service", "lounge",
    "staff", "cleanliness", "location", "room_comfort", "value", "noise", "breakfast",
})

SORT_COLUMN_MAP: dict[str, str] = {
    "avg_score":     "avg_score",
    "total_reviews": "total_reviews",
    "distance":      "distance_km",
    "price":         "price_per_night",
}


def _aspect_sort_column(sort_by: str) -> str | None:
    """If sort_by is '<aspect>_sentiment', return the aspect name; else None."""
    if sort_by and sort_by.endswith("_sentiment"):
        aspect = sort_by[: -len("_sentiment")]
        if aspect in KNOWN_ASPECTS:
            return aspect
    return None


@observe(name="search_hotels")
def search_hotels(params: SearchParams, conn: sqlite3.Connection) -> list[HotelRow]:
    """Build and execute a parameterized SELECT from SearchParams.
    All user-supplied values go through ? placeholders; column names come from allowlists."""
    has_geo = params.user_lat is not None and params.user_lng is not None

    # ── Determine if we need a JOIN to hotel_aspect_sentiment ────────────────
    aspect_sort = _aspect_sort_column(params.sort_by or "")
    needs_aspect_join = bool(params.aspect_filters) or aspect_sort is not None

    if needs_aspect_join:
        # Numbered JOINs (af0, af1, ...) for aspect filters; sort_as for sort aspect.
        join_clauses: list[str] = []
        join_bind: list = []

        if params.aspect_filters:
            bad_aspects = [
                f.aspect for f in params.aspect_filters
                if f.aspect not in KNOWN_ASPECTS
            ]
            if bad_aspects:
                raise ValueError(f"Unknown aspect(s): {bad_aspects}")

            for i, af in enumerate(params.aspect_filters):
                alias = f"af{i}"
                join_clauses.append(
                    f"JOIN hotel_aspect_sentiment {alias} "
                    f"ON {alias}.hotel_id = h.hotel_id "
                    f"AND {alias}.aspect = ? "
                    f"AND {alias}.sentiment >= ? "
                    f"AND {alias}.mention_count >= ?"
                )
                join_bind.extend([af.aspect, af.min_sentiment, af.min_mentions])

        if aspect_sort is not None:
            # Always add a dedicated sort_as alias for the sort aspect
            join_clauses.append(
                "LEFT JOIN hotel_aspect_sentiment sort_as "
                "ON sort_as.hotel_id = h.hotel_id "
                "AND sort_as.aspect = ?"
            )
            join_bind.append(aspect_sort)

        if has_geo:
            select = (
                "SELECT h.*, haversine(h.latitude, h.longitude, ?, ?) AS distance_km "
                "FROM hotels h "
            )
            bind: list = [params.user_lat, params.user_lng]
        else:
            select = "SELECT h.*, NULL AS distance_km FROM hotels h "
            bind = []

        select += " ".join(join_clauses) + " WHERE 1=1"
        bind.extend(join_bind)
    else:
        if has_geo:
            select = (
                "SELECT *, haversine(latitude, longitude, ?, ?) AS distance_km "
                "FROM hotels WHERE 1=1"
            )
            bind = [params.user_lat, params.user_lng]
        else:
            select = "SELECT *, NULL AS distance_km FROM hotels WHERE 1=1"
            bind = []

    # ── scalar filters ──────────────────────────────────────────────────────
    prefix = "h." if needs_aspect_join else ""

    if params.city is not None:
        select += f" AND LOWER({prefix}city) = LOWER(?)"
        bind.append(params.city)

    if params.country is not None:
        select += f" AND LOWER({prefix}country) = LOWER(?)"
        bind.append(params.country)

    if params.min_rating is not None:
        select += f" AND {prefix}avg_score >= ?"
        bind.append(params.min_rating)

    if params.max_rating is not None:
        select += f" AND {prefix}avg_score <= ?"
        bind.append(params.max_rating)

    if params.min_reviews is not None:
        select += f" AND {prefix}total_reviews >= ?"
        bind.append(params.min_reviews)

    if params.price_min is not None:
        select += f" AND {prefix}price_per_night >= ?"
        bind.append(params.price_min)

    if params.price_max is not None:
        select += f" AND {prefix}price_per_night <= ?"
        bind.append(params.price_max)

    # ── geo radius ───────────────────────────────────────────────────────────
    if params.radius_km is not None and has_geo:
        select += f" AND haversine({prefix}latitude, {prefix}longitude, ?, ?) <= ?"
        bind.extend([params.user_lat, params.user_lng, params.radius_km])

    # ── amenity flags (column names from allowlist, never from user input) ──
    if params.required_amenities:
        bad = [a for a in params.required_amenities if a not in AMENITY_ALLOWLIST]
        if bad:
            raise ValueError(f"Unknown amenity column(s): {bad}")
        for amenity in params.required_amenities:
            select += f" AND {prefix}{amenity} = 1"  # column name from allowlist

    # ── ORDER BY ─────────────────────────────────────────────────────────────
    if aspect_sort:
        order = (params.sort_order or "desc").upper()
        if order not in ("ASC", "DESC"):
            order = "DESC"
        select += f" ORDER BY sort_as.sentiment {order}"
    else:
        sort_col = SORT_COLUMN_MAP.get(params.sort_by or "") if params.sort_by else None
        if sort_col == "distance_km" and not has_geo:
            sort_col = None

        order = (params.sort_order or "desc").upper()
        if order not in ("ASC", "DESC"):
            order = "DESC"

        if sort_col:
            select += f" ORDER BY {sort_col} {order}"
        else:
            select += f" ORDER BY {prefix}avg_score DESC"

    # ── LIMIT ────────────────────────────────────────────────────────────────
    if params.limit is not None:
        select += " LIMIT ?"
        bind.append(params.limit)

    rows = conn.execute(select, bind).fetchall()
    return [HotelRow(**dict(r)) for r in rows]
