"""
Deterministic query functions. Each is hand-written SQL that the eventual
bounded / semantic paths will compose. No string-interpolated user input
ever reaches SQL — amenity names go through a whitelist.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

AMENITY_WHITELIST = frozenset({
    "has_wifi", "has_pool", "has_gym", "has_sauna",
    "has_restaurant", "has_room_service", "has_lounge", "has_event_space",
})


def top_hotels_by_score(
    conn: sqlite3.Connection,
    limit: int = 10,
    city: str | None = None,
) -> list[sqlite3.Row]:
    sql = """
        SELECT name, city, country, avg_score, total_reviews, price_per_night
        FROM hotels
        WHERE (? IS NULL OR city = ?)
        ORDER BY avg_score DESC, total_reviews DESC
        LIMIT ?
    """
    return conn.execute(sql, (city, city, limit)).fetchall()


def hotels_near(
    conn: sqlite3.Connection,
    lat: float,
    lng: float,
    radius_km: float,
    limit: int = 10,
) -> list[sqlite3.Row]:
    sql = """
        SELECT
            name, city, country, latitude, longitude, avg_score,
            haversine(latitude, longitude, ?, ?) AS distance_km
        FROM hotels
        WHERE haversine(latitude, longitude, ?, ?) <= ?
        ORDER BY distance_km ASC
        LIMIT ?
    """
    return conn.execute(sql, (lat, lng, lat, lng, radius_km, limit)).fetchall()


def hotels_with_amenities(
    conn: sqlite3.Connection,
    amenities: Iterable[str],
    city: str | None = None,
    limit: int = 20,
) -> list[sqlite3.Row]:
    amenities = list(amenities)
    bad = [a for a in amenities if a not in AMENITY_WHITELIST]
    if bad:
        raise ValueError(f"unknown amenity column(s): {bad}")

    amenity_clause = " AND ".join(f"{a} = 1" for a in amenities) or "1 = 1"
    sql = f"""
        SELECT name, city, country, avg_score, price_per_night
        FROM hotels
        WHERE (? IS NULL OR city = ?)
          AND {amenity_clause}
        ORDER BY avg_score DESC
        LIMIT ?
    """
    return conn.execute(sql, (city, city, limit)).fetchall()


def avg_score_by_city(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    sql = """
        SELECT city, COUNT(*) AS n, ROUND(AVG(avg_score), 2) AS mean_score
        FROM hotels
        GROUP BY city
        ORDER BY mean_score DESC
    """
    return conn.execute(sql).fetchall()


def count_by_country(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    sql = """
        SELECT country, COUNT(*) AS n
        FROM hotels
        GROUP BY country
        ORDER BY n DESC
    """
    return conn.execute(sql).fetchall()
