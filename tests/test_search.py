"""Parameterized path tests — all deterministic, no LLM calls."""
import math
from pathlib import Path

import pytest

from core.db import get_connection
from core.models import SearchParams
from core.search import AMENITY_ALLOWLIST, search_hotels

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hotels.db"


@pytest.fixture(scope="module")
def conn():
    c = get_connection(DB_PATH)
    yield c
    c.close()


# ── city filter ───────────────────────────────────────────────────────────────

def test_filter_by_city(conn):
    results = search_hotels(SearchParams(city="Paris", limit=20), conn)
    assert len(results) > 0
    assert all(r.city == "Paris" for r in results)


def test_filter_by_city_case_insensitive(conn):
    results = search_hotels(SearchParams(city="paris", limit=5), conn)
    assert len(results) > 0
    assert all(r.city == "Paris" for r in results)


# ── rating filters ────────────────────────────────────────────────────────────

def test_filter_min_rating(conn):
    results = search_hotels(SearchParams(min_rating=9.0, limit=20), conn)
    assert len(results) > 0
    assert all(r.avg_score >= 9.0 for r in results)


def test_combined_city_and_rating(conn):
    results = search_hotels(SearchParams(city="London", min_rating=8.0, limit=10), conn)
    assert all(r.city == "London" for r in results)
    assert all(r.avg_score >= 8.0 for r in results)


# ── price filters ─────────────────────────────────────────────────────────────

def test_filter_price_max(conn):
    results = search_hotels(SearchParams(price_max=150, limit=20), conn)
    assert len(results) > 0
    assert all(r.price_per_night <= 150 for r in results)


def test_filter_price_min(conn):
    results = search_hotels(SearchParams(price_min=400, limit=20), conn)
    assert len(results) > 0
    assert all(r.price_per_night >= 400 for r in results)


def test_filter_price_range(conn):
    results = search_hotels(SearchParams(price_min=100, price_max=200, limit=20), conn)
    assert all(100 <= r.price_per_night <= 200 for r in results)


# ── amenity filters ───────────────────────────────────────────────────────────

def test_filter_single_amenity_pool(conn):
    results = search_hotels(
        SearchParams(required_amenities=["has_pool"], limit=20), conn
    )
    assert len(results) > 0
    assert all(r.has_pool == 1 for r in results)


def test_filter_multiple_amenities(conn):
    results = search_hotels(
        SearchParams(required_amenities=["has_pool", "has_gym"], limit=20), conn
    )
    assert len(results) > 0
    assert all(r.has_pool == 1 and r.has_gym == 1 for r in results)


def test_filter_amenity_and_city(conn):
    results = search_hotels(
        SearchParams(city="Paris", required_amenities=["has_restaurant"], limit=10), conn
    )
    assert all(r.city == "Paris" for r in results)
    assert all(r.has_restaurant == 1 for r in results)


def test_invalid_amenity_raises(conn):
    with pytest.raises(ValueError, match="Unknown amenity"):
        search_hotels(SearchParams(required_amenities=["has_jacuzzi"]), conn)


# ── sort order ────────────────────────────────────────────────────────────────

def test_sort_by_avg_score_desc(conn):
    results = search_hotels(SearchParams(sort_by="avg_score", sort_order="desc", limit=10), conn)
    scores = [r.avg_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_sort_by_price_asc(conn):
    results = search_hotels(SearchParams(sort_by="price", sort_order="asc", limit=10), conn)
    prices = [r.price_per_night for r in results]
    assert prices == sorted(prices)


# ── limit ─────────────────────────────────────────────────────────────────────

def test_limit_respected(conn):
    for n in (1, 5, 15):
        results = search_hotels(SearchParams(limit=n), conn)
        assert len(results) == n


# ── geo radius ────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def test_geo_radius_all_within_bounds(conn):
    # Centre of Paris
    lat, lng, radius = 48.8566, 2.3522, 10.0
    results = search_hotels(
        SearchParams(user_lat=lat, user_lng=lng, radius_km=radius, limit=50), conn
    )
    assert len(results) > 0
    for r in results:
        dist = _haversine(lat, lng, r.latitude, r.longitude)
        assert dist <= radius + 0.5, f"{r.name} is {dist:.1f} km away, exceeds {radius} km"


def test_sort_by_distance(conn):
    lat, lng = 51.5074, -0.1278  # central London
    results = search_hotels(
        SearchParams(user_lat=lat, user_lng=lng, sort_by="distance", sort_order="asc", limit=10),
        conn,
    )
    dists = [_haversine(lat, lng, r.latitude, r.longitude) for r in results]
    assert dists == sorted(dists)
