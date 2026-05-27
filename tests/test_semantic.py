"""Semantic path tests — all deterministic, no LLM calls."""
from pathlib import Path

import pytest

from core.db import get_connection
from core.models import QuerySpec
from core.semantic import compile_query_spec, execute_semantic_query

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hotels.db"


@pytest.fixture(scope="module")
def conn():
    c = get_connection(DB_PATH)
    yield c
    c.close()


# ── count queries ──────────────────────────────────────────────────────────────

def test_count_all(conn):
    rows = execute_semantic_query(QuerySpec(metric="count"), conn)
    assert len(rows) == 1
    assert rows[0]["value"] == 1000


def test_count_by_city(conn):
    rows = execute_semantic_query(QuerySpec(metric="count", group_by="city"), conn)
    assert len(rows) == 6  # London, Paris, Barcelona, Milan, Vienna, Amsterdam
    total = sum(r["value"] for r in rows)
    assert total == 1000


def test_count_filtered_by_city(conn):
    rows = execute_semantic_query(
        QuerySpec(metric="count", filters={"city": "Paris"}), conn
    )
    assert len(rows) == 1
    # Verify against direct DB query
    direct = conn.execute(
        "SELECT COUNT(*) FROM hotels WHERE LOWER(city) = LOWER('Paris')"
    ).fetchone()[0]
    assert rows[0]["value"] == direct


# ── avg_score queries ──────────────────────────────────────────────────────────

def test_avg_score_by_city(conn):
    rows = execute_semantic_query(QuerySpec(metric="avg_score", group_by="city"), conn)
    assert len(rows) == 6
    for r in rows:
        assert 1.0 <= r["value"] <= 10.0


def test_avg_score_filtered(conn):
    rows = execute_semantic_query(
        QuerySpec(metric="avg_score", filters={"city": "London"}), conn
    )
    assert len(rows) == 1
    direct = conn.execute(
        "SELECT ROUND(AVG(avg_score), 2) FROM hotels WHERE LOWER(city) = LOWER('London')"
    ).fetchone()[0]
    assert abs(rows[0]["value"] - direct) < 0.01


# ── price metrics ──────────────────────────────────────────────────────────────

def test_avg_price_global(conn):
    rows = execute_semantic_query(QuerySpec(metric="avg_price"), conn)
    assert len(rows) == 1
    assert 80 <= rows[0]["value"] <= 465


def test_avg_price_by_city(conn):
    rows = execute_semantic_query(QuerySpec(metric="avg_price", group_by="city"), conn)
    assert len(rows) == 6
    for r in rows:
        assert 80 <= r["value"] <= 465


def test_min_price(conn):
    rows = execute_semantic_query(QuerySpec(metric="min_price"), conn)
    assert rows[0]["value"] >= 80


def test_max_price(conn):
    rows = execute_semantic_query(QuerySpec(metric="max_price"), conn)
    assert rows[0]["value"] <= 465


# ── amenity filters in semantic path ──────────────────────────────────────────

def test_count_hotels_with_pool(conn):
    rows = execute_semantic_query(
        QuerySpec(metric="count", filters={"has_pool": 1}), conn
    )
    direct = conn.execute("SELECT COUNT(*) FROM hotels WHERE has_pool = 1").fetchone()[0]
    assert rows[0]["value"] == direct


def test_avg_price_for_hotels_with_pool(conn):
    rows = execute_semantic_query(
        QuerySpec(metric="avg_price", filters={"has_pool": 1}), conn
    )
    assert len(rows) == 1
    direct = conn.execute(
        "SELECT ROUND(AVG(price_per_night), 0) FROM hotels WHERE has_pool = 1"
    ).fetchone()[0]
    assert abs(rows[0]["value"] - direct) <= 1


# ── sort and limit ─────────────────────────────────────────────────────────────

def test_count_by_city_sorted_desc(conn):
    rows = execute_semantic_query(
        QuerySpec(metric="count", group_by="city", sort_order="desc"), conn
    )
    values = [r["value"] for r in rows]
    assert values == sorted(values, reverse=True)


def test_limit(conn):
    rows = execute_semantic_query(
        QuerySpec(metric="count", group_by="city", limit=3), conn
    )
    assert len(rows) == 3


# ── validation errors ──────────────────────────────────────────────────────────

def test_invalid_metric_raises(conn):
    with pytest.raises(ValueError, match="Unknown metric"):
        compile_query_spec(QuerySpec(metric="price"))  # not a valid metric key


def test_invalid_group_by_raises(conn):
    with pytest.raises(ValueError, match="Unknown group_by"):
        compile_query_spec(QuerySpec(metric="count", group_by="has_pool"))


def test_invalid_filter_raises(conn):
    with pytest.raises(ValueError, match="Unknown filter"):
        compile_query_spec(QuerySpec(metric="count", filters={"stars": 5}))
