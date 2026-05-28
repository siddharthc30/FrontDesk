"""Phase 2 acceptance tests — all deterministic, no LLM calls."""
from pathlib import Path

import pytest

from core.db import get_connection
from core.fallback import validate_generated_sql

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hotels.db"


@pytest.fixture(scope="module")
def conn():
    c = get_connection(DB_PATH)
    yield c
    c.close()


# ── rejection tests ───────────────────────────────────────────────────────────

def test_reject_drop_table():
    valid, reason = validate_generated_sql("DROP TABLE hotels")
    assert not valid
    assert reason


def test_reject_insert():
    valid, reason = validate_generated_sql(
        "INSERT INTO hotels (name, latitude, longitude) VALUES ('x', 0, 0)"
    )
    assert not valid
    assert reason


def test_reject_unknown_table():
    valid, reason = validate_generated_sql("SELECT * FROM users")
    assert not valid
    assert "users" in reason.lower() or reason


def test_reject_unknown_column():
    # "price" is not a valid column — the real column is "price_per_night"
    valid, reason = validate_generated_sql("SELECT price FROM hotels")
    assert not valid
    assert reason


def test_reject_multiple_statements():
    valid, reason = validate_generated_sql(
        "SELECT * FROM hotels; SELECT * FROM hotels"
    )
    assert not valid
    assert reason


# ── acceptance tests ──────────────────────────────────────────────────────────

def test_valid_select_all():
    valid, reason = validate_generated_sql("SELECT * FROM hotels LIMIT 5")
    assert valid
    assert reason == ""


def test_valid_select_with_where():
    sql = (
        "SELECT name, avg_score FROM hotels "
        "WHERE LOWER(city) = LOWER('Paris') "
        "ORDER BY avg_score DESC LIMIT 5"
    )
    valid, reason = validate_generated_sql(sql)
    assert valid
    assert reason == ""


def test_valid_most_reviews_sql():
    sql = "SELECT * FROM hotels ORDER BY total_reviews DESC LIMIT 1"
    valid, reason = validate_generated_sql(sql)
    assert valid
    assert reason == ""


# ── execution test (no LLM — hardcoded SQL) ───────────────────────────────────

def test_execute_valid_sql_returns_results(conn):
    sql = "SELECT * FROM hotels ORDER BY total_reviews DESC LIMIT 1"
    valid, _ = validate_generated_sql(sql)
    assert valid

    rows = conn.execute(sql).fetchall()
    assert len(rows) == 1

    row = dict(rows[0])
    max_reviews = conn.execute(
        "SELECT MAX(total_reviews) FROM hotels"
    ).fetchone()[0]
    assert row["total_reviews"] == max_reviews


# ── Phase 7: new tables in fallback ──────────────────────────────────────────

def test_valid_reviews_join_sql():
    sql = (
        "SELECT h.city, ROUND(AVG(h.avg_score), 2) AS hotel_avg_score, "
        "ROUND(AVG(r.reviewer_score), 2) AS reviewer_avg_score "
        "FROM hotels h JOIN reviews r ON r.hotel_id = h.hotel_id "
        "GROUP BY h.city ORDER BY hotel_avg_score DESC"
    )
    valid, reason = validate_generated_sql(sql)
    assert valid, reason


def test_valid_aspect_sentiment_sql():
    sql = (
        "SELECT aspect, ROUND(AVG(sentiment), 3) AS avg_sentiment "
        "FROM hotel_aspect_sentiment GROUP BY aspect ORDER BY avg_sentiment DESC"
    )
    valid, reason = validate_generated_sql(sql)
    assert valid, reason


def test_valid_most_reviews_by_hotel():
    sql = "SELECT hotel_id, name, total_reviews FROM hotels ORDER BY total_reviews DESC LIMIT 10"
    valid, reason = validate_generated_sql(sql)
    assert valid, reason


def test_reject_reviews_unknown_column():
    valid, reason = validate_generated_sql(
        "SELECT unknown_col FROM reviews LIMIT 5"
    )
    assert not valid


def test_execute_aspect_sentiment_query(conn):
    sql = (
        "SELECT aspect, ROUND(AVG(sentiment), 3) AS avg_sentiment "
        "FROM hotel_aspect_sentiment GROUP BY aspect ORDER BY avg_sentiment DESC"
    )
    valid, _ = validate_generated_sql(sql)
    assert valid
    rows = conn.execute(sql).fetchall()
    assert len(rows) > 0
    aspects = [dict(r)["aspect"] for r in rows]
    assert "staff" in aspects or "location" in aspects
