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
    # Verify it's actually the hotel with the most reviews by checking against the DB
    max_reviews = conn.execute(
        "SELECT MAX(total_reviews) FROM hotels"
    ).fetchone()[0]
    assert row["total_reviews"] == max_reviews
