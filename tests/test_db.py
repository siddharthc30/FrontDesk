"""Phase 0 acceptance tests for core/db.py (updated for 17-column synthetic schema)."""
import sqlite3
from pathlib import Path

import pytest

from core.db import get_connection, get_schema_info, load_csv_to_sqlite

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hotels.db"

EXPECTED_COLUMNS = {
    "id", "name", "address", "city", "country",
    "latitude", "longitude", "avg_score", "total_reviews",
    "price_per_night",
    "has_wifi", "has_pool", "has_gym", "has_sauna",
    "has_restaurant", "has_room_service", "has_lounge", "has_event_space",
}


@pytest.fixture(scope="module")
def conn():
    c = get_connection(DB_PATH)
    yield c
    c.close()


# ── haversine ──────────────────────────────────────────────────────────────────

def test_haversine_london_to_paris(conn):
    dist = conn.execute(
        "SELECT haversine(51.5074, -0.1278, 48.8566, 2.3522)"
    ).fetchone()[0]
    assert abs(dist - 343) <= 5, f"Expected ~343 km, got {dist:.1f}"


def test_haversine_same_point_zero(conn):
    dist = conn.execute(
        "SELECT haversine(48.8566, 2.3522, 48.8566, 2.3522)"
    ).fetchone()[0]
    assert dist == pytest.approx(0.0, abs=1e-9)


# ── read-only connection ────────────────────────────────────────────────────────

def test_connection_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO hotels (name, latitude, longitude) VALUES ('X', 0, 0)")


# ── schema: core columns ───────────────────────────────────────────────────────

def test_schema_info_column_names(conn):
    info = get_schema_info(conn)
    assert "hotels" in info
    col_names = {c["name"] for c in info["hotels"]["columns"]}
    assert col_names >= EXPECTED_COLUMNS, (
        f"Missing columns: {EXPECTED_COLUMNS - col_names}"
    )


def test_schema_info_row_count(conn):
    info = get_schema_info(conn)
    assert info["hotels"]["row_count"] == 1000


def test_schema_info_sample_rows(conn):
    info = get_schema_info(conn)
    assert len(info["hotels"]["sample_rows"]) == 3


# ── new columns: price + amenities ────────────────────────────────────────────

def test_price_per_night_populated(conn):
    count = conn.execute(
        "SELECT COUNT(*) FROM hotels WHERE price_per_night IS NOT NULL"
    ).fetchone()[0]
    assert count == 1000, "All rows should have a price"


def test_price_range(conn):
    row = conn.execute(
        "SELECT MIN(price_per_night), MAX(price_per_night) FROM hotels"
    ).fetchone()
    assert row[0] >= 80
    assert row[1] <= 465


def test_amenity_columns_are_binary(conn):
    amenities = [
        "has_wifi", "has_pool", "has_gym", "has_sauna",
        "has_restaurant", "has_room_service", "has_lounge", "has_event_space",
    ]
    for col in amenities:
        bad = conn.execute(
            f"SELECT COUNT(*) FROM hotels WHERE {col} NOT IN (0, 1)"
        ).fetchone()[0]
        assert bad == 0, f"Column {col} has non-binary values"


def test_some_hotels_have_pool(conn):
    count = conn.execute("SELECT COUNT(*) FROM hotels WHERE has_pool = 1").fetchone()[0]
    assert count > 0, "Expected at least some hotels to have a pool"


def test_no_extra_columns(conn):
    info = get_schema_info(conn)
    col_names = {c["name"] for c in info["hotels"]["columns"]}
    unexpected = col_names - EXPECTED_COLUMNS
    assert not unexpected, f"Unexpected columns: {unexpected}"
