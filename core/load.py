"""
CSV → SQLite loader. Idempotent: drops + recreates the hotels table on every run.
"""

from __future__ import annotations

import csv
from pathlib import Path

from core.db import connect

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_CSV = REPO_ROOT / "hotels_synthetic.csv"
DEFAULT_DB = REPO_ROOT / "hotels.db"

AMENITY_COLS = [
    "has_wifi",
    "has_pool",
    "has_gym",
    "has_sauna",
    "has_restaurant",
    "has_room_service",
    "has_lounge",
    "has_event_space",
]
INSERT_COLS = [
    "name", "address", "city", "country",
    "latitude", "longitude", "avg_score", "total_reviews", "price_per_night",
    *AMENITY_COLS,
]


def _row_to_tuple(row: dict) -> tuple:
    return (
        row["name"],
        row.get("address"),
        row.get("city"),
        row.get("country"),
        float(row["latitude"]),
        float(row["longitude"]),
        float(row["avg_score"]) if row.get("avg_score") else None,
        int(row["total_reviews"]) if row.get("total_reviews") else None,
        int(row["price_per_night"]) if row.get("price_per_night") else None,
        *(int(row.get(c, 0) or 0) for c in AMENITY_COLS),
    )


def load_csv(csv_path: Path = DEFAULT_CSV, db_path: Path = DEFAULT_DB) -> int:
    schema = SCHEMA_PATH.read_text()

    with connect(db_path) as conn:
        conn.executescript(schema)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [_row_to_tuple(r) for r in reader]

        placeholders = ", ".join(["?"] * len(INSERT_COLS))
        sql = f"INSERT INTO hotels ({', '.join(INSERT_COLS)}) VALUES ({placeholders})"
        conn.executemany(sql, rows)
        conn.commit()

    return len(rows)
