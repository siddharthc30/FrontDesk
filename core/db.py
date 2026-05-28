"""
SQLite connection factory, haversine UDF, CSV loader, schema introspection, and FTS5 setup.
All runtime DB access goes through get_connection() (read-only).
FTS5 setup (one-time write) is done via ensure_fts5().
"""

from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path

EARTH_RADIUS_KM = 6371.0

AMENITY_COLUMNS = [
    "has_wifi",
    "has_pool",
    "has_gym",
    "has_sauna",
    "has_restaurant",
    "has_room_service",
    "has_lounge",
    "has_event_space",
]


def _haversine_fn(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return None
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_csv_to_sqlite(csv_path: str | Path, db_path: str | Path) -> int:
    """Load hotels_synthetic.csv into SQLite, creating data/hotels.db.
    Adds a hotel_id INTEGER PRIMARY KEY AUTOINCREMENT column. Drops and recreates on every run."""
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS hotels")
        conn.execute("""
            CREATE TABLE hotels (
                hotel_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name             TEXT    NOT NULL,
                address          TEXT,
                city             TEXT,
                country          TEXT,
                latitude         REAL    NOT NULL,
                longitude        REAL    NOT NULL,
                avg_score        REAL,
                total_reviews    INTEGER,
                price_per_night  INTEGER,
                has_wifi         INTEGER NOT NULL DEFAULT 0,
                has_pool         INTEGER NOT NULL DEFAULT 0,
                has_gym          INTEGER NOT NULL DEFAULT 0,
                has_sauna        INTEGER NOT NULL DEFAULT 0,
                has_restaurant   INTEGER NOT NULL DEFAULT 0,
                has_room_service INTEGER NOT NULL DEFAULT 0,
                has_lounge       INTEGER NOT NULL DEFAULT 0,
                has_event_space  INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hotels_city      ON hotels(city)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hotels_country   ON hotels(country)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hotels_avg_score ON hotels(avg_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hotels_price     ON hotels(price_per_night)")

        insert_cols = [
            "name", "address", "city", "country",
            "latitude", "longitude", "avg_score", "total_reviews",
            "price_per_night", *AMENITY_COLUMNS,
        ]
        placeholders = ", ".join(["?"] * len(insert_cols))
        sql = f"INSERT INTO hotels ({', '.join(insert_cols)}) VALUES ({placeholders})"

        rows_inserted = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                conn.execute(sql, (
                    row["name"],
                    row.get("address"),
                    row.get("city"),
                    row.get("country"),
                    float(row["latitude"]),
                    float(row["longitude"]),
                    float(row["avg_score"]) if row.get("avg_score") else None,
                    int(row["total_reviews"]) if row.get("total_reviews") else None,
                    int(row["price_per_night"]) if row.get("price_per_night") else None,
                    *(int(row.get(col, 0) or 0) for col in AMENITY_COLUMNS),
                ))
                rows_inserted += 1

        conn.commit()
        return rows_inserted
    finally:
        conn.close()


def ensure_fts5(db_path: str | Path) -> None:
    """Create and populate the reviews_fts virtual table if it doesn't exist.
    Requires a writable connection — called once at app startup.
    Uses 'rebuild' because for a content= table, FTS5 must read from the source table."""
    db_path = Path(db_path).resolve()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts USING fts5(
                positive_review,
                negative_review,
                content='reviews',
                content_rowid='review_id'
            )
        """)
        existing = conn.execute(
            "SELECT COUNT(*) FROM reviews_fts_docsize"
        ).fetchone()[0]
        if existing == 0:
            conn.execute("INSERT INTO reviews_fts(reviews_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a read-only SQLite connection with haversine registered."""
    db_path = Path(db_path).resolve()
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.create_function("haversine", 4, _haversine_fn, deterministic=True)
    return conn


def get_schema_info(conn: sqlite3.Connection) -> dict:
    """Return table names, column names+types, row count, and 3 sample rows."""
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    info = {}
    for table in tables:
        columns = [
            {"name": r[1], "type": r[2], "notnull": bool(r[3]), "pk": bool(r[5])}
            for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
        ]
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        sample_rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table} LIMIT 3").fetchall()]
        info[table] = {
            "columns": columns,
            "row_count": row_count,
            "sample_rows": sample_rows,
        }
    return info


def get_data_dictionary() -> str:
    """Human-readable data dictionary injected into LLM prompts."""
    return """\
Table: hotels (1000 rows)
Primary key: hotel_id
Columns:
- hotel_id: INTEGER, primary key
- name: TEXT, the hotel's name
- address: TEXT, full street address
- city: TEXT, city name (values: London, Paris, Barcelona, Milan, Vienna, Amsterdam)
- country: TEXT, country name (values: United Kingdom, France, Spain, Italy, Austria, Netherlands)
- latitude: REAL, geographic latitude
- longitude: REAL, geographic longitude
- avg_score: REAL, average guest review score on a 1.0-10.0 scale
- total_reviews: INTEGER, total number of guest reviews
- price_per_night: INTEGER, representative nightly price in USD (range: 110-315)
- has_wifi: INTEGER (0 or 1), review-derived: guests mentioned Wi-Fi
- has_pool: INTEGER (0 or 1), review-derived: guests mentioned pool
- has_gym: INTEGER (0 or 1), review-derived: guests mentioned gym
- has_sauna: INTEGER (0 or 1), review-derived: guests mentioned sauna
- has_restaurant: INTEGER (0 or 1), review-derived: guests mentioned restaurant
- has_room_service: INTEGER (0 or 1), review-derived: guests mentioned room service
- has_lounge: INTEGER (0 or 1), review-derived: guests mentioned lounge
- has_event_space: INTEGER (0 or 1), review-derived: guests mentioned event space (unreliable — only 6/1000 detected)

Table: reviews (363429 rows)
Foreign key: hotel_id → hotels.hotel_id
Columns:
- review_id: INTEGER, primary key
- hotel_id: INTEGER, FK to hotels
- review_date: TEXT, date of review
- reviewer_nationality: TEXT
- positive_review: TEXT, guest's positive comments (may be "No Positive" if none)
- positive_word_count: INTEGER
- negative_review: TEXT, guest's negative comments (may be "No Negative" if none)
- negative_word_count: INTEGER
- reviewer_score: REAL, score given by the reviewer (1.0-10.0 scale)
- tags: TEXT, comma-separated tags from the reviewer
- days_since_review: TEXT

Table: hotel_aspect_sentiment (12427 rows)
Foreign key: hotel_id → hotels.hotel_id
Columns:
- hotel_id: INTEGER, FK to hotels
- aspect: TEXT, one of: wifi, pool, gym, sauna, restaurant, room_service, lounge, event_space,
  staff, cleanliness, location, room_comfort, value, noise, breakfast
- sentiment: REAL, fraction of positive mentions (0.0–1.0); typical ranges vary by aspect
- mention_count: INTEGER, total reviews mentioning this aspect
- pos_count: INTEGER, positive mentions
- neg_count: INTEGER, negative mentions
- is_facility: INTEGER (0 or 1), 1 = facility aspect (also has has_X column on hotels)

Available SQL function:
- haversine(lat1, lon1, lat2, lon2): returns distance in km between two coordinates

NOTE: Amenity columns are 0/1 integers derived from review text (mention ≠ confirmed presence). \
Price is a representative nightly rate in USD, not a live booking price. \
avg_score is on a 1-10 scale. sentiment in hotel_aspect_sentiment is a 0-1 ratio."""
