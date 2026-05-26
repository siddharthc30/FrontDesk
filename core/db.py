"""
SQLite connection factory + the haversine UDF the geo path relies on.

Everything that touches the DB goes through connect() so the eventual
Postgres swap is a config change, not a rewrite.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "hotels.db"
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return None
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def connect(path: str | Path = DEFAULT_DB_PATH, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    uri = f"file:{path}?mode=ro" if read_only else f"file:{path}"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.create_function("haversine", 4, _haversine_km, deterministic=True)
    return conn
