"""
Hardcoded sanity queries against hotels.db — the proof that the deterministic
spine (schema + haversine + query functions) works before any LLM is added.

Run:  .venv/bin/python -m scripts.smoke_queries
"""

from core.db import connect
from core.queries import (
    avg_score_by_city,
    count_by_country,
    hotels_near,
    hotels_with_amenities,
    top_hotels_by_score,
)

SEP = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def print_rows(rows, cols: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [max(len(c), max(len(str(r[c])) for r in rows)) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))


def main() -> None:
    with connect(read_only=True) as conn:
        section("1. TOP 10 HOTELS BY avg_score (global)")
        rows = top_hotels_by_score(conn, limit=10)
        print_rows(rows, ["name", "city", "avg_score", "total_reviews", "price_per_night"])

        section("2. TOP 10 IN LONDON")
        rows = top_hotels_by_score(conn, limit=10, city="London")
        print_rows(rows, ["name", "city", "avg_score", "total_reviews"])

        section("3. WITHIN 2 km OF LONDON CENTRE (51.5074, -0.1278)")
        rows = hotels_near(conn, lat=51.5074, lng=-0.1278, radius_km=2.0, limit=10)
        print_rows(rows, ["name", "city", "avg_score", "distance_km"])

        section("4. POOL + GYM IN PARIS")
        rows = hotels_with_amenities(conn, ["has_pool", "has_gym"], city="Paris")
        print_rows(rows, ["name", "city", "avg_score", "price_per_night"])

        section("5. AVG SCORE BY CITY")
        print_rows(avg_score_by_city(conn), ["city", "n", "mean_score"])

        section("6. COUNT BY COUNTRY")
        print_rows(count_by_country(conn), ["country", "n"])


if __name__ == "__main__":
    main()
