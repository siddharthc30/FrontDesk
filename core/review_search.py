"""
Review-search path: FTS5 full-text search over guest review text.
Used for novel/unknown aspects not covered by precomputed hotel_aspect_sentiment.
Fully deterministic — no LLM involved.
"""

from __future__ import annotations

import sqlite3

from core.observability import observe

# Placeholder strings in the source data that should be excluded from matches.
_PLACEHOLDERS = {"no positive", "no negative"}


def _has_term_in_text(term: str, text: str) -> bool:
    """Case-insensitive substring check, excluding placeholder strings."""
    if not text:
        return False
    lower = text.lower()
    if lower.strip() in _PLACEHOLDERS:
        return False
    return term.lower() in lower


@observe(name="search_reviews")
def search_reviews(
    terms: list[str],
    city: str | None = None,
    min_rating: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search reviews by free-text terms and aggregate per hotel.

    Steps:
    1. FTS5 query to find matching reviews (OR across all terms).
    2. For each match, check which column (positive_review / negative_review) contains the term.
    3. Aggregate per hotel: mention_count, pos_count, neg_count, sentiment = pos/(pos+neg).
    4. JOIN with hotels for metadata; apply city/rating filters.
    5. Sort by mention_count desc, apply limit.

    Returns list of dicts with hotel metadata + ad-hoc sentiment stats.
    Confidence for this path is always "medium".
    """
    from core.db import get_connection  # deferred to avoid circular imports at module load

    import os
    db_path = os.environ.get("DB_PATH", "data/hotels.db")

    if not terms:
        return []

    conn = get_connection(db_path)
    try:
        return _search_reviews_conn(terms, city, min_rating, limit, conn)
    finally:
        conn.close()


def search_reviews_conn(
    terms: list[str],
    conn: sqlite3.Connection,
    city: str | None = None,
    min_rating: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """Same as search_reviews but accepts an existing connection (for tests and pipeline)."""
    return _search_reviews_conn(terms, city, min_rating, limit, conn)


def _search_reviews_conn(
    terms: list[str],
    city: str | None,
    min_rating: float | None,
    limit: int,
    conn: sqlite3.Connection,
) -> list[dict]:
    if not terms:
        return []

    # ── FTS5 query ────────────────────────────────────────────────────────────
    fts_expr = " OR ".join(f'"{t}"' for t in terms)
    try:
        fts_rows = conn.execute(
            "SELECT rowid, positive_review, negative_review FROM reviews_fts "
            "WHERE reviews_fts MATCH ? "
            "ORDER BY rank",
            (fts_expr,),
        ).fetchall()
    except sqlite3.OperationalError:
        # FTS table not set up yet or no match supported
        return []

    if not fts_rows:
        return []

    # ── Determine polarity per review ─────────────────────────────────────────
    # Map review_id → hotel_id
    review_ids = [r[0] for r in fts_rows]
    placeholders = ",".join("?" * len(review_ids))
    hotel_map = {
        row[0]: row[1]
        for row in conn.execute(
            f"SELECT review_id, hotel_id FROM reviews WHERE review_id IN ({placeholders})",
            review_ids,
        ).fetchall()
    }

    # Accumulate per-hotel counts
    hotel_pos: dict[int, int] = {}
    hotel_neg: dict[int, int] = {}
    hotel_total: dict[int, int] = {}

    for review_id, pos_text, neg_text in fts_rows:
        hotel_id = hotel_map.get(review_id)
        if hotel_id is None:
            continue

        matched_pos = any(_has_term_in_text(t, pos_text or "") for t in terms)
        matched_neg = any(_has_term_in_text(t, neg_text or "") for t in terms)

        if not matched_pos and not matched_neg:
            continue  # FTS matched but term was in placeholder text

        hotel_total[hotel_id] = hotel_total.get(hotel_id, 0) + 1
        if matched_pos:
            hotel_pos[hotel_id] = hotel_pos.get(hotel_id, 0) + 1
        if matched_neg:
            hotel_neg[hotel_id] = hotel_neg.get(hotel_id, 0) + 1

    if not hotel_total:
        return []

    # ── JOIN with hotels ──────────────────────────────────────────────────────
    hotel_ids = list(hotel_total.keys())
    placeholders = ",".join("?" * len(hotel_ids))
    hotel_bind: list = list(hotel_ids)

    hotel_sql = (
        f"SELECT hotel_id, name, city, country, avg_score, price_per_night, "
        f"latitude, longitude "
        f"FROM hotels WHERE hotel_id IN ({placeholders})"
    )
    filter_clauses: list[str] = []
    if city is not None:
        filter_clauses.append("LOWER(city) = LOWER(?)")
        hotel_bind.append(city)
    if min_rating is not None:
        filter_clauses.append("avg_score >= ?")
        hotel_bind.append(min_rating)
    if filter_clauses:
        hotel_sql += " AND " + " AND ".join(filter_clauses)

    hotel_rows = conn.execute(hotel_sql, hotel_bind).fetchall()

    # ── Build result dicts ────────────────────────────────────────────────────
    results: list[dict] = []
    for row in hotel_rows:
        hid = row[0]
        pos = hotel_pos.get(hid, 0)
        neg = hotel_neg.get(hid, 0)
        total = hotel_total.get(hid, 0)
        denom = pos + neg
        sentiment = round(pos / denom, 3) if denom > 0 else None

        results.append({
            "hotel_id":       hid,
            "name":           row[1],
            "city":           row[2],
            "country":        row[3],
            "avg_score":      row[4],
            "price_per_night": row[5],
            "latitude":       row[6],
            "longitude":      row[7],
            "mention_count":  total,
            "pos_count":      pos,
            "neg_count":      neg,
            "sentiment":      sentiment,
            "search_terms":   terms,
            "confidence":     "medium",
        })

    # Sort by mention_count desc, then sentiment desc
    results.sort(key=lambda r: (r["mention_count"], r["sentiment"] or 0), reverse=True)
    return results[:limit]
