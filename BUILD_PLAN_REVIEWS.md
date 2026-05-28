# BUILD_PLAN_REVIEWS.md — Adding Reviews + Aspect Sentiment

**Audience:** Claude Code (autonomous build agent).
**Context:** The hotel NL-search app is **already built and working** per the original
`BUILD_PLAN.md` — parameterized path, semantic path, SQL fallback, router, narration,
FastAPI, and Streamlit are all functional. This plan adds **review-based sentiment** as an
incremental feature on top of the existing codebase.

**How to use this file:** Work through the phases **in order**. Each phase has a Goal,
Tasks, and Acceptance criteria. Do not start a phase until the previous phase's acceptance
criteria pass. After each phase, stop and report what you did and how you verified it.

**Prime directive:** the existing app must keep working at every step. Extend, don't rebuild.

---

## 0. What changed in the data

The SQLite database (`hotels.db`) has been **replaced** with a new version containing:

| What | Before | After |
|---|---|---|
| Tables | 1 (`hotels`) | 3 (`hotels`, `reviews`, `hotel_aspect_sentiment`) |
| Hotels table | no `hotel_id`, random amenity flags, loose price | `hotel_id` PK added, amenity flags derived from reviews, principled price |
| Reviews | did not exist | 363,429 guest reviews with FK to hotels |
| Aspect sentiment | did not exist | 12,427 precomputed per-hotel per-aspect sentiment rows |

**Two new CSV files** are also in the repo: `hotel_aspect_sentiment.csv` and `reviews.csv`
(flat exports of the new tables, for inspection and for the DB build script).

The `hotels` table schema is backwards-compatible in shape (same column names for amenities
and price) but values have changed — flags are now review-derived, not random. One column was
**added**: `hotel_id` (INTEGER PK). If existing code references hotels by rowid or by name,
update to use `hotel_id`.

---

## 1. What's new architecturally

One new execution tier joins the existing three:

| Existing path | Change |
|---|---|
| Parameterized | **Extend:** add aspect-sentiment filtering (e.g. pool sentiment ≥ 0.7) |
| Semantic | **Extend:** add aspect-sentiment metrics/dimensions (avg sentiment, group-by aspect) |
| SQL fallback | **Extend:** update data dictionary + few-shot examples to cover all 3 tables |
| Router | **Extend:** add `review_search` as a 4th routing option; add known-aspect taxonomy |
| Narration | **Extend:** add confidence-level framing for review-search results |
| **Review-search** | **NEW:** FTS5 full-text search over reviews for novel/unknown aspects |

**Response contract update:** add a `confidence` field (`"high"` | `"medium"` | `"guarded"`)
mapped to the path that ran. Review-search → `"medium"`.

---

## 2. Known aspect taxonomy (reference for router + paths)

These aspects have precomputed sentiment in `hotel_aspect_sentiment`. The router needs this
list to distinguish "known aspect → structured path" from "unknown aspect → review-search."

**Facility aspects** (is_facility=1) — have both `has_X` boolean on hotels AND sentiment:
`wifi`, `pool`, `gym`, `sauna`, `restaurant`, `room_service`, `lounge`, `event_space`

**Experiential aspects** (is_facility=0) — sentiment only, no boolean:
`staff`, `cleanliness`, `location`, `room_comfort`, `value`, `noise`, `breakfast`

**Typical sentiment ranges** (for narration calibration — "good" means different things per
aspect): location 0.90, staff 0.82, cleanliness 0.78, breakfast 0.58, room_comfort 0.56,
noise 0.51, value 0.37.

**`event_space` is unreliable** (only 6/1000 detected). Do not expose it as a filterable
amenity. Present in the data but should not be offered to users.

**Semantic note on `has_X` flags:** these now mean "guest reviews evidence this amenity,"
NOT "the hotel definitely has/lacks this." Absence of mention ≠ absence of amenity.

**Important:** `avg_score` is on a **1–10 scale** (Booking.com), not 1–5.

---

## 3. Build phases

### Phase 0 — Data migration + schema update
**Goal:** swap in the new database, update `core/db.py`, verify nothing breaks.

**Tasks:**
1. Replace `hotels.db` with the new version (3 tables).
2. Update `core/db.py`:
   - **Introspect the new schema** — confirm all 3 tables, their columns, and row counts.
   - **Create the FTS5 virtual table** if it doesn't exist:
     ```sql
     CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts USING fts5(
         positive_review, negative_review,
         content='reviews', content_rowid='review_id'
     );
     ```
     Populate it from the `reviews` table. This requires a writable connection for the
     one-time setup; then switch back to read-only for all application queries.
   - **Update the data dictionary** to cover all 3 tables + their columns + relationships.
     This dictionary is fed to the LLM in prompts. Add the `hotel_aspect_sentiment` table
     description and the `reviews` table description. Note the `haversine()` function still
     exists.
   - If existing code references hotels by rowid or lacks `hotel_id`, update to use
     `hotel_id` as the primary key / join column.
3. Run the **existing test suite** — fix any breakage caused by the schema changes (e.g.
   `hotel_id` column added, flag values changed). The parameterized and semantic paths should
   still pass their existing tests with minor adjustments to expected values (since amenity
   flags changed from random to review-derived).

**Acceptance:**
- All 3 tables introspected correctly.
- FTS5 query `SELECT COUNT(*) FROM reviews_fts WHERE reviews_fts MATCH 'pool'` returns a
  positive count.
- Existing test suite passes (with any needed value adjustments).
- Data dictionary covers all 3 tables.

---

### Phase 1 — Extend parameterized path
**Goal:** `search_hotels` can filter/sort by precomputed aspect sentiment.

**Tasks:**
Update `core/search.py` (or wherever `search_hotels` lives):
1. Add an `aspect_filters` parameter: a list of
   `{aspect: str, min_sentiment: float, min_mentions: int}`. When provided, JOIN
   `hotel_aspect_sentiment` and filter:
   `a.aspect = ? AND a.sentiment >= ? AND a.mention_count >= ?`.
   Multiple aspect filters combine with AND (hotel must satisfy all).
2. Add aspect-based sorting: allow `sort_by` to accept an aspect name (e.g.
   `"pool_sentiment"`), which sorts by `hotel_aspect_sentiment.sentiment` for that aspect.
3. **Exclude `event_space`** from the valid amenity filter set (it's unreliable).
4. All new SQL must use `?` parameterized queries.

**Acceptance:**
- New tests pass:
  - Filter: hotels with pool sentiment ≥ 0.7 and ≥ 10 mentions → returns subset.
  - Sort by staff sentiment desc → top result has highest staff sentiment.
  - Combined: city = Paris + has_pool = 1 + pool sentiment ≥ 0.6 → correct intersection.
  - `event_space` rejected if passed as an amenity filter.
- Existing parameterized tests still pass.

---

### Phase 2 — Extend semantic path
**Goal:** analytics/aggregation queries can use aspect sentiment.

**Tasks:**
Update `core/semantic.py` (or wherever the `QuerySpec` compiler lives):
1. Add `aspect` as an optional field on `QuerySpec`. When `metric = "avg_sentiment"` (new
   metric), `aspect` is required — e.g. `{metric: "avg_sentiment", aspect: "staff",
   group_by: "city"}`.
2. The compiler generates a JOIN to `hotel_aspect_sentiment` filtered by `aspect = ?`, then
   applies the aggregate (`AVG(sentiment)`) with the requested GROUP BY.
3. Add `aspect` as a valid `group_by` dimension — e.g. "average sentiment across all
   aspects" → `GROUP BY aspect`.
4. Reject invalid specs (e.g. `avg_sentiment` without `aspect`) with a clear error.

**Acceptance:**
- New tests pass:
  - `avg_sentiment` for `staff` grouped by city → per-city staff sentiment averages.
  - `count` of hotels grouped by `has_pool` → correct split.
  - `avg_sentiment` for `breakfast` where `avg_score >= 9.0` → correct filtered average.
  - Invalid spec (avg_sentiment without aspect) → rejection error.
- Existing semantic tests still pass.

---

### Phase 3 — Review-search path (NEW)
**Goal:** answer questions about novel/unknown aspects by searching review text at runtime.

**Tasks:**
Create `core/review_search.py`:
1. Function signature: `search_reviews(terms: list[str], city: str | None, min_rating:
   float | None, limit: int = 10) -> list[dict]`.
2. Use FTS5 to find reviews matching the terms: `reviews_fts MATCH 'term1 OR term2'`.
3. For each matching review, determine polarity: check if the term appears in
   `positive_review` vs `negative_review` (simple keyword check on each column separately —
   FTS5 finds the review, Python checks which column has the match).
4. Aggregate per hotel: `mention_count`, `pos_count`, `neg_count`,
   `sentiment = pos / (pos + neg)`.
5. JOIN with `hotels` for metadata. Apply optional city/rating filters.
6. Sort by mention_count (default) or sentiment, apply limit.
7. **Exclude** the placeholder strings `"No Positive"` and `"No Negative"` from matching.
8. Return hotel rows + ad-hoc sentiment stats. Mark `confidence = "medium"` in the response.

**Acceptance:**
- Tests pass:
  - "rooftop" returns hotels with pos/neg counts.
  - "helipad" returns empty list (no matches), not an error.
  - City filter restricts results correctly.
  - Placeholders "No Positive" / "No Negative" excluded.
- No LLM involved — this path is fully deterministic.

---

### Phase 4 — Update router
**Goal:** router knows about the new path and the known-aspect taxonomy.

**Tasks:**
Update `core/router.py`:
1. Add `"review_search"` as a routing option.
2. Add the **known aspect taxonomy** (§2 of this plan) to the router's system prompt. The
   router must know which aspects are precomputed so it can distinguish:
   - "hotels with good pools" → known aspect (`pool`) → **parameterized** (filter by
     `pool_sentiment`).
   - "hotels with good rooftop bars" → unknown aspect → **review_search** (search for
     "rooftop bar").
3. Update routing logic / few-shot examples to include:
   - Known-aspect filter queries → parameterized (with aspect_filters).
   - Known-aspect analytics → semantic (with aspect metric).
   - Unknown-aspect queries → review_search (extract search terms).
   - Add ~5 new few-shot routing examples covering these cases.
4. Update the router's structured output to include `search_terms: list[str] | None` for
   the review_search path.
5. If the router returns `review_search`, extract the search terms from the structured
   output and pass them to `search_reviews()`.

**Acceptance:**
- Test with **≥10 new questions** covering:
  - Known-aspect filter: "top hotels with great staff" → parameterized with aspect_filter
  - Known-aspect analytics: "average breakfast rating by city" → semantic
  - Novel-aspect: "hotels with rooftop terrace" → review_search with terms ["rooftop",
    "terrace"]
  - Novel-aspect: "hotels where guests mention pets" → review_search
  - Still-working: existing test questions route correctly as before.
- Full pipeline (router → execute → return rows) works from a script.

---

### Phase 5 — Update narration
**Goal:** narration handles review-search results with appropriate confidence framing.

**Tasks:**
Update `core/narrate.py`:
1. When path = `"review_search"`:
   - Use phrasing like "Based on searching guest reviews for '[terms]'…"
   - Cite mention counts: "N guests mentioned this."
   - Do NOT present ad-hoc results as if they were precomputed scores.
2. When results include aspect-sentiment data (from parameterized/semantic with aspects):
   - Calibrate against the typical ranges in §2. Don't call 0.85 staff sentiment
     "exceptional" when the average is 0.82.
3. Add `confidence` to the response object: `"high"` for parameterized/semantic, `"medium"`
   for review_search, `"guarded"` for fallback.

**Acceptance:**
- Review-search narration includes the lower-confidence framing and mentions search terms.
- Aspect-sentiment narration references actual sentiment values, not vague claims.
- Existing narration tests still pass.

---

### Phase 6 — Update API + frontend
**Goal:** the new features are exposed end-to-end.

**Tasks:**

**API (`api/main.py`):**
1. Update the Pydantic response model to include `confidence: str`.
2. No new endpoints needed — the existing `/query` endpoint handles the new path
   transparently (router picks it).

**Frontend (`frontend/app.py`):**
1. When results include aspect-sentiment data, show it in the results table (e.g. an extra
   column for the relevant aspect's sentiment and mention count).
2. Show the `confidence` level in the transparency panel.
3. When path = `review_search`, visually distinguish the results (e.g. a note saying "Results
   from searching guest reviews" rather than the standard structured-query display).
4. The city dropdown and map should still work as before.

**Acceptance:**
- End-to-end test with 5 question types:
  1. "Top 5 hotels in Paris with good pools" → parameterized + aspect filter
  2. "Average staff sentiment by city" → semantic + aspect metric
  3. "Hotels with nice rooftop bars" → review-search (novel aspect)
  4. "How many hotels have a gym?" → semantic (existing, still works)
  5. "Best value hotels in London" → parameterized + value aspect sort
- Each returns correct results with appropriate confidence labeling.
- Transparency panel shows the path and query for each.

---

### Phase 7 — Update SQL fallback + data dictionary
**Goal:** the fallback tier knows about the new tables.

**Tasks:**
Update `core/fallback.py`:
1. Update the data dictionary / schema prompt to include `reviews` and
   `hotel_aspect_sentiment` tables with their columns, types, relationships, and join keys.
2. Add 2–3 new few-shot examples involving the new tables:
   - "Hotels with the most reviews" → `SELECT h.name, h.total_reviews FROM hotels h ORDER BY
     h.total_reviews DESC LIMIT 10`
   - "Average reviewer score vs hotel avg_score by city" → join `reviews` to `hotels`
   - "Which aspect has the highest average sentiment?" → `SELECT aspect,
     AVG(sentiment) FROM hotel_aspect_sentiment GROUP BY aspect ORDER BY AVG(sentiment) DESC`
3. Update the `sqlglot` validation to recognize the new table and column names as valid.

**Acceptance:**
- Arbitrary questions involving the new tables produce validated SQL + correct rows.
- `sqlglot` still blocks writes and references to non-existent columns.
- Existing fallback tests still pass.

---

## 4. Non-negotiable reminders (carried from the original plan)

These still apply — don't violate them while extending:
- Core logic never imports FastAPI or Streamlit.
- LLM never writes SQL except in the guarded fallback tier.
- All SQL is read-only and parameterized (`?` placeholders).
- Narration is grounded in returned rows only.
- Transparency is mandatory (show what ran).
- Decline honestly when the data can't answer.
- LLM provider stays abstracted via env vars.

---

## 5. Working agreement

- **Run the existing test suite first** (Phase 0) before changing any logic — establish your
  baseline.
- **Extend existing functions** rather than rewriting them. Add parameters, not new code paths
  that duplicate logic.
- Keep the review-search path **deterministic** — the LLM's only new role is routing to it
  and narrating its results.
- Stop at each phase boundary and report.
