# Hotel Natural-Language Search — Project Context & Build Plan

**What this document is:** A self-contained context handoff for the project. It captures
the idea, every architectural decision *and the reasoning behind it*, the locked v1 scope,
the build plan, and the known gotchas — so it can be pasted into another chat or shared
without re-explaining anything from scratch.

**Status:** Architecture revised to a **tiered hybrid** (parameterized + semantic + guarded
text-to-SQL fallback). v1 scope and stack locked. Ready to implement (data layer first).
**Last updated:** 2026-05-25

---

## 1. The idea

Build an application where a user asks a question in **natural language** about a stored,
**structured dataset of hotels** and gets back an **exact, grounded answer**, a few short
**insights**, and (where relevant) the matching hotels **plotted on a map**.

**Driving requirement (clarified):** the system must accept *any* question about the stored
hotel data and return a grounded answer — or an honest "I can't answer that from this data."
The query below is one *example* of what users might ask, **not** the boundary of what's
allowed:

> "Give me the top 10 rated hotels near me that have good pool facilities or amenities."

Other questions must also work: "what's the average price in Dallas?", "how many hotels have
a pool?", "which city has the most 5-star hotels?", etc.

---

## 2. Mental model (the reframes that shaped the design)

1. **Structured data → the database does the computation, not vector retrieval.** Filtering,
   sorting, aggregation, and geo math belong in the DB engine. We *do* use text-to-SQL — but
   only as a guarded, last-resort tier inside a hybrid, never as the unconstrained default.
   Vector search stays deferred for the unstructured/qualitative features (reviews; see §8).

2. **The trust layer IS the project — even more so now.** Once you answer *arbitrary*
   questions, a confidently **wrong** answer (valid SQL, wrong logic) becomes the core risk,
   because the user can't tell and may act on it. Making answers trustworthy, auditable, and
   correctable is the real work; the demo is an afternoon.

3. **The example query is a hybrid of three constraint types:** structured ("top 10 rated" →
   sort + limit), geospatial ("near me" → location + radius), and qualitative/fuzzy ("good
   pool" → semantic understanding of reviews; the hard part, deferred — see §8).

4. **Data sets the ceiling — and there are two ceilings:**
   - *Capability ceiling* — the question *shape* isn't built yet (fixable: add a path).
   - *Data ceiling* — the *field* doesn't exist ("friendliest staff"); not fixable by any
     model or path, only by changing the data. Decline honestly here.

---

## 3. Architecture — tiered hybrid (router + three paths)

Two goals in tension: **answer any question** (breadth) and **don't make the LLM a single
point of failure** (reliability). The hybrid resolves this with one principle:

> **Try the most-bounded path that fits; fall back to a less-bounded one only when needed.**
> Raw text-to-SQL is the floor, exercised rarely.

An LLM **router** (via structured output / function-calling) classifies each question and
sends it to one of three paths — or declines:

1. **Parameterized path — `search_hotels(params)`.** Fixed levers: `location`, `radius_km`,
   `min_rating`, `required_amenities`, `price_min`/`max`, `sort_by`, `limit`. Deterministic
   code; most reliable. Handles the common search / filter / sort / geo shape. (This is the
   original "Option B" — LLM fills a form, never writes logic.)

2. **Semantic path — structured query-spec over a defined semantic layer.** You predefine
   metrics (`avg_price`, `count`), groupable dimensions (`city`, `has_pool`), and filterable
   fields, each with SQL *you* wrote. The model emits a structured spec — e.g.
   `{ "metric": "avg_price", "group_by": "city", "where": {"has_pool": true} }` — and a
   **deterministic compiler** turns it into SQL. Composable (any combination of your building
   blocks), so it covers analytical questions (counts, averages, group-by, comparisons)
   **without the model ever writing raw SQL.** This is the sweet spot: Option B's safety with
   most of open text-to-SQL's breadth.

3. **Text-to-SQL fallback — the floor, for genuinely arbitrary questions.** The model writes a
   `SELECT`, wrapped in guardrails (see §4). This is the *only* tier where the model authors
   SQL, and it runs rarely, so the single-point-of-failure surface is minimized.

Or the **router declines** (out of scope / data ceiling).

**Pipeline (each stage fails in isolation, so it's debuggable):**

```
question
  → [LLM router] choose the path, or decline            (the fuzzy routing step)
       ├─ parameterized : fill search_hotels params  → deterministic search
       ├─ semantic      : emit query-spec            → deterministic compiler → SQL
       └─ text-to-SQL   : write SQL → validate (sqlglot) + read-only connection → run
  → [execute on SQLite]  returns real rows               (never hallucinates)
  → [LLM] answer/insights, fed ONLY the returned rows    (grounded, can't invent)
  → [frontend] render table + map + answer + SHOW what ran (transparency)
```

**Why it satisfies both goals:** the fallback guarantees *something* answers any question, while
most traffic hits the bounded paths, so the risky SQL-authoring tier is exercised rarely and
always under guard + transparency.

---

## 4. Reliability — defense in depth (now a core deliverable)

The model is fallible; the goal is that it can never fail **silently**. No single check is
enough — they stack:

- **Shrink what the model must get right.** One clean table, a data dictionary (column
  descriptions), and a handful of *few-shot* (question → SQL/spec) examples in the prompt —
  the highest-leverage, lowest-effort accuracy boost.
- **Validate deterministically (the fallback tier).** `sqlglot` parses the generated SQL:
  confirm a single read-only `SELECT`, block writes (`DROP`/`DELETE`/`UPDATE`/`INSERT`), and
  verify it only references columns that exist (catches hallucinated columns). Back it with a
  **read-only SQLite connection** so a bad query physically can't mutate data.
- **Verify by execution.** If the SQL errors, feed the DB error back to the model for **one**
  self-correction retry; turns a silent failure into a fix or an honest decline.
- **Redundancy (optional).** Self-consistency (generate the query 2–3×, compare results, flag
  disagreement) or a second LLM "verifier" pass — this is the literal single-point-of-failure
  fix. Costs extra tokens; use judiciously against the zero-cost goal.
- **Transparency / human-in-the-loop (cheapest, most powerful for v1).** Always surface the
  query that ran (or a plain-English restatement) so a wrong query fails *visibly*. For v1
  this single mitigation buys more safety than all the others combined.

---

## 5. Tech stack & decisions (with reasoning)

- **Backend: Python + FastAPI.** For modularity and to learn the full AI-tool deploy loop
  (secrets, CORS, cold-start handling) as a transferable skill.

- **Frontend: Streamlit.** Python-native, fast. Kept "dumb": input + rendering only, no logic.

- **Database: SQLite for v1; Postgres as the migration target.** Reasoned from the data: tiny
  volume (scale is a non-factor), uniform/relational structure (NoSQL solves problems we don't
  have), queries are filter + sort + radius. Geo needs no spatial index at this scale — a
  custom **`haversine` SQL function registered on the connection** lets even the SQL paths do
  "near me." v1 data is **read-only seed**, so SQLite ships as a file *in the repo* (no DB
  service, no creds, no cold start; the ephemeral-disk pitfall doesn't apply because we never
  write). Migrate to free managed **Postgres (Supabase/Neon)** when persistent writes, PostGIS,
  or `pgvector` (phase-2 semantic) arrive. Keep data access behind a thin layer so the swap is
  config, not a rewrite.

- **Orchestration: plain Python + direct LLM API calls. NOT LangChain/LangGraph for v1.**
  The pipeline is small; frameworks hide the mechanics you want to learn; prebuilt SQL *agents*
  re-introduce the single-point-of-failure (more model autonomy) and burn extra tokens. Adopt
  **LangGraph later** only if orchestration becomes genuinely graph-shaped (multi-tool routing,
  real retry loops, stateful agents) — build the manual version first, adopt when complexity
  creates a felt need.

- **SQL safety: `sqlglot`.** Validates the fallback tier's generated SQL (read-only `SELECT`,
  no writes, real columns). (Note: earlier deemed unnecessary under pure Option B — it's back
  in scope precisely because a tier now generates SQL.)

- **LLM: a general hosted free-tier model** with structured output / function-calling (for
  routing + query-specs) and decent SQL generation (for the fallback). **Not** a self-hosted
  Hugging Face text-to-SQL model — that breaks zero-cost and is unnecessary at one-table scale.
  Provider still open (§10). Keep calls lightweight to respect the zero-cost goal.

- **Core architectural principle:** core logic = **pure Python functions that never import
  FastAPI or Streamlit**. Both depend on the core; the core depends on neither. Frontend/client
  swaps stay trivial.

- **Deployment (zero cost):** FastAPI on Render free tier; Streamlit on Community Cloud pointed
  at the Render URL. Friction (CORS, cold-start keep-alive) is solvable (see §10).

---

## 6. Data sourcing & schema

- **Decision: OpenStreetMap via the Overpass API for the v1 seed.** Lowest friction (no auth),
  unambiguously storable (ODbL: store/modify/use even commercially with attribution +
  share-alike), and it gives real `name` + coordinates (plus some real `stars`,
  `swimming_pool`, `addr:*` tags where present). Hotels are tagged `tourism=hotel`; query nodes,
  ways, and relations with `out geom` to resolve coordinates. Since v1 synthesizes the
  qualitative fields anyway, OSM's sparse ratings cost nothing.

- **Rejected for v1:**
  - *Google Places* — clean data but its policy forbids storing content (only `place_id` is
    exempt; results must show on a Google Map). Kills any "store my own DB" plan.
  - *Amadeus Self-Service* — legal constraints now **exclude amenities/address/ratings from the
    response**; ratings come only from a separate, sentiment-based Hotel Ratings API; plus OAuth
    friction and a limited test dataset. Its sentiment ratings are a good fit for the *deferred*
    review feature — not for v1.

- **Flow:** Overpass real skeleton (name + coords) → synthesize realistic `rating`, `price`,
  amenity flags → load SQLite. You don't need 1000 hotels; ~50–200 exercises every path. Traps:
  cluster synthetic coordinates around real cities (or "near me" breaks); never validate the
  future review-sentiment feature on synthetic reviews.

- **Schema (one table for v1; a `reviews` table is added only at phase 2).** SQLite has dynamic
  typing and no native boolean, so amenities are `INTEGER` 0/1:

```sql
CREATE TABLE hotels (
    id              INTEGER PRIMARY KEY,
    osm_id          TEXT,          -- e.g. "node/123456" — traceability & dedup
    name            TEXT NOT NULL,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    city            TEXT,          -- from addr:city, for display/filtering/grouping
    rating          REAL,          -- 1.0-5.0 (real stars where present, else synthetic)
    price_per_night INTEGER,       -- representative nightly price (synthetic; NOT live)
    has_pool        INTEGER DEFAULT 0,
    has_spa         INTEGER DEFAULT 0,
    has_wifi        INTEGER DEFAULT 0,
    has_parking     INTEGER DEFAULT 0,
    data_source     TEXT           -- 'osm' or 'synthetic' — honesty/auditability
);
```

  Notes: amenity booleans map one-to-one onto both `required_amenities` (parameterized path)
  and the semantic layer's filterable fields. `data_source` preserves real-vs-synthetic for the
  validation trap. `price` is representative/synthetic (real prices are date-dependent).

*(All third-party API facts above are time-sensitive — re-verify current free-tier terms and
storage rights before relying on them.)*

---

## 7. v1 scope (locked)

- SQLite seed (~50–200 hotels) per the schema above, from Overpass + synthesis.
- **Tiered hybrid:** LLM router → parameterized search / semantic query-spec / guarded
  text-to-SQL fallback.
- Answers **any** question about the data with a grounded answer, or declines honestly at the
  data ceiling.
- LLM does routing + query-specs via structured output; writes SQL **only** in the guarded
  fallback tier.
- Deterministic execution on SQLite (including Haversine geo).
- Grounded narration/insights from the returned rows only.
- **Transparency:** always surface which path/query ran.
- Frontend: text input → results table + map markers + answer/insights + the query shown.

---

## 8. Deferred (post-v1)

- Deep "good pool" quality: sentiment over pool-specific review text (precomputed score or
  `pgvector` semantic search) — needs real reviews to validate.
- Write operations (changing data via natural language).
- Multi-table / generic "point at any database" support.
- Heavy orchestration (LangGraph), self-consistency at scale, and spatial indexing (PostGIS) —
  none needed until complexity or data grows.

---

## 9. Build plan (sequenced as vertical slices)

Always keep a working end-to-end slice; build the deterministic spine before anything fuzzy.

1. **Skeleton + boundaries.** One public repo: `core/` (pure logic), `api/` (FastAPI),
   `frontend/` (Streamlit). Define the request/response JSON contract. venv, deps, secrets.

2. **Data layer.** Overpass pull → normalize to the schema → synthesize gaps → load SQLite.
   Register the `haversine` SQL function. Prove the deterministic queries with hardcoded inputs.
   No LLM, no API, no UI. This spine is where correctness lives.

3. **Bounded paths (no LLM yet).** Build `search_hotels(params)` and the semantic query-spec
   compiler; test both deterministically. This is the reliable core.

4. **Guarded text-to-SQL fallback.** SQL generation + `sqlglot` validation + read-only
   execution + one-shot retry + transparency. Test on arbitrary questions.

5. **Router.** LLM (function-calling) picks a path or declines; test on a question suite. The
   whole product now runs as a plain script — key milestone.

6. **Grounded narration.** Answer/insights from the returned rows only. Test standalone.

7. **Wrap in FastAPI (thin).** Pydantic models, one endpoint (router → execute → narrate),
   returns answer + rows + insights + the query that ran. Swagger docs come free.

8. **Streamlit client (dumb).** Input → call API → render table, map markers (`pydeck`/`folium`,
   or `st.map` for quick), insights, and the shown query. No business logic.

9. **Deploy.** FastAPI on Render free tier; Streamlit on Community Cloud. CORS allow-list,
   keep-alive ping, env vars for LLM key + API URL.

**Incremental option:** for the fastest path to *any-question*, ship step 4 (fallback) +
transparency first, then add step 3 (bounded paths) to harden the common cases. For
reliability-first, build step 3 before step 4.

---

## 10. Gotchas & open decisions

**Gotchas:**
- **Confidently-wrong answers** — the core risk, concentrated in the fallback tier. Mitigate
  with validation + transparency + grounding + (optional) self-consistency.
- **SQL safety** — read-only connection + `sqlglot` (single `SELECT`, no writes, real columns).
- **Geo in SQL** — register a custom `haversine` function and tell the model it exists.
- **Force-fitting out-of-scope into a path** — firm router system prompt to decline; surface
  the query so substitutions are visible.
- **Data ceiling** — decline honestly when the field doesn't exist.
- **Multi-call token cost** (routing, retries, self-consistency) vs the zero-cost goal — keep
  calls lean, use a free-tier model.
- **Others:** Google content can't be stored; SQLite ephemeral disk is fine *only* while
  read-only; CORS must allow the Streamlit origin; Render free cold starts (~15 min idle, 30-50s
  wake) need a keep-alive ping; synthetic-data realism + the review-validation trap.

**Open decisions / next step:**
- Pick the free-tier LLM provider (structured output/function-calling + decent SQL generation).
- Define the semantic layer's metric / dimension / filter set.
- Curate the few-shot (question → SQL/spec) examples for the fallback.
- Finalize the amenity flag list.
- Choose build order: bounded-first (reliability) vs fallback-first (fastest any-question).
- **Immediate next step:** build step 2 — the data layer and schema — and prove the
  deterministic queries with hardcoded inputs before adding any LLM.
