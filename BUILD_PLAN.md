# BUILD_PLAN.md — Hotel Natural-Language Search

**Audience:** Claude Code (autonomous build agent).
**How to use this file:** Work through the phases **in order**. Each phase has a Goal,
Tasks, and Acceptance criteria. Do not start a phase until the previous phase's acceptance
criteria pass. After each phase, stop and report what you did and how you verified it.
**Prime directive:** build the deterministic spine first; wire the LLM in last. Keep a
working, testable slice at every step.

> If a file named `hotel-nl-search-project-context.md` exists in the repo, read it first for
> full rationale. This file is the actionable subset.

---

## 0. Project in one paragraph

A user asks a natural-language question about a local dataset of hotels and gets a **grounded**
answer, short insights, and the matching hotels on a map. The system must answer *any* question
about the data, or decline honestly when the data can't support it. Reliability is achieved with
a **tiered hybrid**: an LLM router sends each question to the most-bounded path that fits, and
only falls back to model-authored SQL as a last resort.

---

## 1. Current state (DO NOT REBUILD)

- A **SQLite database already exists and is populated** with hotel data. This is the source of
  truth. **Do not regenerate, reseed, or overwrite it.**
- Your first job is to *inspect* the real database and build everything against its actual
  schema — not against any assumed schema.

---

## 2. Non-negotiable architectural principles

Respect these throughout. They are the point of the design; violating them defeats it.

1. **Core logic is transport-agnostic.** All business logic lives in a `core/` package made of
   **pure Python functions that never import FastAPI or Streamlit.** The API and the frontend
   depend on `core`; `core` depends on neither.
2. **The LLM never writes SQL except in the guarded fallback tier.** In the parameterized and
   semantic paths, the model only emits structured data (parameters / a query-spec). Your code
   builds the SQL.
3. **All SQL runs read-only and parameterized.** Use a read-only SQLite connection
   (`file:...?mode=ro` URI). Bind values with `?` placeholders — never string-format values
   into SQL.
4. **The fallback tier is validated before execution** with `sqlglot`: confirm a single
   read-only `SELECT`, block any write/DDL (`INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`ATTACH`/
   `PRAGMA`), and verify it references only real tables/columns.
5. **Narration is grounded.** The answer/insights LLM call is fed **only the rows returned by
   the query** — never the raw question-to-data mapping. It must not invent rows or values.
6. **Transparency is mandatory.** Every response includes which path ran and the exact query
   (or a plain-English restatement) that produced the answer.
7. **Decline honestly.** If a question can't be answered from the data (the field doesn't
   exist) or isn't about the hotels, return a clear decline — do not force-fit a query.
8. **LLM provider is abstracted and configured via env vars.** Put all model calls behind one
   thin module (e.g. `core/llm.py`) so the provider/model is swappable. Do not hardcode a
   vendor. Read the API key and model name from environment variables.

---

## 3. Tech stack & scope

- **Language/runtime:** Python (use a virtualenv).
- **Core:** plain Python. **Database:** the existing SQLite file. **SQL safety:** `sqlglot`.
- **API:** FastAPI (thin). **Frontend:** Streamlit (dumb: input + rendering only).
- **Orchestration:** plain Python + direct LLM API calls. **Do NOT use LangChain or LangGraph.**
- **LLM:** a general hosted model with structured-output / function-calling support, behind the
  `core/llm.py` abstraction. Keep calls lightweight.

### Out of scope (do NOT build)
- Review-sentiment / "good pool" quality analysis, vector search, `pgvector`.
- Any write operations on the data.
- Multi-table or "point at any database" generality.
- Migrating off SQLite. Self-consistency / multi-sample voting (leave hooks, don't build).
- Regenerating or modifying the dataset.

---

## 4. Repo structure (create/confirm)

```
core/        # pure logic: db access, paths, router, narration, llm client — no web imports
  db.py          # read-only connection, registers haversine(), schema introspection
  search.py      # parameterized path: search_hotels(params)
  semantic.py    # semantic path: query-spec model + deterministic compiler
  fallback.py    # guarded text-to-SQL: generate -> sqlglot validate -> execute -> retry
  router.py      # LLM router: pick a path or decline
  narrate.py     # grounded answer/insights from returned rows
  llm.py         # provider-agnostic LLM client (env-configured)
  models.py      # shared dataclasses/Pydantic: params, query-spec, response contract
api/         # FastAPI app importing core
frontend/    # Streamlit app calling the API over HTTP
tests/       # pytest, one file per core module
.env.example # documents required env vars (LLM key, model, API URL)
README.md
```

---

## 5. Shared response contract (define once, in `core/models.py`)

Every question, regardless of path, resolves to one response object:

- `answer`: short natural-language answer (str)
- `insights`: 1–3 grounded observation sentences (str)
- `hotels`: list of result rows (may be empty)
- `path`: which path ran — `"parameterized" | "semantic" | "fallback" | "declined"`
- `query_ran`: the SQL or query-spec that produced the result, or null if declined
- `declined`: bool, with a `reason` when true

---

## 6. Build phases

### Phase 0 — Orient & scaffold
**Goal:** working skeleton and ground-truth knowledge of the real data.
**Tasks:** create the repo structure and venv; install deps; build `core/db.py` with a
read-only connection and a registered `haversine(lat1, lon1, lat2, lon2)` SQL function; write a
small introspection routine that prints the actual tables, columns, types, and a few sample
rows; from that, write a short data dictionary (column → meaning) into the repo.
**Acceptance:** `pip install` succeeds; running the introspection prints the real schema +
samples; a unit test confirms `haversine` returns correct distances for known coordinates.

### Phase 1 — Deterministic paths (NO LLM yet)
**Goal:** the reliable core, provable without any model.
**Tasks:** implement `search_hotels(params)` (filters, amenity flags, price range, geo radius
via `haversine`, sort, limit) using parameterized SQL. Implement the semantic path: a
`QuerySpec` model (metric, dimensions/group_by, filters) plus a **deterministic compiler** that
turns a spec into parameterized SQL. Derive the supported metrics/dimensions/filters from the
**actual** columns found in Phase 0.
**Acceptance:** pytest covers both paths with hardcoded inputs; geo radius, amenity filters,
sorting, and at least one aggregation (e.g. count, average, group-by) all return correct
results. No LLM involved.

### Phase 2 — Guarded text-to-SQL fallback
**Goal:** answer arbitrary questions safely.
**Tasks:** build `core/llm.py` (env-configured). In `fallback.py`: prompt the model with the
data dictionary + a handful of few-shot (question → SQL) examples to produce one `SELECT`;
validate it with `sqlglot` per principle #4; execute on the read-only connection; on a database
error, feed the error back for **one** self-correction retry, then give up gracefully. Return
the SQL that actually ran.
**Acceptance:** arbitrary questions yield validated SQL + correct rows; tests prove that
write/DDL statements and references to non-existent columns are **rejected** before execution;
the executed SQL is surfaced in the response.

### Phase 3 — Router
**Goal:** send each question to the right path, or decline.
**Tasks:** in `router.py`, use function-calling/structured output to classify the question and
choose `parameterized | semantic | fallback`, or decline (out-of-scope / data-ceiling). System
prompt must forbid force-fitting an unanswerable question into a path.
**Acceptance:** a written suite of ~15 varied questions (search, aggregation, arbitrary,
out-of-scope, unanswerable-from-data) routes as expected; out-of-scope and data-ceiling
questions decline cleanly. The full pipeline now runs as a plain script.

### Phase 4 — Grounded narration
**Goal:** human-readable answer + insights, grounded in results.
**Tasks:** `narrate.py` takes the returned rows (only) and produces `answer` + `insights`.
Handle the empty-result case honestly.
**Acceptance:** tests confirm narration references only values present in the supplied rows;
empty results produce an honest "no matches" answer, not an invented one.

### Phase 5 — FastAPI wrapper (thin)
**Goal:** expose the pipeline over HTTP.
**Tasks:** Pydantic request (`question`, optional `user_lat`/`user_lng`) and response (the
Section 5 contract). One endpoint: router → execute → narrate. Validation + error handling.
**Acceptance:** endpoint works via the auto-generated Swagger docs; `core` still imports and
runs with FastAPI uninstalled (proves the boundary).

### Phase 6 — Streamlit client (dumb)
**Goal:** the user-facing app.
**Tasks:** text input + a way to supply location for "near me"; call the API; render the results
table, a map with markers from the coordinates, the insights, and **the query/path that ran**.
No business logic in the frontend.
**Acceptance:** end-to-end works locally — a typed question returns a table, map markers,
insights, and the visible query.

### Phase 7 — Deploy prep
**Goal:** ready for zero-cost deployment (FastAPI on a free host, Streamlit on Community Cloud).
**Tasks:** pin requirements; configure a CORS allow-list for the Streamlit origin; document all
env vars in `.env.example`; add a README with run + deploy steps and a note about a keep-alive
ping for free-tier cold starts. Do not commit secrets.
**Acceptance:** the app runs from a clean checkout following only the README; deployment steps
are documented.

---

## 7. Working agreement

- Inspect the real database before writing query logic; derive specifics from it.
- Build and **test deterministic components before** introducing any LLM call.
- Keep LLM calls lightweight and provider-agnostic (zero-cost goal).
- Stop at each phase boundary and report; ask before deviating from Section 2 principles or
  pulling in dependencies not listed in Section 3.
