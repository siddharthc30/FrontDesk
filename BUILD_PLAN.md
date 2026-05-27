# BUILD_PLAN.md — Hotel Natural-Language Search

**Audience:** Claude Code (autonomous build agent).
**How to use this file:** Work through the phases **in order**. Each phase has a Goal,
Tasks, Acceptance criteria, and explicit Anti-drift guardrails. Do not start a phase until
the previous phase's acceptance criteria pass. After each phase, stop and report what you
did and how you verified it.

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

- A **CSV file (`hotels_sample.csv`) already exists** with 1000 hotels. This is the data source.
  **Do not regenerate, reseed, or overwrite it.**
- The CSV has these exact columns: `name, address, city, country, latitude, longitude,
  avg_score, total_reviews`.
- Cities in the data: London (~281), Paris (~292), Barcelona (~131), Milan (~119),
  Vienna (~103), Amsterdam (~74).
- `avg_score` ranges roughly 1.0–10.0 (not 1–5). `total_reviews` is an integer count.
- There are **NO amenity columns** (no `has_pool`, `has_spa`, etc.), **NO price column**, and
  **NO `osm_id`**. The schema sketch in the context doc is aspirational — the real data is
  simpler. Build against reality.
- Your first job in Phase 0 is to load this CSV into SQLite, inspect it, and build everything
  against its actual schema.

---

## 2. Non-negotiable architectural principles

Respect these throughout. They are the point of the design; violating them defeats it.

1. **Core logic is transport-agnostic.** All business logic lives in a `core/` package made of
   **pure Python functions that never import FastAPI or Streamlit.** The API and the frontend
   depend on `core`; `core` depends on neither. To verify: you must be able to
   `import core.search` in a plain Python script with only the core dependencies installed.
2. **The LLM never writes SQL except in the guarded fallback tier.** In the parameterized and
   semantic paths, the model only emits structured data (parameters / a query-spec). Your code
   builds the SQL deterministically.
3. **All SQL runs read-only and parameterized.** Use a read-only SQLite connection
   (`file:...?mode=ro` URI). Bind values with `?` placeholders — never string-format user
   values into SQL. The only exception is column/table names in the semantic compiler (which
   come from your allowlist, not from user input).
4. **The fallback tier is validated before execution** with `sqlglot`: confirm a single
   read-only `SELECT`, block any write/DDL (`INSERT`/`UPDATE`/`DELETE`/`DROP`/`ALTER`/`ATTACH`/
   `PRAGMA`), and verify it references only real tables/columns from the data dictionary.
5. **Narration is grounded.** The answer/insights LLM call is fed **only the rows returned by
   the query** — never the raw question-to-data mapping. It must not invent rows or values.
6. **Transparency is mandatory.** Every response includes which path ran and the exact query
   (or a plain-English restatement) that produced the answer.
7. **Decline honestly.** If a question can't be answered from the data (the field doesn't
   exist, e.g. "which hotels have a pool?") or isn't about the hotels, return a clear decline
   with a reason — do not force-fit a query. Specifically: questions about amenities (pool,
   spa, wifi, parking), price, or review text cannot be answered because those columns do not
   exist in the data.
8. **LLM provider is Gemini, abstracted behind `core/llm.py`.** All model calls go through
   this module. The API key and model name are read from environment variables. See §3A for
   Gemini-specific details.

---

## 3. Tech stack & scope

- **Language/runtime:** Python 3.11+ (use a virtualenv).
- **Core:** plain Python + `pydantic` for models. **Database:** SQLite (loaded from the CSV).
  **SQL safety:** `sqlglot`.
- **API:** FastAPI (thin wrapper). **Frontend:** Streamlit (dumb: input + rendering only).
- **Orchestration:** plain Python + direct Gemini API calls via `google-genai` SDK.
  **Do NOT use LangChain, LangGraph, or any agent framework.**
- **LLM:** Google Gemini 2.5 Flash, via the `google-genai` Python SDK. See §3A.
- **Maps:** `folium` (for Streamlit map rendering).
- **Dependencies (exhaustive — do NOT add others without justification):**
  `google-genai`, `pydantic`, `sqlglot`, `fastapi`, `uvicorn`, `streamlit`, `folium`,
  `streamlit-folium`, `requests`, `python-dotenv`, `pytest`.

### 3A. Gemini 2.5 Flash — specific integration details

**SDK:** Use the `google-genai` Python package (NOT the older `google-generativeai` package).
Install: `pip install google-genai`.

**Client initialization (in `core/llm.py`):**
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
```

**Model string:** `"gemini-2.5-flash-preview-05-20"` — read from env var `GEMINI_MODEL` with
this as the default.

**How to make a basic call:**
```python
response = client.models.generate_content(
    model=model_name,
    contents=user_message,
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        temperature=0.0,
    ),
)
text = response.text  # the model's text reply
```

**How to use function calling (for the router — Phase 3):**

Gemini function calling works by declaring Python-style function schemas as `types.FunctionDeclaration`
objects, wrapping them in a `types.Tool`, and passing them in the config. The model responds with
a `function_call` part when it wants to invoke a function.

```python
# 1. Declare the functions the model can call
search_hotels_fn = types.FunctionDeclaration(
    name="search_hotels",
    description="Search/filter/sort hotels by location, rating, city, etc.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "city": types.Schema(type="STRING", description="Filter by city name"),
            "min_rating": types.Schema(type="NUMBER", description="Minimum avg_score"),
            # ... all SearchParams fields
        },
    ),
)

semantic_query_fn = types.FunctionDeclaration(
    name="semantic_query",
    description="Run an analytical query: count, average, min, max, group-by.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "metric": types.Schema(type="STRING", enum=["count", "avg_score", ...]),
            # ... all QuerySpec fields
        },
    ),
)

text_to_sql_fn = types.FunctionDeclaration(
    name="text_to_sql",
    description="Write a raw SQL SELECT for questions that don't fit the other tools.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "sql": types.Schema(type="STRING", description="A single SELECT statement"),
        },
        required=["sql"],
    ),
)

decline_fn = types.FunctionDeclaration(
    name="decline",
    description="The question cannot be answered from the available hotel data.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "reason": types.Schema(type="STRING", description="Why it can't be answered"),
        },
        required=["reason"],
    ),
)

# 2. Wrap in a Tool and pass to the call
tools = types.Tool(function_declarations=[
    search_hotels_fn, semantic_query_fn, text_to_sql_fn, decline_fn
])

response = client.models.generate_content(
    model=model_name,
    contents=user_question,
    config=types.GenerateContentConfig(
        system_instruction=ROUTER_SYSTEM_PROMPT,
        temperature=0.0,
        tools=[tools],
        # Force the model to call exactly one function:
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY"  # "ANY" = must call a function; "AUTO" = may or may not
            )
        ),
    ),
)

# 3. Extract the function call from the response
part = response.candidates[0].content.parts[0]
if part.function_call:
    fn_name = part.function_call.name   # e.g. "search_hotels"
    fn_args = dict(part.function_call.args)  # e.g. {"city": "Paris", "min_rating": 8.0}
```

**How to use JSON mode (for narration — Phase 4):**

For the narration step, use `response_mime_type` to force JSON output:

```python
response = client.models.generate_content(
    model=model_name,
    contents=narration_prompt,
    config=types.GenerateContentConfig(
        system_instruction="...",
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=types.Schema(
            type="OBJECT",
            properties={
                "answer": types.Schema(type="STRING"),
                "insights": types.Schema(type="STRING"),
            },
            required=["answer", "insights"],
        ),
    ),
)
result = json.loads(response.text)
```

**Environment variables (document in `.env.example`):**
```
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-2.5-flash-preview-05-20
API_URL=http://localhost:8000   # for Streamlit to call FastAPI
```

**Rate limits (free tier):** ~250 requests/day, generous TPM. Keep calls lean:
- Router call: 1 call per question
- Fallback SQL generation: 1 call (+ 1 retry on error = 2 max)
- Narration: 1 call
- Total per question: 2–3 calls typical, 4 max

**Anti-drift:** Do NOT use the `google-generativeai` package (old SDK). Do NOT use
`model.start_chat()` or multi-turn chat mode — each call is independent. Do NOT stream
responses — use simple `generate_content`. Do NOT build a conversation history for the
model — each call gets exactly the context it needs (system prompt + the specific input for
that step).

### Out of scope (do NOT build)
- Review-sentiment / "good pool" quality analysis, vector search, `pgvector`.
- Any write operations on the data.
- Multi-table or "point at any database" generality.
- Migrating off SQLite. Self-consistency / multi-sample voting (leave hooks, don't build).
- Regenerating or modifying the dataset.
- Amenity filtering (the data has no amenity columns).
- Price filtering (the data has no price column).
- Any UI framework other than Streamlit. No React, no HTML artifacts.

---

## 4. Repo structure (create/confirm)

```
hotel-nl-search/
├── core/                    # pure logic — NO FastAPI, NO Streamlit imports
│   ├── __init__.py
│   ├── db.py                # read-only SQLite connection, haversine(), schema introspection
│   ├── search.py            # parameterized path: search_hotels(params)
│   ├── semantic.py          # semantic path: QuerySpec model + deterministic SQL compiler
│   ├── fallback.py          # guarded text-to-SQL: generate → sqlglot validate → execute
│   ├── router.py            # LLM router: classify question → pick path or decline
│   ├── narrate.py           # grounded answer/insights from returned rows
│   ├── llm.py               # Gemini client wrapper (env-configured)
│   ├── models.py            # Pydantic models: SearchParams, QuerySpec, PipelineResponse
│   └── pipeline.py          # orchestrates: router → path execution → narration → response
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app, one POST endpoint, imports core
├── frontend/
│   └── app.py               # Streamlit app, calls API over HTTP, renders results
├── tests/
│   ├── test_db.py           # haversine, connection, schema introspection
│   ├── test_search.py       # parameterized path with hardcoded inputs
│   ├── test_semantic.py     # semantic compiler with hardcoded QuerySpecs
│   ├── test_fallback.py     # sqlglot validation: block writes, bad columns
│   └── test_pipeline.py     # end-to-end with mocked LLM (optional)
├── data/
│   ├── hotels_sample.csv    # the source CSV (DO NOT MODIFY)
│   └── hotels.db            # SQLite DB created from the CSV in Phase 0
├── data_dictionary.md       # column → meaning, generated from real schema in Phase 0
├── .env.example
├── requirements.txt
└── README.md
```

**Anti-drift:**
- Every `.py` file in `core/` must have zero imports from `fastapi`, `streamlit`, `uvicorn`,
  or `starlette`. If you find yourself importing these, stop — you're violating principle #1.
- `api/main.py` imports from `core` and from `fastapi`. That's it.
- `frontend/app.py` imports from `streamlit`, `requests`, `folium`. It never imports `core`.
  It talks to the API over HTTP only.

---

## 5. Shared response contract (define in `core/models.py`)

Every question, regardless of path, resolves to one `PipelineResponse`:

```python
from pydantic import BaseModel
from typing import Optional

class HotelRow(BaseModel):
    """One hotel from query results. Fields match the DB schema exactly."""
    id: int
    name: str
    address: str
    city: str
    country: str
    latitude: float
    longitude: float
    avg_score: float
    total_reviews: int
    distance_km: Optional[float] = None  # only when geo query used

class PipelineResponse(BaseModel):
    answer: str                          # short natural-language answer
    insights: str                        # 1–3 grounded observation sentences
    hotels: list[HotelRow]               # result rows (may be empty)
    path: str                            # "parameterized" | "semantic" | "fallback" | "declined"
    query_ran: Optional[str] = None      # the SQL or query-spec that produced the result
    declined: bool = False
    decline_reason: Optional[str] = None
```

Also define in `models.py`:

```python
class SearchParams(BaseModel):
    """Input to the parameterized path. Every field is optional."""
    city: Optional[str] = None
    country: Optional[str] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    min_reviews: Optional[int] = None
    sort_by: Optional[str] = None        # "avg_score" | "total_reviews" | "distance"
    sort_order: Optional[str] = "desc"   # "asc" | "desc"
    limit: Optional[int] = 10
    user_lat: Optional[float] = None     # for geo queries
    user_lng: Optional[float] = None
    radius_km: Optional[float] = None    # max distance from user

class QuerySpec(BaseModel):
    """Input to the semantic path. Analytical queries."""
    metric: str                          # "count" | "avg_score" | "avg_reviews" | "min_score" | "max_score"
    group_by: Optional[str] = None       # "city" | "country"
    filters: Optional[dict] = None       # e.g. {"city": "Paris", "min_rating": 8.0}
    sort_by: Optional[str] = None        # the metric column, for ordering groups
    sort_order: Optional[str] = "desc"
    limit: Optional[int] = None
```

**Anti-drift:** Do NOT add fields for amenities, price, or any column that doesn't exist in
the database. If you're tempted to add `has_pool`, `price_min`, etc., stop — those columns
do not exist. The data has: `name, address, city, country, latitude, longitude, avg_score,
total_reviews`. That's it.

---

## 6. Build phases

---

### Phase 0 — Orient & scaffold

**Goal:** working skeleton, SQLite DB loaded from CSV, and ground-truth knowledge of the data.

**Tasks:**

1. Create the directory structure from §4.
2. Create a virtualenv. Create `requirements.txt` with the exact packages from §3.
   `pip install -r requirements.txt`.
3. Copy `hotels_sample.csv` into `data/`.
4. Build `core/db.py`:
   - A function `load_csv_to_sqlite(csv_path, db_path)` that reads the CSV and creates a
     SQLite database with a `hotels` table. Add an `id INTEGER PRIMARY KEY AUTOINCREMENT`
     column. Map CSV columns to DB columns exactly as they are (no renaming). Run this once
     to create `data/hotels.db`.
   - A function `get_connection(db_path) -> sqlite3.Connection` that opens a **read-only**
     connection using the URI `file:{db_path}?mode=ro`. This is the only way any code
     accesses the DB.
   - Register a **`haversine(lat1, lon1, lat2, lon2)` custom SQL function** on every
     connection. Implementation: standard Haversine formula, returns distance in km. Use
     `math.radians`, `math.sin`, `math.cos`, `math.sqrt`, `math.atan2`, earth radius =
     6371 km. Register via `connection.create_function("haversine", 4, haversine_fn)`.
   - A function `get_schema_info(conn) -> dict` that introspects the DB: returns table
     names, column names + types, row count, and 3 sample rows per table. Use
     `PRAGMA table_info(hotels)` and `SELECT * FROM hotels LIMIT 3`.
   - A function `get_data_dictionary() -> str` that returns a human-readable string
     describing each column. This string will be injected into LLM prompts later.
     Format it like:
     ```
     Table: hotels (1000 rows)
     Columns:
     - id: INTEGER, auto-incrementing primary key
     - name: TEXT, the hotel's name
     - address: TEXT, full street address
     - city: TEXT, city name (values: London, Paris, Barcelona, Milan, Vienna, Amsterdam)
     - country: TEXT, country name (values: United Kingdom, France, Spain, Italy, Austria, Netherlands)
     - latitude: REAL, geographic latitude
     - longitude: REAL, geographic longitude
     - avg_score: REAL, average guest review score on a 1.0-10.0 scale
     - total_reviews: INTEGER, total number of guest reviews

     Available SQL function:
     - haversine(lat1, lon1, lat2, lon2): returns distance in km between two coordinates

     NOTE: This database does NOT contain: price, amenity flags (pool/spa/wifi/parking),
     star rating, room types, or review text. Questions about these topics cannot be answered.
     ```
5. Write `data_dictionary.md` to the repo root with the above content.
6. Create `.env.example` with the env vars from §3A.

**Acceptance criteria:**
- `data/hotels.db` exists with 1000 rows.
- `python -c "from core.db import get_connection, get_schema_info; ..."` prints the real
  schema and sample rows.
- A unit test (`tests/test_db.py`) confirms:
  - `haversine(51.5074, -0.1278, 48.8566, 2.3522)` ≈ 343 km (London → Paris, ±5 km).
  - `haversine(lat, lon, lat, lon)` = 0.0 for any point.
  - The connection is truly read-only: an `INSERT` raises an `OperationalError`.
  - `get_schema_info` returns the correct column names.

**Anti-drift:**
- Do NOT rename CSV columns. Keep them exactly as they are.
- Do NOT add synthetic columns (no price, no amenities). The DB mirrors the CSV + an `id`.
- Do NOT use pandas to query the DB at runtime. Pandas is only for the one-time CSV→SQLite
  load (or use the `csv` module directly). All runtime queries use `sqlite3`.

---

### Phase 1 — Deterministic paths (NO LLM yet)

**Goal:** the two reliable, deterministic query paths, provable with hardcoded inputs and no
model involved.

---

#### Phase 1A — The Parameterized Path (`core/search.py`)

**What this path does:** Handles the common "find me hotels" pattern — filtering, sorting,
and geo-radius queries. The caller provides a `SearchParams` object (filled by the router
in Phase 3; tested with hardcoded values now). Your code deterministically builds a
parameterized SQL query and executes it. **The LLM never touches SQL here.**

**How it works, step by step:**

1. Receive a `SearchParams` object (defined in `models.py` — see §5).
2. Start with a base query: `SELECT *, haversine(latitude, longitude, ?, ?) AS distance_km
   FROM hotels WHERE 1=1`. (If no user location is provided, omit the haversine select and
   the distance-related clauses.)
3. For each non-None filter in `SearchParams`, append an `AND` clause with a `?` placeholder:
   - `city` → `AND LOWER(city) = LOWER(?)`
   - `country` → `AND LOWER(country) = LOWER(?)`
   - `min_rating` → `AND avg_score >= ?`
   - `max_rating` → `AND avg_score <= ?`
   - `min_reviews` → `AND total_reviews >= ?`
   - `radius_km` (requires `user_lat` + `user_lng`) →
     `AND haversine(latitude, longitude, ?, ?) <= ?`
4. Append `ORDER BY`:
   - If `sort_by` is `"distance"` (requires user location): `ORDER BY distance_km`
   - If `sort_by` is `"avg_score"`: `ORDER BY avg_score`
   - If `sort_by` is `"total_reviews"`: `ORDER BY total_reviews`
   - Default: `ORDER BY avg_score DESC`
   - Append `ASC` or `DESC` per `sort_order`.
5. Append `LIMIT ?` (default 10).
6. Execute with `cursor.execute(sql, params_list)` — all values are bound via `?`.
7. Return the result rows as a list of `HotelRow` objects.

**Function signature:**
```python
def search_hotels(params: SearchParams, conn: sqlite3.Connection) -> list[HotelRow]:
```

**Example — what a call looks like (hardcoded, no LLM):**
```python
params = SearchParams(city="Paris", min_rating=8.5, sort_by="avg_score", limit=5)
results = search_hotels(params, conn)
# → returns top 5 Paris hotels with avg_score >= 8.5, sorted by score descending
```

**Example — geo query:**
```python
params = SearchParams(user_lat=48.8566, user_lng=2.3522, radius_km=50, sort_by="distance", limit=10)
results = search_hotels(params, conn)
# → returns 10 nearest hotels within 50km of central Paris, sorted by distance
```

**Anti-drift:**
- Do NOT let this function accept raw SQL strings. It only accepts `SearchParams`.
- Do NOT add filter fields for columns that don't exist (price, amenities).
- Every user-supplied value goes through `?` parameterization. Column names come from a
  fixed allowlist in your code, never from user input.
- The haversine call is only added when `user_lat` and `user_lng` are both provided.

---

#### Phase 1B — The Semantic Path (`core/semantic.py`)

**What this path does:** Handles analytical questions — counting, averaging, grouping,
comparing. "How many hotels are in London?", "What's the average score by city?", "Which
city has the most hotels?" The caller provides a `QuerySpec` object. Your code
deterministically compiles it into SQL. **The LLM never touches SQL here either.**

**The semantic layer — what the model can ask for (and nothing else):**

Define these as explicit allowlists in the code:

```python
ALLOWED_METRICS = {
    "count":        "COUNT(*)",
    "avg_score":    "ROUND(AVG(avg_score), 2)",
    "avg_reviews":  "ROUND(AVG(total_reviews), 0)",
    "min_score":    "MIN(avg_score)",
    "max_score":    "MAX(avg_score)",
    "min_reviews":  "MIN(total_reviews)",
    "max_reviews":  "MAX(total_reviews)",
    "total_reviews_sum": "SUM(total_reviews)",
}

ALLOWED_GROUP_BY = {"city", "country"}

ALLOWED_FILTERS = {
    "city":          ("LOWER(city) = LOWER(?)",          str),
    "country":       ("LOWER(country) = LOWER(?)",       str),
    "min_rating":    ("avg_score >= ?",                   float),
    "max_rating":    ("avg_score <= ?",                   float),
    "min_reviews":   ("total_reviews >= ?",               int),
}
```

**How the deterministic compiler works, step by step:**

1. Receive a `QuerySpec` (metric, optional group_by, optional filters, optional sort, limit).
2. **Validate** every field against the allowlists above. If any field is not in the
   allowlist, raise a `ValueError` — do NOT try to handle it creatively. This is the safety
   guarantee: the model can only pick from a menu, never write SQL.
3. Build the SELECT clause:
   - If `group_by`: `SELECT {group_by_col}, {ALLOWED_METRICS[metric]} AS value FROM hotels`
   - Else: `SELECT {ALLOWED_METRICS[metric]} AS value FROM hotels`
4. Build WHERE from filters (same pattern as parameterized path — `?` placeholders).
5. If `group_by`: append `GROUP BY {group_by_col}`.
6. Append `ORDER BY`:
   - Default: `ORDER BY value {sort_order}` (so "which city has the highest avg score?"
     gets `ORDER BY value DESC`).
   - If `group_by` is used and no sort specified, default to `ORDER BY value DESC`.
7. Append `LIMIT ?` if specified.
8. Execute and return rows. For aggregations (no group_by), return a single-row result.

**Function signatures:**
```python
def compile_query_spec(spec: QuerySpec) -> tuple[str, list]:
    """Returns (sql_string, param_values). Raises ValueError on invalid fields."""

def execute_semantic_query(spec: QuerySpec, conn: sqlite3.Connection) -> list[dict]:
    """Compiles and executes. Returns list of result dicts."""
```

**Example — count with group-by:**
```python
spec = QuerySpec(metric="count", group_by="city")
sql, params = compile_query_spec(spec)
# sql = "SELECT city, COUNT(*) AS value FROM hotels GROUP BY city ORDER BY value DESC"
# params = []
```

**Example — average score filtered:**
```python
spec = QuerySpec(metric="avg_score", filters={"city": "London"})
sql, params = compile_query_spec(spec)
# sql = "SELECT ROUND(AVG(avg_score), 2) AS value FROM hotels WHERE LOWER(city) = LOWER(?)"
# params = ["London"]
```

**Example — top 3 cities by review count:**
```python
spec = QuerySpec(metric="total_reviews_sum", group_by="city", limit=3)
sql, params = compile_query_spec(spec)
# sql = "SELECT city, SUM(total_reviews) AS value FROM hotels GROUP BY city ORDER BY value DESC LIMIT ?"
# params = [3]
```

**Anti-drift:**
- The compiler ONLY accepts values from the allowlists. No dynamic column construction.
- If a `QuerySpec` asks for `metric="price"` or `group_by="has_pool"`, it's a `ValueError`,
  not a creative workaround. These fields don't exist.
- Column names in the SQL come from your hardcoded allowlists (strings you wrote), not from
  user input or the LLM. This is what makes this path safe — it's a controlled menu.
- Do NOT add an "other" or "custom" metric option. The allowlist is exhaustive.

---

**Phase 1 Acceptance criteria:**
- `tests/test_search.py` covers:
  - Filter by city → only that city's hotels returned.
  - Filter by min_rating → all returned hotels have `avg_score >= threshold`.
  - Geo radius → all returned hotels are within the radius (verify with haversine).
  - Sort by avg_score desc → results are in descending order.
  - Limit → correct number of results.
  - Combined filters (city + min_rating + limit).
- `tests/test_semantic.py` covers:
  - `count` with no group_by → returns single number matching `SELECT COUNT(*) FROM hotels`.
  - `count` group_by `city` → returns 6 rows (one per city), counts sum to 1000.
  - `avg_score` filtered by city → matches manual calculation.
  - Invalid metric name → raises `ValueError`.
  - Invalid group_by name → raises `ValueError`.
- All tests pass with NO LLM calls. Pure deterministic logic.

---

### Phase 2 — Guarded text-to-SQL fallback (`core/fallback.py`)

**Goal:** Answer genuinely arbitrary questions that don't fit the parameterized or semantic
paths — safely, with validation, and with transparency.

**When this path is used:** Only when the router (Phase 3) determines that neither the
parameterized path nor the semantic path can handle the question. Examples: "Show me hotels
where the name contains 'Grand'", "Which hotel has the most reviews?", "List hotels in
Paris or Barcelona with a score above 9", "What's the score difference between London and
Paris hotels?" These are valid questions the data CAN answer, but they don't fit the rigid
`SearchParams` or `QuerySpec` shapes.

**How it works, step by step:**

1. **Build the prompt.** Construct a prompt for Gemini that includes:
   - The data dictionary (from `core/db.py`'s `get_data_dictionary()`).
   - 5–8 few-shot examples of (question → correct SQL). These examples are hardcoded
     strings in `fallback.py`, NOT generated. They teach the model the table name, column
     names, and the haversine function. Examples to include:

     ```
     Q: "Which hotel has the highest review score?"
     SQL: SELECT * FROM hotels ORDER BY avg_score DESC LIMIT 1

     Q: "How many hotels are in Paris?"
     SQL: SELECT COUNT(*) AS count FROM hotels WHERE LOWER(city) = LOWER('Paris')

     Q: "Show me hotels within 10km of latitude 51.5, longitude -0.12"
     SQL: SELECT *, haversine(latitude, longitude, 51.5, -0.12) AS distance_km FROM hotels WHERE haversine(latitude, longitude, 51.5, -0.12) <= 10 ORDER BY distance_km ASC

     Q: "What is the average score of hotels in each country?"
     SQL: SELECT country, ROUND(AVG(avg_score), 2) AS avg_score FROM hotels GROUP BY country ORDER BY avg_score DESC

     Q: "List hotels whose name contains 'Palace'"
     SQL: SELECT * FROM hotels WHERE name LIKE '%Palace%' ORDER BY avg_score DESC
     ```

   - An explicit instruction: "Write a single SELECT statement against the `hotels` table.
     Use only the columns listed above. Do not use INSERT, UPDATE, DELETE, DROP, ALTER,
     ATTACH, or PRAGMA. Return only the SQL, no explanation."
   - The user's question.

2. **Call Gemini** via `core/llm.py` to generate the SQL. Use `temperature=0.0` for
   determinism. Extract the SQL string from the response (strip markdown fences if present).

3. **Validate with `sqlglot`** (before execution):
   ```python
   import sqlglot

   def validate_sql(sql: str, allowed_tables: set, allowed_columns: set) -> tuple[bool, str]:
       """Returns (is_valid, error_message)."""
       try:
           parsed = sqlglot.parse(sql, dialect="sqlite")
       except sqlglot.errors.ParseError as e:
           return False, f"Parse error: {e}"

       # Must be exactly one statement
       if len(parsed) != 1:
           return False, "Must be exactly one SQL statement"

       stmt = parsed[0]

       # Must be a SELECT
       if not isinstance(stmt, sqlglot.exp.Select):
           return False, "Only SELECT statements are allowed"

       # Block write/DDL keywords anywhere in the tree
       BLOCKED_TYPES = (
           sqlglot.exp.Insert, sqlglot.exp.Update, sqlglot.exp.Delete,
           sqlglot.exp.Drop, sqlglot.exp.Create, sqlglot.exp.Alter,
       )
       for node in stmt.walk():
           if isinstance(node, BLOCKED_TYPES):
               return False, f"Blocked statement type: {type(node).__name__}"

       # Check all referenced table names
       for table in stmt.find_all(sqlglot.exp.Table):
           if table.name.lower() not in {t.lower() for t in allowed_tables}:
               return False, f"Unknown table: {table.name}"

       # Check all referenced column names
       for col in stmt.find_all(sqlglot.exp.Column):
           if col.name.lower() not in {c.lower() for c in allowed_columns}:
               return False, f"Unknown column: {col.name}"

       return True, ""
   ```

4. **If validation fails:** return the error as a decline. Do NOT retry with a different
   prompt. Do NOT execute invalid SQL.

5. **Execute on the read-only connection.** Wrap in try/except.

6. **If execution fails (SQLite error):** do **one** self-correction retry:
   - Send a new prompt to Gemini: "The following SQL failed with this error: {error}.
     The original question was: {question}. Please write a corrected SELECT statement."
   - Validate the new SQL with `sqlglot` again.
   - If validation passes, execute. If it fails again, give up and return a decline with
     the error message.

7. **Return** the result rows AND the SQL that actually ran (for transparency).

**Function signatures:**
```python
def validate_generated_sql(sql: str) -> tuple[bool, str]:
    """Validate SQL with sqlglot. Returns (is_valid, error_reason)."""

async def execute_fallback(question: str, conn: sqlite3.Connection) -> tuple[list[dict], str]:
    """Generate SQL, validate, execute. Returns (rows, sql_that_ran).
    Raises FallbackError on unrecoverable failure."""
```

**Anti-drift:**
- The few-shot examples are hardcoded strings, NOT generated by the LLM.
- `sqlglot` validation runs BEFORE execution. Never execute unvalidated SQL.
- The retry loop runs at most ONCE. Do not build a multi-retry loop.
- Do NOT catch the `sqlglot` validation failure and try to "fix" the SQL yourself. If
  `sqlglot` says no, the answer is no.
- The only SQL that runs is SQL that passed `validate_generated_sql`. Period.
- Do NOT import or use `pandas` to run queries. Use `sqlite3` directly.
- Strip markdown code fences (` ```sql ` / ` ``` `) from the model's response before
  passing to `sqlglot`. Models often wrap SQL in fences.

**Acceptance criteria:**
- A test generates SQL for "Which hotel has the most reviews?" → valid SQL → correct result.
- A test with a hardcoded `DROP TABLE hotels` string → `validate_generated_sql` returns
  `(False, ...)`.
- A test with a hardcoded `SELECT price FROM hotels` → rejected (unknown column `price`).
- A test with `SELECT * FROM users` → rejected (unknown table `users`).
- A test with `INSERT INTO hotels VALUES (...)` → rejected (not a SELECT).
- The SQL that actually ran is always returned alongside the results.

---

### Phase 3 — Router (`core/router.py`)

**Goal:** Given a natural-language question, use Gemini's function calling to classify it
and route it to the correct path — or decline.

**How it works:**

1. Define four Gemini function declarations (as shown in §3A):
   - `search_hotels` — for filter/sort/geo queries. Parameters mirror `SearchParams`.
   - `semantic_query` — for analytical queries. Parameters mirror `QuerySpec`.
   - `text_to_sql` — for arbitrary questions. Parameter: `sql` (the raw SELECT).
   - `decline` — for unanswerable questions. Parameter: `reason`.

2. Write a detailed **router system prompt**. This is the highest-leverage prompt in the
   whole system. It must include:

   ```
   You are a query router for a hotel database. Given a user's question, you must call
   exactly ONE of the available functions.

   DATABASE SCHEMA:
   {data_dictionary}

   ROUTING RULES (follow in order):

   1. If the question asks to FIND, LIST, SHOW, or SEARCH for specific hotels (with optional
      filters like city, rating, location) → call `search_hotels` with the appropriate
      parameters.
      Examples: "top 5 hotels in Paris", "hotels near me with score above 8",
      "best reviewed hotels in London"

   2. If the question asks for COUNTS, AVERAGES, COMPARISONS, RANKINGS OF GROUPS, or other
      ANALYTICAL/AGGREGATE answers → call `semantic_query` with the appropriate metric,
      group_by, and filters.
      Examples: "how many hotels in each city?", "average score in Barcelona",
      "which city has the highest rated hotels?", "total reviews across all hotels"

   3. If the question is answerable from the data but doesn't fit the above patterns
      (complex conditions, name searches, unusual comparisons) → call `text_to_sql` with
      a valid SELECT statement.
      Examples: "hotels with 'Grand' in the name", "score difference between Paris and
      London", "hotels with more than 2000 reviews"

   4. If the question CANNOT be answered from this data → call `decline` with a clear reason.
      MUST decline for: questions about price, cost, amenities (pool, spa, wifi, parking,
      gym), room types, availability, booking, star classification, review text/sentiment,
      photos, or any field not in the schema.
      Examples: "cheapest hotel in Paris" → decline (no price data),
      "hotels with a pool" → decline (no amenity data),
      "what do guests say about Hotel X?" → decline (no review text)

   IMPORTANT:
   - NEVER force-fit a question. If in doubt, decline with a reason rather than guess.
   - The database has ONLY these columns: name, address, city, country, latitude, longitude,
     avg_score, total_reviews.
   - The `avg_score` is on a 1-10 scale, not 1-5.
   - For "near me" queries, use the user's provided coordinates with the haversine function.
   - Cities in the data: London, Paris, Barcelona, Milan, Vienna, Amsterdam.
   ```

3. Call Gemini with `tool_config=FunctionCallingConfig(mode="ANY")` to force exactly one
   function call.

4. Parse the response:
   - Extract `function_call.name` and `function_call.args`.
   - Map to the corresponding path:
     - `"search_hotels"` → build `SearchParams` from args → call `search.search_hotels()`
     - `"semantic_query"` → build `QuerySpec` from args → call `semantic.execute_semantic_query()`
     - `"text_to_sql"` → extract `sql` arg → pass to `fallback.execute_fallback()` (BUT
       still validate with sqlglot — the router's SQL suggestion is untrusted)
     - `"decline"` → return a decline response with the reason

**The `text_to_sql` routing detail:** When the router calls `text_to_sql`, it provides
a SQL string in the args. However, you should NOT blindly trust this SQL. Instead, pass
the **original question** to `fallback.execute_fallback()`, which will generate its own
SQL with the full few-shot prompt, validate it, and execute it. The router's job is only
to decide which PATH to take, not to write the SQL. Alternatively, you CAN use the
router-provided SQL but MUST still validate it with `sqlglot` before execution.

**Function signature:**
```python
async def route_question(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> tuple[str, SearchParams | QuerySpec | str | None]:
    """Returns (path_name, path_input).
    path_name is "parameterized" | "semantic" | "fallback" | "declined".
    path_input is the corresponding params/spec/sql/decline_reason."""
```

**Test suite (~15 questions) — include these exact questions in `tests/test_router.py`:**

| Question | Expected path | Why |
|---|---|---|
| "Show me the top 5 hotels in Paris" | parameterized | filter + sort + limit |
| "Hotels near latitude 51.5, longitude -0.1" | parameterized | geo query |
| "Best hotels in London with score above 9" | parameterized | filter + sort |
| "How many hotels are in each city?" | semantic | count + group_by |
| "What's the average score in Barcelona?" | semantic | avg + filter |
| "Which city has the most hotels?" | semantic | count + group_by + sort |
| "Total reviews across all hotels" | semantic | sum, no group |
| "Hotels with 'Grand' in the name" | fallback | LIKE query |
| "Which hotel has the most reviews?" | fallback | ORDER BY + LIMIT 1 |
| "Compare London and Paris average scores" | fallback | multi-group comparison |
| "What's the cheapest hotel in Paris?" | declined | no price data |
| "Which hotels have a pool?" | declined | no amenity data |
| "What do guests say about this hotel?" | declined | no review text |
| "What's the weather like in London?" | declined | not about hotel data |
| "Book me a room at the Ritz" | declined | not a data query |

**Anti-drift:**
- The router makes ONE Gemini call per question. Not two, not three. One.
- Use `mode="ANY"` in the tool config to force a function call (not `mode="AUTO"`).
- Do NOT let the router "chain" calls or make follow-up calls. It classifies once.
- The router prompt must explicitly list the columns that DON'T exist (price, amenities)
  to prevent force-fitting.
- Do NOT build a fallback-within-the-router. If the function call parsing fails, return
  a decline, not a retry.
- For the `text_to_sql` path: always validate the SQL with sqlglot before execution,
  regardless of whether it came from the router or from the dedicated fallback prompt.

**Acceptance criteria:**
- The 15-question test suite routes correctly (at least 12/15 — some borderline cases
  between semantic and fallback are acceptable, as long as they produce correct results).
- All "declined" questions produce a decline with a sensible reason.
- The full pipeline now works as a script: question → router → path execution → rows.
  (Narration is not wired yet — that's Phase 4.)

---

### Phase 4 — Grounded narration (`core/narrate.py`)

**Goal:** Turn raw query results into a human-readable answer and 1–3 insight sentences,
grounded entirely in the returned data.

**How it works:**

1. Receive: the original question, the result rows (as dicts/`HotelRow` list), the path
   that ran, and the SQL/spec that ran.
2. Build a prompt for Gemini:
   ```
   You are a helpful assistant that answers questions about hotels based ONLY on the
   data provided below. Do not invent, guess, or add any information not present in
   the results.

   USER QUESTION: {question}

   QUERY RESULTS ({len(rows)} rows):
   {formatted_rows}

   Respond in JSON with exactly two fields:
   - "answer": A direct, concise answer to the user's question (1-3 sentences).
   - "insights": 1-3 additional observations from the data that might interest the user.
     Each insight must reference specific values from the results.

   If no results were returned, say so honestly. Do not make up hotels or scores.
   ```
3. Call Gemini with `response_mime_type="application/json"` and the schema from §3A.
4. Parse the JSON response. If parsing fails, return a generic "I found {n} results"
   answer rather than crashing.

**How to format rows for the prompt:**
- For ≤ 20 rows: include all rows as a simple table or JSON list.
- For > 20 rows: include the first 15 rows and a note "... and {n} more rows".
  The narration doesn't need to see every row; it needs enough to ground its statements.
- For single-value results (aggregations): format as "Result: {value}".

**Function signature:**
```python
async def narrate_results(
    question: str,
    rows: list[dict],
    path: str,
    query_ran: str | None,
) -> tuple[str, str]:
    """Returns (answer, insights)."""
```

**Anti-drift:**
- The narration prompt receives ONLY the result rows. It does NOT receive the raw question-
  to-SQL mapping, the database schema, or any data beyond what the query returned.
- Do NOT pass the full database or schema to the narration prompt. Only the rows.
- If the narration model returns text mentioning hotels, scores, or counts not in the
  result rows, that's a grounding violation — but we can't perfectly detect it. The prompt
  design (rows-only input) is the mitigation.
- Do NOT skip narration and return raw rows as the answer. The narration step is mandatory.
- Handle empty results gracefully: "No hotels matched your criteria" is a valid answer.

**Acceptance criteria:**
- Given 5 rows of Paris hotels, the narration mentions Paris and specific scores from those
  5 rows.
- Given 0 rows, the narration says no results were found (not "here are some hotels").
- Given a single aggregation value (e.g., count=281), the narration reports that number.
- The JSON parsing is robust to minor model output variations.

---

### Phase 4.5 — Pipeline orchestrator (`core/pipeline.py`)

**Goal:** Wire phases 1–4 into a single function that takes a question and returns a
`PipelineResponse`.

**How it works:**
```python
async def ask(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
    db_path: str = "data/hotels.db",
) -> PipelineResponse:
    conn = get_connection(db_path)
    try:
        # 1. Route
        path, path_input = await route_question(question, user_lat, user_lng)

        # 2. Execute the chosen path
        if path == "declined":
            return PipelineResponse(
                answer=f"I can't answer that: {path_input}",
                insights="",
                hotels=[],
                path="declined",
                declined=True,
                decline_reason=path_input,
            )
        elif path == "parameterized":
            rows = search_hotels(path_input, conn)
            query_ran = str(path_input)  # show the params
        elif path == "semantic":
            sql, params = compile_query_spec(path_input)
            rows = execute_semantic_query(path_input, conn)
            query_ran = sql
        elif path == "fallback":
            rows, query_ran = await execute_fallback(question, conn)

        # 3. Narrate
        answer, insights = await narrate_results(question, rows, path, query_ran)

        # 4. Build response
        return PipelineResponse(
            answer=answer,
            insights=insights,
            hotels=[HotelRow(**r) for r in rows] if path == "parameterized" else [],
            path=path,
            query_ran=query_ran,
        )
    finally:
        conn.close()
```

**Anti-drift:**
- This is the ONLY place that orchestrates the full flow. `api/main.py` calls this.
  `frontend/app.py` calls the API. Nobody else orchestrates.
- Error handling: if any step raises an unexpected exception, catch it at this level and
  return a decline response with the error message — never let a raw exception reach the
  user.

---

### Phase 5 — FastAPI wrapper (`api/main.py`)

**Goal:** Expose the pipeline over HTTP with one POST endpoint.

**Tasks:**

1. Create `api/main.py` with a FastAPI app.
2. Define Pydantic request/response models that mirror the core models:
   ```python
   class QuestionRequest(BaseModel):
       question: str
       user_lat: Optional[float] = None
       user_lng: Optional[float] = None

   # Response uses PipelineResponse from core/models.py (or a mirror of it)
   ```
3. One endpoint: `POST /ask`
   - Receives `QuestionRequest`.
   - Calls `core.pipeline.ask(question, user_lat, user_lng)`.
   - Returns the `PipelineResponse` as JSON.
4. Add CORS middleware:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # tighten for production
       allow_methods=["POST"],
       allow_headers=["*"],
   )
   ```
5. Add a `GET /health` endpoint that returns `{"status": "ok"}` (for keep-alive pings).

**Anti-drift:**
- `api/main.py` contains NO business logic. It is a thin HTTP wrapper around
  `core.pipeline.ask()`.
- Do NOT add multiple endpoints. One `POST /ask` and one `GET /health`. That's it.
- Do NOT add authentication, rate limiting, or caching. Not in scope.
- Do NOT add WebSocket support, streaming, or background tasks.
- The FastAPI app must work with: `uvicorn api.main:app --reload`.

**Acceptance criteria:**
- `POST /ask` with `{"question": "How many hotels are in Paris?"}` returns a valid
  `PipelineResponse` JSON.
- `GET /health` returns `{"status": "ok"}`.
- `core/` can still be imported and used without FastAPI installed (test by temporarily
  uninstalling fastapi and running `python -c "from core.pipeline import ask"`).

---

### Phase 6 — Streamlit client (`frontend/app.py`)

**Goal:** A simple, clean UI that takes a question, calls the API, and renders the results.

**Layout:**

1. **Title and description** at the top: "Hotel Search" or similar.
2. **Text input** for the question.
3. **Optional location inputs:** two number inputs (latitude, longitude) for "near me"
   queries, with a note explaining they're optional.
4. **Submit button.**
5. **Results area** (shown after submit):
   - **Answer** — displayed prominently.
   - **Insights** — displayed below the answer.
   - **Transparency box** — an expander or info box showing:
     - Which path ran (parameterized / semantic / fallback / declined).
     - The SQL or query-spec that ran.
   - **Results table** — if hotels were returned, show them in `st.dataframe`.
   - **Map** — if hotels with coordinates were returned, show them on a `folium` map with
     markers. Each marker popup shows the hotel name and score.
     Use `streamlit-folium` to render the map:
     ```python
     import folium
     from streamlit_folium import st_folium

     m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
     for hotel in hotels:
         folium.Marker(
             [hotel.latitude, hotel.longitude],
             popup=f"{hotel.name} ({hotel.avg_score})",
         ).add_to(m)
     st_folium(m, width=700, height=500)
     ```

**How it calls the API:**
```python
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

response = requests.post(
    f"{API_URL}/ask",
    json={
        "question": question,
        "user_lat": user_lat if user_lat else None,
        "user_lng": user_lng if user_lng else None,
    },
)
data = response.json()
```

**Anti-drift:**
- The frontend does NOT import `core`. It calls the API over HTTP. This is the boundary.
- No business logic in the frontend. No SQL, no routing, no LLM calls.
- Do NOT use `st.map()` — it's too limited. Use `folium` + `streamlit-folium`.
- Do NOT build a chat interface with message history. It's a single question → answer flow.
- Handle API errors gracefully: if the API is down, show a user-friendly error message,
  not a raw traceback.
- Run with: `streamlit run frontend/app.py`.

**Acceptance criteria:**
- Typing "Show me the top 5 hotels in Paris" and clicking submit shows: an answer about
  Paris hotels, insights, a table of 5 hotels, map markers in Paris, and the query that ran.
- Typing "What's the cheapest hotel?" shows a decline with a reason about missing price data.
- If the API is not running, a clean error message appears (not a crash).

---

### Phase 7 — Deploy prep

**Goal:** Ready for zero-cost deployment.

**Tasks:**

1. **Pin all dependencies** in `requirements.txt` with exact versions.
2. Create separate requirements files if needed:
   - `requirements-api.txt` — core + fastapi + uvicorn
   - `requirements-frontend.txt` — streamlit + folium + streamlit-folium + requests
3. **`.env.example`** with all required env vars:
   ```
   GEMINI_API_KEY=your-gemini-api-key
   GEMINI_MODEL=gemini-2.5-flash-preview-05-20
   API_URL=http://localhost:8000
   ```
4. **README.md** with:
   - Project description (one paragraph).
   - Prerequisites (Python 3.11+, Gemini API key).
   - Setup steps (clone, venv, install, copy .env, run).
   - How to run locally: `uvicorn api.main:app` in one terminal,
     `streamlit run frontend/app.py` in another.
   - Deployment notes:
     - FastAPI on Render free tier (or similar). Note: free tier sleeps after ~15 min idle,
       wakes in 30-50s. Set up a keep-alive ping (e.g., cron job hitting `/health`).
     - Streamlit on Community Cloud, pointed at the Render URL via `API_URL` env var.
     - CORS: update `allow_origins` to the Streamlit Community Cloud URL.
   - Architecture overview (the three paths, why).
   - Data attribution: "Hotel data sourced from [source]. Scores and reviews are real."

**Anti-drift:**
- Do NOT commit `.env` or any API keys to the repo.
- Do NOT add Docker, docker-compose, or Kubernetes configs. Not in scope.
- Do NOT add CI/CD pipelines. Not in scope.
- The README should enable someone to go from clone to running app in under 5 minutes.

**Acceptance criteria:**
- A fresh `git clone` → follow README → app runs locally end-to-end.
- No secrets in the repo.
- All env vars documented.

---

## 7. Working agreement

- **Inspect the real database/CSV before writing query logic;** derive specifics from it.
- **Build and test deterministic components before** introducing any LLM call.
- Keep LLM calls lightweight (zero-cost goal: ~2-3 Gemini calls per question).
- **Stop at each phase boundary** and report what you built and how you verified it.
- Ask before deviating from Section 2 principles or adding dependencies not listed in §3.
- **If something doesn't work, say so.** Don't silently skip a failing test or acceptance
  criterion.
- **Do NOT refactor working code** from previous phases unless the current phase requires it.
- **Do NOT add features not described in this plan.** No "nice to haves," no "while I'm
  here" additions. The scope is locked.
