# FrontDesk — Hotel Natural Language Search

Ask any natural-language question about 1,000 hotels across six European cities and get a grounded answer, key insights, and an interactive map.

**Cities covered:** London · Paris · Barcelona · Milan · Vienna · Amsterdam

---

## How it works

Every question is routed through a tiered hybrid pipeline:

| Path | Used for | Example |
|---|---|---|
| **Parameterized** | Filter / sort / geo queries | "Top 5 Paris hotels with score > 8" |
| **Semantic** | Counts, averages, group-by analytics | "How many hotels are in each city?" |
| **Fallback** | Arbitrary SQL (LLM-generated, sqlglot-validated) | "Hotels with 'Grand' in the name" |
| **Declined** | Questions the data can't answer | "What's the weather in London?" |

The LLM writes SQL only in the fallback path, and only after sqlglot validates it as a safe, read-only SELECT. Narration is grounded entirely in the returned rows — the model never invents data.

---

## Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier: ~250 req/day)
- (Optional) A [Langfuse](https://cloud.langfuse.com) project for observability tracing

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/siddharthc30/FrontDesk.git
cd FrontDesk

# 2. Create a virtualenv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (required)
# Optionally set LANGFUSE_* keys for tracing

# 4. The database is already committed at data/hotels.db (1,000 hotels)
#    If you need to rebuild it from the CSV:
python -c "from core.db import load_csv_to_sqlite; load_csv_to_sqlite('data/hotels_synthetic.csv', 'data/hotels.db')"
```

---

## Running locally

You need **two terminals** running simultaneously:

```bash
# Terminal 1 — FastAPI backend
source .venv/bin/activate
uvicorn api.main:app --reload
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs

# Terminal 2 — Streamlit frontend
source .venv/bin/activate
streamlit run frontend/app.py
# → http://localhost:8501
```

Open **http://localhost:8501** in your browser and start asking questions.

---

## Interactive terminal test (no UI needed)

```bash
source .venv/bin/activate
python test_interactive.py

# For geo queries (e.g. "hotels near me"):
USER_LAT=51.5074 USER_LNG=-0.1278 python test_interactive.py
```

---

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Keep-alive ping → `{"status": "ok"}` |
| `POST` | `/ask` | SSE stream: step events + final `PipelineResponse` |
| `POST` | `/ask/sync` | Single JSON `PipelineResponse` (good for curl / Swagger) |

**Example — sync endpoint:**
```bash
curl -s -X POST http://localhost:8000/ask/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "How many hotels are in each city?"}' | python -m json.tool
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Your Google Gemini API key |
| `GEMINI_MODEL` | No | Model string (default: `gemini-2.5-flash-preview-05-20`) |
| `API_URL` | No | FastAPI base URL seen by Streamlit (default: `http://localhost:8000`) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key — enables tracing if set |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `LANGFUSE_BASE_URL` | No | Langfuse host (default: `https://cloud.langfuse.com`) |

---

## Deployment notes

- **FastAPI** → deploy on [Render](https://render.com) free tier (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`). Free tier sleeps after ~15 min idle; set up a cron job hitting `/health` to keep it warm.
- **Streamlit** → deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), pointed at the Render URL via the `API_URL` env var.
- **CORS** → update `allow_origins` in `api/main.py` to your Streamlit Community Cloud URL before going to production.

---

## Data

Hotel data (1,000 records) covers six European cities with: name, address, city, country, coordinates, average guest review score (1–10 scale), total review count, nightly price, and amenity flags (Wi-Fi, pool, gym, sauna, restaurant, room service, lounge, event space).