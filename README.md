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

## Voice mode (optional)

Toggle "🎙️ Voice mode" at the top of the Streamlit app to ask questions hands-free.

**How it works:**
1. Tap the mic button and speak your question naturally — include city, filters, everything.
2. The audio is sent to the backend and transcribed via OpenAI Whisper.
3. The transcribed question is shown so you can verify it, then submitted to the pipeline.
4. Results appear as usual (table, map, insights), **plus** the answer is read back aloud.

**The app works fully without any voice keys set** — voice degrades gracefully to text-only if the API keys are missing.

### Enabling voice

1. Get an [ElevenLabs API key](https://elevenlabs.io) (free tier: ~10,000 characters/month, enough for demo use).
2. Find your voice ID at https://elevenlabs.io/voice-lab (or use the default Rachel voice: `21m00Tcm4TlvDq8ikWAM`).
3. Add to `.env`:

```env
TTS_PROVIDER=elevenlabs
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

Speech-to-text (STT) uses OpenAI Whisper — it reuses the `OPENAI_API_KEY` you've already configured. Set `STT_PROVIDER=openai` (the default) and no additional setup is needed.

**Alternative TTS provider (OpenAI):** Set `TTS_PROVIDER=openai` to use OpenAI's TTS instead of ElevenLabs. No extra key needed — it reuses `OPENAI_API_KEY`. Optionally set `OPENAI_TTS_VOICE` (default: `nova`) and `OPENAI_TTS_MODEL` (default: `tts-1`).

> **ElevenLabs free-tier note:** The free plan allows ~10,000 characters/month. A typical answer + insights is ~300–500 characters, so you get roughly 20–30 voice responses per month on the free tier before needing to upgrade.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Keep-alive ping → `{"status": "ok"}` |
| `POST` | `/ask` | SSE stream: step events + final `PipelineResponse` |
| `POST` | `/ask/sync` | Single JSON `PipelineResponse` (good for curl / Swagger) |
| `POST` | `/api/stt` | Transcribe base64-encoded audio → `{"text": "..."}` |

Add `"voice": true` to any `/ask` or `/ask/sync` request body to receive a base64-encoded MP3 in the `audio_b64` response field.

**Example — sync endpoint with voice:**
```bash
curl -s -X POST http://localhost:8000/ask/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "How many hotels are in each city?", "voice": true}' | python -m json.tool
```

**Example — sync endpoint (text only):**
```bash
curl -s -X POST http://localhost:8000/ask/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "How many hotels are in each city?"}' | python -m json.tool
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ (if LLM_PROVIDER=gemini) | Your Google Gemini API key |
| `OPENAI_API_KEY` | ✅ (if LLM_PROVIDER=openai) | Your OpenAI API key |
| `LLM_PROVIDER` | No | `gemini` or `openai` (default: `gemini`) |
| `API_URL` | No | FastAPI base URL seen by Streamlit (default: `http://localhost:8000`) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key — enables tracing if set |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `LANGFUSE_BASE_URL` | No | Langfuse host (default: `https://cloud.langfuse.com`) |
| `TTS_PROVIDER` | No | `elevenlabs` or `openai` (default: `elevenlabs`) |
| `ELEVENLABS_API_KEY` | No | ElevenLabs key — required for voice output when `TTS_PROVIDER=elevenlabs` |
| `ELEVENLABS_VOICE_ID` | No | ElevenLabs voice ID (default: Rachel `21m00Tcm4TlvDq8ikWAM`) |
| `ELEVENLABS_MODEL_ID` | No | ElevenLabs model (default: `eleven_turbo_v2_5`) |
| `OPENAI_TTS_VOICE` | No | OpenAI TTS voice (default: `nova`) — used when `TTS_PROVIDER=openai` |
| `OPENAI_TTS_MODEL` | No | OpenAI TTS model (default: `tts-1`) |
| `STT_PROVIDER` | No | `openai` (default) — server-side speech-to-text using Whisper |

---

## Deployment notes

- **FastAPI** → deploy on [Render](https://render.com) free tier (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`). Free tier sleeps after ~15 min idle; set up a cron job hitting `/health` to keep it warm.
- **Streamlit** → deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), pointed at the Render URL via the `API_URL` env var.
- **CORS** → update `allow_origins` in `api/main.py` to your Streamlit Community Cloud URL before going to production.

---

## Data

Hotel data (1,000 records) covers six European cities with: name, address, city, country, coordinates, average guest review score (1–10 scale), total review count, nightly price, and amenity flags (Wi-Fi, pool, gym, sauna, restaurant, room service, lounge, event space).