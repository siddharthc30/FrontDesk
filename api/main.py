"""
Phase 5 — FastAPI wrapper.

Exposes the pipeline over HTTP. Contains NO business logic — it is a thin
wrapper around core.pipeline.

Endpoints:
  POST /ask         → Server-Sent Events stream (step events + final result)
  POST /ask/sync    → Single JSON response (PipelineResponse)
  POST /api/stt     → Transcribe base64-encoded audio to text (server-side STT)
  GET  /health      → {"status": "ok"}

Run with:
  uvicorn api.main:app --reload
"""

from __future__ import annotations

import base64
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # load .env before any core import touches os.environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Hotel NL Search", version="2.0.0")


@app.on_event("startup")
def _startup():
    """One-time FTS5 index setup on first launch."""
    import os
    from core.db import ensure_fts5
    db_path = os.environ.get("DB_PATH", "data/hotels.db")
    ensure_fts5(db_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / helper models ────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None
    voice: bool = False   # when True, response includes base64 MP3 audio


class TranscribeRequest(BaseModel):
    audio_b64: str          # base64-encoded audio bytes
    content_type: str = "audio/wav"


# ── TTS helper ─────────────────────────────────────────────────────────────────

def _attach_audio(result, answer: str, insights: str) -> object:
    """Call TTS and return a copy of result with audio_b64 set (or unchanged on failure)."""
    from core.tts import synthesize

    tts_text = answer
    if insights:
        tts_text += "  " + insights

    audio_bytes = synthesize(tts_text)
    if audio_bytes:
        return result.model_copy(
            update={"audio_b64": base64.b64encode(audio_bytes).decode()}
        )
    return result


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Keep-alive ping."""
    return {"status": "ok"}


@app.post("/ask")
async def ask_endpoint(req: QuestionRequest):
    """Stream pipeline progress as Server-Sent Events.

    Each event has an `event` field ("step" | "result" | "error") and a `data`
    field containing a JSON-encoded payload.
    """
    from core.pipeline import ask_stream  # imported here → core stays FastAPI-free

    async def event_generator():
        try:
            async for event_type, data in ask_stream(
                req.question, req.user_lat, req.user_lng
            ):
                if event_type == "step":
                    yield {"event": "step", "data": json.dumps(data)}
                elif event_type == "result":
                    if req.voice and not data.declined:
                        data = _attach_audio(data, data.answer, data.insights)
                    yield {"event": "result", "data": data.model_dump_json()}
                elif event_type == "error":
                    yield {"event": "error", "data": json.dumps(data)}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(event_generator())


@app.post("/ask/sync")
async def ask_sync_endpoint(req: QuestionRequest):
    """Non-streaming endpoint — returns a single PipelineResponse JSON object.

    Useful for curl / Swagger UI testing.
    """
    from core.pipeline import ask  # imported here → core stays FastAPI-free

    result = await ask(req.question, req.user_lat, req.user_lng)
    if req.voice and not result.declined:
        result = _attach_audio(result, result.answer, result.insights)
    return result


@app.post("/api/stt")
async def stt_endpoint(req: TranscribeRequest):
    """Server-side speech-to-text transcription.

    Accepts base64-encoded audio and returns the transcript.
    Uses the STT provider configured via STT_PROVIDER env var (default: openai/whisper).
    """
    from core.stt import transcribe

    audio_bytes = base64.b64decode(req.audio_b64)
    text = transcribe(audio_bytes, req.content_type)
    if text is None:
        return {"text": None, "error": "Transcription failed or STT not configured"}
    return {"text": text}
