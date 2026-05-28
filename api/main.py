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
import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # load .env before any core import touches os.environ

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("frontdesk.api")

# ── Config from env ────────────────────────────────────────────────────────────

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
IS_PROD = ENVIRONMENT == "production"

APP_TOKEN = os.environ.get("APP_TOKEN", "")  # empty = auth disabled (dev convenience)

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

# Per-IP limits (slowapi syntax: "<count>/<period>")
RATE_LIMIT_ASK = os.environ.get("RATE_LIMIT_ASK", "20/minute")
RATE_LIMIT_STT = os.environ.get("RATE_LIMIT_STT", "10/minute")

# Cap STT payload size to prevent DoS (base64-encoded audio).
MAX_STT_PAYLOAD_BYTES = int(os.environ.get("MAX_STT_PAYLOAD_BYTES", str(5 * 1024 * 1024)))

# Cap TTS input length so a single request can't run up unbounded provider cost.
MAX_TTS_TEXT_CHARS = int(os.environ.get("MAX_TTS_TEXT_CHARS", "2000"))

RATE_LIMIT_TTS = os.environ.get("RATE_LIMIT_TTS", "20/minute")


# ── App + middleware ──────────────────────────────────────────────────────────

_fastapi_kwargs: dict = {"title": "Hotel NL Search", "version": "2.0.0"}
if IS_PROD:
    # Disable Swagger / ReDoc / OpenAPI schema endpoints in production.
    _fastapi_kwargs.update(docs_url=None, redoc_url=None, openapi_url=None)

app = FastAPI(**_fastapi_kwargs)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-App-Token", "Accept"],
)


@app.on_event("startup")
def _startup():
    """One-time FTS5 index setup on first launch."""
    from core.db import ensure_fts5
    db_path = os.environ.get("DB_PATH", "data/hotels.db")
    ensure_fts5(db_path)

    if IS_PROD and not APP_TOKEN:
        logger.warning(
            "ENVIRONMENT=production but APP_TOKEN is empty — auth is DISABLED."
        )


# ── Auth dependency ────────────────────────────────────────────────────────────

def require_app_token(x_app_token: Optional[str] = Header(default=None)) -> None:
    """Reject requests without the shared-secret header when APP_TOKEN is set.

    If APP_TOKEN is empty (development default), auth is skipped.
    """
    if not APP_TOKEN:
        return
    if not x_app_token or x_app_token != APP_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-App-Token.",
        )


# ── Request / helper models ────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None
    user_city: Optional[str] = Field(default=None, max_length=80)
    voice: bool = False   # when True, response includes base64 MP3 audio


class TranscribeRequest(BaseModel):
    audio_b64: str          # base64-encoded audio bytes
    content_type: str = "audio/wav"


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Keep-alive ping."""
    return {"status": "ok"}


@app.post("/ask")
@limiter.limit(RATE_LIMIT_ASK)
async def ask_endpoint(request: Request, req: QuestionRequest, _auth: None = None):
    """Stream pipeline progress as Server-Sent Events.

    Each event has an `event` field ("step" | "result" | "error") and a `data`
    field containing a JSON-encoded payload. Internal exception text is logged
    server-side; clients receive a generic message.
    """
    require_app_token(request.headers.get("X-App-Token"))

    from core.pipeline import ask_stream  # imported here → core stays FastAPI-free

    async def event_generator():
        try:
            async for event_type, data in ask_stream(
                req.question, req.user_lat, req.user_lng, req.user_city
            ):
                if event_type == "step":
                    yield {"event": "step", "data": json.dumps(data)}
                elif event_type == "result":
                    # Audio is fetched via POST /api/tts after the result
                    # arrives — keeps the SSE event small so intermediate
                    # proxies don't drop the large base64 payload.
                    yield {"event": "result", "data": data.model_dump_json()}
                elif event_type == "error":
                    # data["message"] is already sanitised by core/pipeline.py
                    yield {"event": "error", "data": json.dumps(data)}
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled exception in /ask stream")
            yield {
                "event": "error",
                "data": json.dumps({"message": "An unexpected error occurred."}),
            }

    return EventSourceResponse(event_generator())


@app.post("/ask/sync")
@limiter.limit(RATE_LIMIT_ASK)
async def ask_sync_endpoint(request: Request, req: QuestionRequest):
    """Non-streaming endpoint — returns a single PipelineResponse JSON object."""
    require_app_token(request.headers.get("X-App-Token"))

    from core.pipeline import ask  # imported here → core stays FastAPI-free

    try:
        result = await ask(req.question, req.user_lat, req.user_lng, req.user_city)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled exception in /ask/sync")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

    # Voice clients fetch audio separately via POST /api/tts after receiving
    # the result. Keeping audio out of this response avoids large-payload
    # proxy issues on Streamlit Cloud / Render.
    return result


@app.post("/api/stt")
@limiter.limit(RATE_LIMIT_STT)
async def stt_endpoint(request: Request, req: TranscribeRequest):
    """Server-side speech-to-text transcription.

    Accepts base64-encoded audio (size-capped) and returns the transcript.
    """
    require_app_token(request.headers.get("X-App-Token"))

    if len(req.audio_b64) > MAX_STT_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio payload too large.")

    from core.stt import transcribe

    try:
        audio_bytes = base64.b64decode(req.audio_b64, validate=True)
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 audio payload.")

    try:
        text = transcribe(audio_bytes, req.content_type)
    except Exception:  # noqa: BLE001
        logger.exception("STT transcription crashed")
        return {"text": None, "error": "Transcription failed."}

    if text is None:
        return {"text": None, "error": "Transcription failed or STT not configured."}
    return {"text": text}


@app.post("/api/tts")
@limiter.limit(RATE_LIMIT_TTS)
async def tts_endpoint(request: Request, req: TtsRequest):
    """Server-side text-to-speech.

    Side-channel endpoint: voice clients call this AFTER receiving the final
    answer from /ask, instead of inlining audio in the SSE result event.
    Keeps the SSE payload small so intermediate proxies (Streamlit Cloud /
    Render / Cloudflare) don't drop the large base64 audio blob.
    """
    require_app_token(request.headers.get("X-App-Token"))

    if len(req.text) > MAX_TTS_TEXT_CHARS:
        raise HTTPException(status_code=413, detail="Text too long for TTS.")

    from core.tts import synthesize

    try:
        audio_bytes = synthesize(req.text)
    except Exception:  # noqa: BLE001
        logger.exception("TTS synthesis crashed")
        raise HTTPException(status_code=502, detail="TTS provider error.")

    if not audio_bytes:
        # synthesize() returns None when no key / unknown provider / non-fatal
        # provider failure. Be explicit so the client can surface it.
        raise HTTPException(
            status_code=503,
            detail="TTS is not configured or the provider returned no audio.",
        )

    return {"audio_b64": base64.b64encode(audio_bytes).decode()}
