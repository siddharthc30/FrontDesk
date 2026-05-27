"""
Phase 5 — FastAPI wrapper.

Exposes the pipeline over HTTP. Contains NO business logic — it is a thin
wrapper around core.pipeline.

Endpoints:
  POST /ask       → Server-Sent Events stream (step events + final result)
  POST /ask/sync  → Single JSON response (PipelineResponse)
  GET  /health    → {"status": "ok"}

Run with:
  uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # load .env before any core import touches os.environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="Hotel NL Search", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ──────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    user_lat: Optional[float] = None
    user_lng: Optional[float] = None


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
    return result
