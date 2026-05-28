"""Phase 8c — STT provider abstraction (server-side).

transcribe(audio_bytes, content_type) -> text | None

Provider selection via STT_PROVIDER env var (default: openai/whisper).
Returns None if STT is unavailable or fails — callers degrade gracefully.
Does NOT import FastAPI or Streamlit.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_STT_PROVIDER: str = os.environ.get("STT_PROVIDER", "openai").lower()

_CONTENT_TYPE_EXT: dict[str, str] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


def transcribe(audio_bytes: bytes, content_type: str = "audio/wav") -> Optional[str]:
    """Transcribe audio bytes to text via the configured STT provider.

    Returns None if STT is unavailable, not configured, or fails.
    """
    if not audio_bytes:
        return None

    if _STT_PROVIDER == "openai":
        return _openai_transcribe(audio_bytes, content_type)
    else:
        logger.warning("Unknown STT_PROVIDER=%r — transcription skipped", _STT_PROVIDER)
        return None


def _openai_transcribe(audio_bytes: bytes, content_type: str) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("OPENAI_API_KEY not set — STT skipped")
        return None

    base_ct = content_type.split(";")[0].strip().lower()
    ext = _CONTENT_TYPE_EXT.get(base_ct, "wav")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"audio.{ext}", io.BytesIO(audio_bytes), content_type),
        )
        text = transcript.text.strip()
        return text if text else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI STT failed: %s", exc)
        return None
