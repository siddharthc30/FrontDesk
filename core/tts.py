"""Phase 8a — TTS provider abstraction.

synthesize(text) -> MP3 bytes | None

Provider selection via TTS_PROVIDER env var (default: elevenlabs).
Returns None if TTS is not configured or fails — callers degrade gracefully to text-only.
Does NOT import FastAPI or Streamlit.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_TTS_PROVIDER: str = os.environ.get("TTS_PROVIDER", "elevenlabs").lower()


def synthesize(text: str) -> Optional[bytes]:
    """Convert text to MP3 audio bytes via the configured TTS provider.

    Returns None if TTS is unavailable, not configured, or fails.
    """
    if not text or not text.strip():
        return None

    if _TTS_PROVIDER == "elevenlabs":
        return _elevenlabs_synthesize(text)
    elif _TTS_PROVIDER == "openai":
        return _openai_tts_synthesize(text)
    else:
        logger.warning("Unknown TTS_PROVIDER=%r — voice disabled", _TTS_PROVIDER)
        return None


def _elevenlabs_synthesize(text: str) -> Optional[bytes]:
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        logger.debug("ELEVENLABS_API_KEY not set — TTS skipped")
        return None

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (default premade)
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

    try:
        import requests as _requests

        resp = _requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("ElevenLabs TTS failed: %s", exc)
        return None


def _openai_tts_synthesize(text: str) -> Optional[bytes]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("OPENAI_API_KEY not set — TTS skipped")
        return None

    voice = os.environ.get("OPENAI_TTS_VOICE", "nova")
    model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=model,
            voice=voice,  # type: ignore[arg-type]
            input=text,
            response_format="mp3",
        )
        return response.content
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI TTS failed: %s", exc)
        return None
