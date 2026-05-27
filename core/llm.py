"""Gemini client wrapper. All LLM calls in the pipeline go through this module.

Langfuse observability is wired here via OpenTelemetry auto-instrumentation.
If LANGFUSE_PUBLIC_KEY is not set the app works normally without any tracing.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

# ── Optional Langfuse + google-genai auto-instrumentation ─────────────────────
# Must run BEFORE any genai.Client() calls so the OTel patch is in place.
if os.environ.get("LANGFUSE_PUBLIC_KEY"):
    try:
        from langfuse import get_client as _get_langfuse_client
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        _langfuse = _get_langfuse_client()
        GoogleGenAIInstrumentor().instrument()
    except Exception as _lf_err:  # noqa: BLE001
        import warnings
        warnings.warn(f"Langfuse init failed (tracing disabled): {_lf_err}", stacklevel=1)


def get_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")


def get_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


async def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Call Gemini with a plain text prompt. Returns the model's text reply."""
    client = get_client()
    model = get_model()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text
