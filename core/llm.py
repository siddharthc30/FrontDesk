"""Gemini client wrapper. All LLM calls in the pipeline go through this module."""

from __future__ import annotations

import os

from google import genai
from google.genai import types


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
