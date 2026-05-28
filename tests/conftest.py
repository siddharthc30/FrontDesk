"""Pytest conftest.

The real `core.llm` module imports the google-genai SDK at import time, which
isn't installed in every dev environment. For unit tests that monkeypatch the
LLM call site, we don't need the real SDK — we install a stub module under
`core.llm` BEFORE any test imports `core.router` / `core.narrate`, so the
deferred `from core.llm import function_call` inside those modules picks up
the stub.
"""
from __future__ import annotations

import sys
import types


def _install_llm_stub() -> None:
    if "core.llm" in sys.modules:
        return
    stub = types.ModuleType("core.llm")

    async def _missing_function_call(*args, **kwargs):
        raise RuntimeError(
            "core.llm.function_call was not patched by the test; the stub "
            "is in place because the real SDK isn't importable."
        )

    async def _missing_chat_completion(*args, **kwargs):
        raise RuntimeError(
            "core.llm.chat_completion was not patched by the test; the stub "
            "is in place because the real SDK isn't importable."
        )

    stub.function_call = _missing_function_call
    stub.chat_completion = _missing_chat_completion
    sys.modules["core.llm"] = stub


_install_llm_stub()
