"""Shared Langfuse @observe shim.

Import `observe` from this module in every pipeline step. When Langfuse is
installed it's the real decorator; otherwise it's a no-op that lets the app
run unchanged. Provider-level auto-instrumentation (Gemini / OpenAI) is wired
in core/llm.py at import time.
"""

from __future__ import annotations

try:
    from langfuse import observe  # type: ignore[assignment]
except ImportError:  # Langfuse not installed
    def observe(*_args, **_kwargs):  # type: ignore[misc]
        """No-op decorator used when Langfuse is unavailable."""
        def _decorator(fn):
            return fn
        return _decorator
