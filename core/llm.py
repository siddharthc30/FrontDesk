"""LLM provider abstraction layer.

All LLM calls in the pipeline go through this module.

Provider selection:
    LLM_PROVIDER=gemini   (default) — uses google-genai SDK
    LLM_PROVIDER=openai             — uses openai SDK

Callers use the three provider-agnostic functions:
    chat_completion(messages, system, model_tier, response_json, temperature) -> str
    function_call(messages, tools, system, model_tier) -> (fn_name, fn_args)
    call_llm(prompt, temperature) -> str   # backward-compat thin wrapper
"""

from __future__ import annotations

import json
import os
from typing import Any

# ── Provider selection ─────────────────────────────────────────────────────────

_PROVIDER: str = os.environ.get("LLM_PROVIDER", "gemini").lower()


# ── Langfuse + provider auto-instrumentation ──────────────────────────────────
# Must run BEFORE any SDK client is instantiated, since the instrumentor
# monkey-patches the SDK's network calls.

def _init_langfuse_instrumentation(provider: str) -> None:
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return
    try:
        from langfuse import get_client as _get_langfuse_client
        _get_langfuse_client()  # initialises the global Langfuse client / OTel exporter

        if provider == "gemini":
            from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
            GoogleGenAIInstrumentor().instrument()
        elif provider == "openai":
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
    except Exception as _lf_err:  # noqa: BLE001
        import warnings
        warnings.warn(
            f"Langfuse init failed (tracing disabled): {_lf_err}", stacklevel=1
        )


_init_langfuse_instrumentation(_PROVIDER)


# ── Gemini initialisation ──────────────────────────────────────────────────────

if _PROVIDER == "gemini":
    from google import genai as _genai
    from google.genai import types as _gtypes

    _gemini_client = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    _ROUTER_MODEL: str = os.environ.get(
        "GEMINI_ROUTER_MODEL",
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"),
    )
    _MAIN_MODEL: str = os.environ.get(
        "GEMINI_MAIN_MODEL",
        os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"),
    )


# ── OpenAI initialisation ──────────────────────────────────────────────────────

elif _PROVIDER == "openai":
    import openai as _openai_sdk

    _base_url: str | None = os.environ.get("LLM_BASE_URL") or None
    _openai_client = _openai_sdk.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=_base_url,
    )
    _ROUTER_MODEL = os.environ.get("OPENAI_ROUTER_MODEL", "gpt-4.1-nano")
    _MAIN_MODEL = os.environ.get("OPENAI_MAIN_MODEL", "gpt-4.1-mini")

else:
    raise ValueError(
        f"Unknown LLM_PROVIDER={_PROVIDER!r}. Set LLM_PROVIDER to 'gemini' or 'openai'."
    )


# ── Model tier helpers ─────────────────────────────────────────────────────────

def _resolve_model(model_tier: str) -> str:
    """Map a logical tier ('router' | 'main') to the provider's model string."""
    if model_tier == "router":
        return _ROUTER_MODEL
    return _MAIN_MODEL


# ── Gemini: JSON-schema ↔ types.Schema conversion ─────────────────────────────

def _json_schema_to_gemini(schema: dict) -> Any:
    """Recursively convert a JSON Schema dict to a Gemini types.Schema object."""
    TYPE_MAP = {
        "string":  "STRING",
        "number":  "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array":   "ARRAY",
        "object":  "OBJECT",
    }
    raw_type = schema.get("type", "string")
    gemini_type = TYPE_MAP.get(raw_type, "STRING")

    kwargs: dict[str, Any] = {"type": gemini_type}

    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "enum" in schema:
        kwargs["enum"] = schema["enum"]
    if "properties" in schema:
        kwargs["properties"] = {
            k: _json_schema_to_gemini(v)
            for k, v in schema["properties"].items()
        }
    if "required" in schema:
        kwargs["required"] = schema["required"]
    if "items" in schema:
        kwargs["items"] = _json_schema_to_gemini(schema["items"])

    return _gtypes.Schema(**kwargs)  # type: ignore[name-defined]


def _openai_tools_to_gemini(tools: list[dict]) -> Any:
    """Convert an OpenAI-style tools list to a Gemini types.Tool object."""
    declarations = []
    for tool in tools:
        fn = tool["function"]
        raw_params = fn.get("parameters", {"type": "object", "properties": {}})
        decl = _gtypes.FunctionDeclaration(  # type: ignore[name-defined]
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=_json_schema_to_gemini(raw_params),
        )
        declarations.append(decl)
    return _gtypes.Tool(function_declarations=declarations)  # type: ignore[name-defined]


# ── Provider implementations ───────────────────────────────────────────────────

async def _gemini_chat_completion(
    messages: list[dict],
    system: str | None,
    model: str,
    response_json: bool,
    json_schema: dict | None,
    temperature: float,
) -> str:
    """Gemini path for chat_completion."""
    # Combine all user messages into a single prompt string (all our calls are single-turn).
    user_content = "\n\n".join(
        m["content"] for m in messages if m.get("role") != "system"
    )

    cfg_kwargs: dict[str, Any] = {"temperature": temperature}
    if system:
        cfg_kwargs["system_instruction"] = system
    if response_json:
        cfg_kwargs["response_mime_type"] = "application/json"
        if json_schema is not None:
            cfg_kwargs["response_schema"] = _json_schema_to_gemini(json_schema)

    response = _gemini_client.models.generate_content(  # type: ignore[name-defined]
        model=model,
        contents=user_content,
        config=_gtypes.GenerateContentConfig(**cfg_kwargs),  # type: ignore[name-defined]
    )
    return response.text


async def _openai_chat_completion(
    messages: list[dict],
    system: str | None,
    model: str,
    response_json: bool,
    temperature: float,
) -> str:
    """OpenAI path for chat_completion."""
    oai_messages: list[dict] = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": oai_messages,
        "temperature": temperature,
    }
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}

    resp = _openai_client.chat.completions.create(**kwargs)  # type: ignore[name-defined]
    return resp.choices[0].message.content or ""


async def _gemini_function_call(
    messages: list[dict],
    tools: list[dict],
    system: str | None,
    model: str,
) -> tuple[str, dict]:
    """Gemini path for function_call."""
    user_content = "\n\n".join(
        m["content"] for m in messages if m.get("role") != "system"
    )

    gemini_tool = _openai_tools_to_gemini(tools)

    cfg_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "tools": [gemini_tool],
        "tool_config": _gtypes.ToolConfig(  # type: ignore[name-defined]
            function_calling_config=_gtypes.FunctionCallingConfig(mode="ANY"),  # type: ignore[name-defined]
        ),
    }
    if system:
        cfg_kwargs["system_instruction"] = system

    response = _gemini_client.models.generate_content(  # type: ignore[name-defined]
        model=model,
        contents=user_content,
        config=_gtypes.GenerateContentConfig(**cfg_kwargs),  # type: ignore[name-defined]
    )

    part = response.candidates[0].content.parts[0]
    if not part.function_call:
        raise ValueError("Gemini did not return a function call.")

    fn_name: str = part.function_call.name
    fn_args: dict = dict(part.function_call.args)
    return fn_name, fn_args


async def _openai_function_call(
    messages: list[dict],
    tools: list[dict],
    system: str | None,
    model: str,
) -> tuple[str, dict]:
    """OpenAI path for function_call."""
    oai_messages: list[dict] = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)

    resp = _openai_client.chat.completions.create(  # type: ignore[name-defined]
        model=model,
        messages=oai_messages,
        tools=tools,
        tool_choice="required",  # force exactly one function call (equivalent to Gemini's "ANY")
        temperature=0.0,
    )

    msg = resp.choices[0].message
    if not msg.tool_calls:
        raise ValueError("OpenAI did not return a tool call.")

    tc = msg.tool_calls[0]
    fn_name = tc.function.name
    fn_args = json.loads(tc.function.arguments)
    return fn_name, fn_args


# ── Public provider-agnostic interface ─────────────────────────────────────────

async def chat_completion(
    messages: list[dict],
    system: str | None = None,
    model_tier: str = "main",
    response_json: bool = False,
    json_schema: dict | None = None,
    temperature: float = 0.0,
) -> str:
    """Send a chat completion request to the active provider.

    Args:
        messages:      OpenAI-style list of {"role": ..., "content": ...} dicts.
        system:        Optional system instruction / system prompt.
        model_tier:    "router" (cheap/fast) or "main" (capable). Mapped per provider.
        response_json: If True, request JSON-formatted output.
        json_schema:   Optional JSON Schema dict describing the expected output structure.
                       For Gemini this becomes a response_schema; for OpenAI it is informational
                       only (json_object mode is used instead of structured outputs).
        temperature:   Sampling temperature.

    Returns:
        The model's text response as a string.
    """
    model = _resolve_model(model_tier)

    if _PROVIDER == "openai":
        return await _openai_chat_completion(
            messages, system, model, response_json, temperature
        )
    else:
        return await _gemini_chat_completion(
            messages, system, model, response_json, json_schema, temperature
        )


async def function_call(
    messages: list[dict],
    tools: list[dict],
    system: str | None = None,
    model_tier: str = "router",
) -> tuple[str, dict]:
    """Make a function-calling LLM request to the active provider.

    Args:
        messages:   OpenAI-style message list (typically a single user message).
        tools:      OpenAI-style tools list:
                    [{"type": "function", "function": {"name": ..., "description": ...,
                                                        "parameters": <JSON Schema>}}]
                    Internally converted to Gemini FunctionDeclaration when provider=gemini.
        system:     Optional system prompt.
        model_tier: "router" or "main".

    Returns:
        (function_name: str, function_args: dict)
    """
    model = _resolve_model(model_tier)

    if _PROVIDER == "openai":
        return await _openai_function_call(messages, tools, system, model)
    else:
        return await _gemini_function_call(messages, tools, system, model)


async def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Backward-compatible single-prompt text completion.

    Wraps chat_completion() with a single user message. Used by core/fallback.py.
    """
    return await chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
