"""
LLM router: classifies the user's question and routes to the correct path.
Makes exactly ONE LLM function-calling request per question.

Tools are defined in the provider-agnostic OpenAI format (list of dicts).
core/llm.py converts them internally to Gemini FunctionDeclaration objects when
LLM_PROVIDER=gemini.
"""

from __future__ import annotations

from core.db import get_data_dictionary
from core.models import QuerySpec, SearchParams
from core.observability import observe as _observe

# ── Router system prompt ──────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = f"""You are a query router for a hotel database. Given a user's question, \
you must call exactly ONE of the available functions.

DATABASE SCHEMA:
{get_data_dictionary()}

ROUTING RULES (follow in order):

1. If the question asks to FIND, LIST, SHOW, or SEARCH for specific hotels (with optional
   filters like city, rating, price, location, amenities) → call `search_hotels` with the
   appropriate parameters.
   Examples: "top 5 hotels in Paris", "hotels near me with score above 8",
   "best reviewed hotels in London", "cheap hotels with a pool in Barcelona",
   "hotels under 200 per night with a gym"

2. If the question asks for COUNTS, AVERAGES, COMPARISONS, RANKINGS OF GROUPS, or other
   ANALYTICAL/AGGREGATE answers → call `semantic_query` with the appropriate metric,
   group_by, and filters.
   Examples: "how many hotels are in each city?", "average score in Barcelona",
   "which city has the highest rated hotels?", "total reviews across all hotels",
   "average price per night by city", "how many hotels have a pool?"

3. If the question is answerable from the data but doesn't fit the above patterns
   (complex conditions, name searches, unusual comparisons, OR/AND across amenities) →
   call `text_to_sql` with a valid SELECT statement.
   Examples: "hotels with 'Grand' in the name", "score difference between Paris and London",
   "hotels with more than 2000 reviews", "cheapest hotel with both a pool and a gym"

4. If the question CANNOT be answered from this data → call `decline` with a clear reason.
   MUST decline for: questions about live availability, booking, check-in/check-out dates,
   room type availability, review text/sentiment, photos, or any field not in the schema.
   Examples: "book me a room at the Ritz" → decline (not a data query),
   "what do guests say about Hotel X?" → decline (no review text),
   "what's the weather like in London?" → decline (not about hotel data)

IMPORTANT:
- NEVER force-fit a question. If in doubt, decline with a reason rather than guess.
- The database HAS: name, address, city, country, latitude, longitude, avg_score,
  total_reviews, price_per_night, has_wifi, has_pool, has_gym, has_sauna,
  has_restaurant, has_room_service, has_lounge, has_event_space.
- Amenity columns are 0/1 integers. Price (price_per_night) ranges from 80 to 465 USD.
- The avg_score is on a 1-10 scale, not 1-5.
- Cities in the data: London, Paris, Barcelona, Milan, Vienna, Amsterdam.
- For "near me" queries, use the user's provided coordinates with the haversine function.
"""

# ── Provider-agnostic tool definitions (OpenAI format) ────────────────────────
# core/llm.py converts these to Gemini FunctionDeclarations when provider=gemini.

ROUTER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search/filter/sort hotels by location, rating, price, or amenities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":               {"type": "string", "description": "Filter by city name"},
                    "country":            {"type": "string", "description": "Filter by country name"},
                    "min_rating":         {"type": "number", "description": "Minimum avg_score"},
                    "max_rating":         {"type": "number", "description": "Maximum avg_score"},
                    "min_reviews":        {"type": "integer", "description": "Minimum total_reviews"},
                    "price_min":          {"type": "integer", "description": "Minimum price_per_night (USD)"},
                    "price_max":          {"type": "integer", "description": "Maximum price_per_night (USD)"},
                    "required_amenities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of amenity column names to require, e.g. [\"has_pool\",\"has_gym\"]. "
                            "Valid values: has_wifi, has_pool, has_gym, has_sauna, "
                            "has_restaurant, has_room_service, has_lounge, has_event_space"
                        ),
                    },
                    "sort_by":    {
                        "type": "string",
                        "description": "One of: avg_score, total_reviews, distance, price",
                    },
                    "sort_order": {"type": "string", "enum": ["asc", "desc"]},
                    "limit":      {"type": "integer"},
                    "user_lat":   {"type": "number"},
                    "user_lng":   {"type": "number"},
                    "radius_km":  {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_query",
            "description": (
                "Run an analytical query: count, average, min, max, sum, group-by. "
                "Use for questions about statistics, rankings of groups, or aggregates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "count", "avg_score", "avg_reviews", "min_score", "max_score",
                            "min_reviews", "max_reviews", "total_reviews_sum",
                            "avg_price", "min_price", "max_price",
                        ],
                    },
                    "group_by":   {"type": "string", "enum": ["city", "country"]},
                    "filters":    {
                        "type": "object",
                        "description": (
                            "Key-value filters. Keys: city, country, min_rating, max_rating, "
                            "min_reviews, price_min, price_max, has_wifi, has_pool, has_gym, "
                            "has_sauna, has_restaurant, has_room_service, has_lounge, has_event_space"
                        ),
                    },
                    "sort_order": {"type": "string", "enum": ["asc", "desc"]},
                    "limit":      {"type": "integer"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_to_sql",
            "description": (
                "Arbitrary SQL SELECT for questions that don't fit the other tools. "
                "Use for complex WHERE conditions, LIKE searches, OR clauses, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement"},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decline",
            "description": "The question cannot be answered from the available hotel data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why it can't be answered"},
                },
                "required": ["reason"],
            },
        },
    },
]


@_observe(name="route_question")
async def route_question(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> tuple[str, SearchParams | QuerySpec | str | None]:
    """Route a natural-language question to the correct execution path.

    Returns (path_name, path_input) where path_name is one of:
      "parameterized" | "semantic" | "fallback" | "declined"
    and path_input is the corresponding SearchParams / QuerySpec / decline_reason.
    """
    from core.llm import function_call  # deferred to keep module importable without LLM setup

    # ── Build the question payload (include coordinates if provided) ──────────
    question_with_context = question
    if user_lat is not None and user_lng is not None:
        question_with_context += f" [User location: lat={user_lat}, lng={user_lng}]"

    # ── Single LLM function-calling request ───────────────────────────────────
    try:
        fn_name, fn_args = await function_call(
            messages=[{"role": "user", "content": question_with_context}],
            tools=ROUTER_TOOLS,
            system=ROUTER_SYSTEM_PROMPT,
            model_tier="router",
        )
    except Exception as e:  # noqa: BLE001
        return "declined", f"Router call failed: {e}"

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if fn_name == "search_hotels":
        return "parameterized", SearchParams(**fn_args)

    if fn_name == "semantic_query":
        return "semantic", QuerySpec(**fn_args)

    if fn_name == "text_to_sql":
        # Router chose the fallback path; pass the original question so
        # execute_fallback() builds its own carefully-prompted SQL.
        return "fallback", question

    if fn_name == "decline":
        return "declined", fn_args.get("reason", "Cannot answer from available data.")

    return "declined", f"Unknown function: {fn_name}"
