"""
LLM router: classifies the user's question and routes to the correct path.
Makes exactly ONE Gemini function-calling request per question.
"""

from __future__ import annotations

from core.db import get_data_dictionary
from core.models import QuerySpec, SearchParams

# ── Langfuse @observe decorator (optional) ────────────────────────────────────
try:
    from langfuse import observe as _observe
except ImportError:
    def _observe(name: str | None = None, **_kw):  # type: ignore[misc]
        def _decorator(fn):
            return fn
        return _decorator

# ── router system prompt ─────────────────────────────────────────────────────

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
    from google.genai import types

    from core.llm import get_client, get_model

    # ── function declarations ─────────────────────────────────────────────────
    search_hotels_fn = types.FunctionDeclaration(
        name="search_hotels",
        description="Search/filter/sort hotels by location, rating, price, or amenities.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "city":               types.Schema(type="STRING"),
                "country":            types.Schema(type="STRING"),
                "min_rating":         types.Schema(type="NUMBER"),
                "max_rating":         types.Schema(type="NUMBER"),
                "min_reviews":        types.Schema(type="INTEGER"),
                "price_min":          types.Schema(type="INTEGER"),
                "price_max":          types.Schema(type="INTEGER"),
                "required_amenities": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                    description=(
                        "List of amenity column names to require, e.g. "
                        '["has_pool","has_gym"]. '
                        "Valid values: has_wifi, has_pool, has_gym, has_sauna, "
                        "has_restaurant, has_room_service, has_lounge, has_event_space"
                    ),
                ),
                "sort_by":    types.Schema(type="STRING",
                    description='One of: "avg_score", "total_reviews", "distance", "price"'),
                "sort_order": types.Schema(type="STRING", enum=["asc", "desc"]),
                "limit":      types.Schema(type="INTEGER"),
                "user_lat":   types.Schema(type="NUMBER"),
                "user_lng":   types.Schema(type="NUMBER"),
                "radius_km":  types.Schema(type="NUMBER"),
            },
        ),
    )

    semantic_query_fn = types.FunctionDeclaration(
        name="semantic_query",
        description=(
            "Run an analytical query: count, average, min, max, sum, group-by. "
            "Use for questions about statistics, rankings of groups, or aggregates."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "metric": types.Schema(
                    type="STRING",
                    enum=list([
                        "count", "avg_score", "avg_reviews", "min_score", "max_score",
                        "min_reviews", "max_reviews", "total_reviews_sum",
                        "avg_price", "min_price", "max_price",
                    ]),
                ),
                "group_by":   types.Schema(type="STRING", enum=["city", "country"]),
                "filters":    types.Schema(
                    type="OBJECT",
                    description=(
                        "Key-value filters. Keys: city, country, min_rating, max_rating, "
                        "min_reviews, price_min, price_max, has_wifi, has_pool, has_gym, "
                        "has_sauna, has_restaurant, has_room_service, has_lounge, has_event_space"
                    ),
                ),
                "sort_order": types.Schema(type="STRING", enum=["asc", "desc"]),
                "limit":      types.Schema(type="INTEGER"),
            },
            required=["metric"],
        ),
    )

    text_to_sql_fn = types.FunctionDeclaration(
        name="text_to_sql",
        description=(
            "Arbitrary SQL SELECT for questions that don't fit the other tools. "
            "Use for complex WHERE conditions, LIKE searches, OR clauses, etc."
        ),
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "sql": types.Schema(type="STRING", description="A single SELECT statement"),
            },
            required=["sql"],
        ),
    )

    decline_fn = types.FunctionDeclaration(
        name="decline",
        description="The question cannot be answered from the available hotel data.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "reason": types.Schema(type="STRING"),
            },
            required=["reason"],
        ),
    )

    tools = types.Tool(function_declarations=[
        search_hotels_fn, semantic_query_fn, text_to_sql_fn, decline_fn,
    ])

    # ── build the question payload (include coordinates if provided) ──────────
    question_with_context = question
    if user_lat is not None and user_lng is not None:
        question_with_context += f" [User location: lat={user_lat}, lng={user_lng}]"

    # ── single Gemini call ────────────────────────────────────────────────────
    client = get_client()
    model = get_model()
    response = client.models.generate_content(
        model=model,
        contents=question_with_context,
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_SYSTEM_PROMPT,
            temperature=0.0,
            tools=[tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    )

    # ── parse function call ───────────────────────────────────────────────────
    try:
        part = response.candidates[0].content.parts[0]
        if not part.function_call:
            return "declined", "Router did not return a function call."
        fn_name = part.function_call.name
        fn_args = dict(part.function_call.args)
    except (IndexError, AttributeError) as e:
        return "declined", f"Failed to parse router response: {e}"

    # ── dispatch ──────────────────────────────────────────────────────────────
    if fn_name == "search_hotels":
        return "parameterized", SearchParams(**fn_args)

    if fn_name == "semantic_query":
        return "semantic", QuerySpec(**fn_args)

    if fn_name == "text_to_sql":
        # Router gave SQL but we pass the original question to fallback for its own generation
        return "fallback", question

    if fn_name == "decline":
        return "declined", fn_args.get("reason", "Cannot answer from available data.")

    return "declined", f"Unknown function: {fn_name}"
