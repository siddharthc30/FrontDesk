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

# ── Known aspect taxonomy ────────────────────────────────────────────────────
# Facility aspects: have both a has_X boolean on hotels AND precomputed sentiment
FACILITY_ASPECTS = [
    "wifi", "pool", "gym", "sauna", "restaurant", "room_service", "lounge",
    # event_space excluded: unreliable (only 6/1000 hotels detected)
]

# Experiential aspects: sentiment only, no boolean flag
EXPERIENTIAL_ASPECTS = [
    "staff", "cleanliness", "location", "room_comfort", "value", "noise", "breakfast",
]

KNOWN_ASPECTS = FACILITY_ASPECTS + EXPERIENTIAL_ASPECTS

_ASPECT_TAXONOMY_TEXT = f"""\
KNOWN ASPECT TAXONOMY (precomputed in hotel_aspect_sentiment table):
  Facility aspects (also have has_X boolean on hotels): {", ".join(FACILITY_ASPECTS)}
  Experiential aspects (sentiment only): {", ".join(EXPERIENTIAL_ASPECTS)}

ASPECT ROUTING RULES:
- "hotels with good [known aspect]" → use `search_hotels` with aspect_filters
  e.g. "hotels with great staff" → search_hotels with aspect_filters=[{{aspect:"staff", min_sentiment:0.7}}]
- "average [known aspect] by city" → use `semantic_query` with metric="avg_sentiment", aspect="...", group_by="city"
- "[unknown/novel aspect]" (not in the known list above) → use `review_search` with search terms
  e.g. "hotels with rooftop bars" → review_search with terms=["rooftop", "bar"]
  e.g. "hotels where guests mention pets" → review_search with terms=["pet", "dog", "cat"]

Typical sentiment ranges for calibration (not for routing decisions):
  location ~0.90, staff ~0.82, cleanliness ~0.78, breakfast ~0.58,
  room_comfort ~0.56, noise ~0.51, value ~0.37
"""

# ── Router system prompt ──────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = f"""You are a query router for a hotel database. Given a user's question, \
you must call exactly ONE of the available functions.

DATABASE SCHEMA:
{get_data_dictionary()}

{_ASPECT_TAXONOMY_TEXT}

RESOLVING DEICTIC REFERENCES (read carefully):

The user's request may be preceded by a [User context: ...] line containing
some or all of: city="<name>", lat=<float>, lng=<float>. Use it as follows:

- CITY-SCOPE DEICTICS — phrases like "my city", "this city", "in this city",
  "in town", "here", "around here", "locally" — refer to the user_city from
  [User context]. Substitute it as the `city` parameter (search_hotels) or
  a `city` filter (semantic_query). Do NOT use radius_km for these phrases;
  they mean city-scope, not proximity. Do NOT try to reverse-geocode lat/lng
  into a city — only use the explicit user_city string.

- PROXIMITY DEICTICS — phrases like "near me", "close to me", "within X km",
  "walking distance", "nearby" — refer to the user_lat/user_lng from
  [User context]. Pass them as user_lat/user_lng + radius_km. Do NOT also
  apply the user_city as a filter for these phrases.

- If a city-scope deictic is present but user_city is NOT in [User context],
  call `decline` with reason "Please select a city to use 'my city' (or
  similar). I don't know which city you mean." Do not silently drop the
  constraint. Do not infer a city from lat/lng.

- If a proximity deictic is present but lat/lng are NOT in [User context],
  call `decline` with reason "Please select a city so I know where 'near
  me' is."

ROUTING RULES (follow in order):

1. If the question asks to FIND, LIST, SHOW, or SEARCH for specific hotels with filters
   (city, rating, price, amenities, or KNOWN aspects listed above) → call `search_hotels`.
   Examples:
   - "top 5 hotels in Paris" → search_hotels(city="Paris", limit=5)
   - "hotels with pool in my city" + [User context: city="Paris"] →
     search_hotels(city="Paris", required_amenities=["has_pool"])  (no radius_km)
   - "hotels near me" + [User context: lat=48.85, lng=2.35] →
     search_hotels(user_lat=48.85, user_lng=2.35, radius_km=5)  (no city filter)
   - "hotels in my city" with NO user_city in [User context] →
     decline(reason="Please select a city to use 'my city'.")
   - "hotels near me with score above 8" → search_hotels(min_rating=8.0, user_lat=..., user_lng=..., radius_km=...)
   - "best hotels with great staff" → search_hotels(aspect_filters=[{{aspect:"staff", min_sentiment:0.75}}])
   - "hotels with a pool in Barcelona sorted by pool sentiment" → search_hotels(city="Barcelona", required_amenities=["has_pool"], sort_by="pool_sentiment")
   - "best value hotels in London" → search_hotels(city="London", sort_by="value_sentiment")

2. If the question asks for COUNTS, AVERAGES, COMPARISONS, or other AGGREGATE answers,
   including aggregate sentiment for KNOWN aspects → call `semantic_query`.
   Examples:
   - "how many hotels are in each city?" → semantic_query(metric="count", group_by="city")
   - "average staff sentiment by city" → semantic_query(metric="avg_sentiment", aspect="staff", group_by="city")
   - "average breakfast rating by city" → semantic_query(metric="avg_sentiment", aspect="breakfast", group_by="city")
   - "how many hotels have a pool?" → semantic_query(metric="count", filters={{"has_pool": 1}})

3. If the question asks about a NOVEL or UNKNOWN aspect (not in the known taxonomy above),
   requiring a search through actual review text → call `review_search` with relevant terms.
   Examples:
   - "hotels with rooftop terraces" → review_search(search_terms=["rooftop", "terrace"])
   - "hotels where guests mention pets" → review_search(search_terms=["pet", "dog", "cat"])
   - "hotels with butler service" → review_search(search_terms=["butler"])
   - "hotels with private beach access" → review_search(search_terms=["private beach"])

4. If the question is answerable from the data but doesn't fit the above patterns
   (complex conditions, name searches, unusual comparisons, OR/AND across amenities,
   OR questions about the database schema/structure) →
   call `text_to_sql` with a valid SELECT statement.
   Examples:
   - "hotels with 'Grand' in the name" → text_to_sql
   - "cheapest hotel with both a pool and a gym" → text_to_sql
   - "how many columns are there in the hotel table?" → text_to_sql(sql="SELECT COUNT(*) AS column_count FROM pragma_table_info('hotels')")
   - "what columns does the hotels table have?" → text_to_sql(sql="SELECT name FROM pragma_table_info('hotels')")
   - "what data do you have on hotels?" → text_to_sql(sql="SELECT name FROM pragma_table_info('hotels')")
   - "describe the hotel table" → text_to_sql(sql="SELECT name FROM pragma_table_info('hotels')")
   - "what fields are in the database?" → text_to_sql(sql="SELECT name FROM pragma_table_info('hotels')")
   - "show me the table structure" → text_to_sql(sql="SELECT name FROM pragma_table_info('hotels')")

5. If the question CANNOT be answered from this data → call `decline` with a clear reason.
   MUST decline for: live availability, booking, check-in dates, photos.
   Do NOT decline schema/metadata questions about the hotel table structure — those are answerable via text_to_sql (rule 4).

IMPORTANT:
- NEVER force-fit a question. If in doubt, decline with a reason rather than guess.
- avg_score is on a 1-10 scale, not 1-5.
- Cities: London, Paris, Barcelona, Milan, Vienna, Amsterdam.
- For "near me" queries, use user coordinates with haversine.
- Amenity has_X flags mean "guests mentioned this" — not confirmed presence.
"""

# ── Provider-agnostic tool definitions (OpenAI format) ────────────────────────

ROUTER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": (
                "Search/filter/sort hotels by location, rating, price, amenities, or known aspect sentiment. "
                "Use aspect_filters for known aspects (staff, pool, cleanliness, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city":               {"type": "string", "description": "Filter by city name"},
                    "country":            {"type": "string", "description": "Filter by country name"},
                    "min_rating":         {"type": "number", "description": "Minimum avg_score (1-10)"},
                    "max_rating":         {"type": "number", "description": "Maximum avg_score (1-10)"},
                    "min_reviews":        {"type": "integer", "description": "Minimum total_reviews"},
                    "price_min":          {"type": "integer", "description": "Minimum price_per_night (USD)"},
                    "price_max":          {"type": "integer", "description": "Maximum price_per_night (USD)"},
                    "required_amenities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Amenity column names to require. "
                            "Valid: has_wifi, has_pool, has_gym, has_sauna, "
                            "has_restaurant, has_room_service, has_lounge"
                        ),
                    },
                    "aspect_filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "aspect":        {"type": "string", "description": f"One of: {', '.join(KNOWN_ASPECTS)}"},
                                "min_sentiment": {"type": "number", "description": "Minimum sentiment (0.0-1.0)"},
                                "min_mentions":  {"type": "integer", "description": "Minimum mention count"},
                            },
                            "required": ["aspect"],
                        },
                        "description": "Filter by precomputed aspect sentiment for known aspects.",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": (
                            "Sort column: avg_score, total_reviews, distance, price, "
                            "or <aspect>_sentiment (e.g. staff_sentiment, pool_sentiment)"
                        ),
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
                "Analytical query: count, average, min, max, sum, group-by. "
                "Use avg_sentiment metric for known aspects. "
                "Use for statistics, rankings of groups, or aggregates."
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
                            "avg_sentiment",
                        ],
                    },
                    "aspect": {
                        "type": "string",
                        "description": f"Required when metric=avg_sentiment. One of: {', '.join(KNOWN_ASPECTS)}",
                    },
                    "group_by":   {"type": "string", "enum": ["city", "country", "aspect"]},
                    "filters":    {
                        "type": "object",
                        "description": (
                            "Key-value filters on hotels. Keys: city, country, min_rating, max_rating, "
                            "min_reviews, price_min, price_max, has_wifi, has_pool, has_gym, "
                            "has_sauna, has_restaurant, has_room_service, has_lounge"
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
            "name": "review_search",
            "description": (
                "Search guest review text for novel/unknown aspects not in the precomputed taxonomy. "
                "Returns hotels ranked by how many guests mentioned the terms. "
                "Use for aspects like 'rooftop bar', 'butler service', 'private beach', 'pet-friendly', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to search for in review text (2-4 terms max)",
                    },
                    "city":       {"type": "string", "description": "Optional: restrict to one city"},
                    "min_rating": {"type": "number", "description": "Optional: minimum hotel avg_score"},
                    "limit":      {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["search_terms"],
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


# ReviewSearchParams is used as the path_input for "review_search"
from pydantic import BaseModel
from typing import Optional


class ReviewSearchParams(BaseModel):
    search_terms: list[str]
    city: Optional[str] = None
    min_rating: Optional[float] = None
    limit: int = 10


@_observe(name="route_question")
async def route_question(
    question: str,
    user_lat: float | None = None,
    user_lng: float | None = None,
    user_city: str | None = None,
) -> tuple[str, SearchParams | QuerySpec | ReviewSearchParams | str | None]:
    """Route a natural-language question to the correct execution path.

    Returns (path_name, path_input) where path_name is one of:
      "parameterized" | "semantic" | "review_search" | "fallback" | "declined"
    """
    from core.llm import function_call  # deferred to keep module importable without LLM setup

    ctx_parts: list[str] = []
    if user_city:
        ctx_parts.append(f'city="{user_city}"')
    if user_lat is not None and user_lng is not None:
        ctx_parts.append(f"lat={user_lat}")
        ctx_parts.append(f"lng={user_lng}")

    question_with_context = question
    if ctx_parts:
        question_with_context += f" [User context: {', '.join(ctx_parts)}]"

    try:
        fn_name, fn_args = await function_call(
            messages=[{"role": "user", "content": question_with_context}],
            tools=ROUTER_TOOLS,
            system=ROUTER_SYSTEM_PROMPT,
            model_tier="router",
        )
    except Exception as e:  # noqa: BLE001
        return "declined", f"Router call failed: {e}"

    if fn_name == "search_hotels":
        return "parameterized", SearchParams(**fn_args)

    if fn_name == "semantic_query":
        return "semantic", QuerySpec(**fn_args)

    if fn_name == "review_search":
        return "review_search", ReviewSearchParams(**fn_args)

    if fn_name == "text_to_sql":
        return "fallback", question

    if fn_name == "decline":
        return "declined", fn_args.get("reason", "Cannot answer from available data.")

    return "declined", f"Unknown function: {fn_name}"
