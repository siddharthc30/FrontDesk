"""Pydantic models shared across all pipeline layers."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HotelRow(BaseModel):
    """One hotel from query results. Fields match the DB schema exactly."""
    hotel_id: int
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    latitude: float
    longitude: float
    avg_score: Optional[float] = None
    total_reviews: Optional[int] = None
    price_per_night: Optional[int] = None
    has_wifi: int = 0
    has_pool: int = 0
    has_gym: int = 0
    has_sauna: int = 0
    has_restaurant: int = 0
    has_room_service: int = 0
    has_lounge: int = 0
    has_event_space: int = 0
    distance_km: Optional[float] = None  # only populated for geo queries


class AspectFilter(BaseModel):
    """Filter on precomputed sentiment for a known aspect."""
    aspect: str
    min_sentiment: float = 0.0
    min_mentions: int = 0


class SearchParams(BaseModel):
    """Input to the parameterized path. Every field is optional."""
    city: Optional[str] = None
    country: Optional[str] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    min_reviews: Optional[int] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    required_amenities: Optional[list[str]] = None  # e.g. ["has_pool", "has_gym"]
    aspect_filters: Optional[list[AspectFilter]] = None  # e.g. pool sentiment >= 0.7
    sort_by: Optional[str] = None       # "avg_score" | "total_reviews" | "distance" | "price" | "<aspect>_sentiment"
    sort_order: Optional[str] = "desc"  # "asc" | "desc"
    limit: Optional[int] = 10
    user_lat: Optional[float] = None    # for geo queries
    user_lng: Optional[float] = None
    radius_km: Optional[float] = None   # max distance from user


class QuerySpec(BaseModel):
    """Input to the semantic path. Analytical/aggregate queries."""
    metric: str                     # see ALLOWED_METRICS in semantic.py
    aspect: Optional[str] = None    # required when metric="avg_sentiment"
    group_by: Optional[str] = None  # "city" | "country" | "aspect"
    filters: Optional[dict] = None  # e.g. {"city": "Paris", "min_rating": 8.0, "has_pool": 1}
    sort_by: Optional[str] = None
    sort_order: Optional[str] = "desc"
    limit: Optional[int] = None


class PipelineResponse(BaseModel):
    answer: str                          # short natural-language answer
    insights: str                        # 1–3 grounded observation sentences
    hotels: list[HotelRow]               # result rows (may be empty)
    path: str                            # "parameterized" | "semantic" | "review_search" | "fallback" | "declined"
    confidence: str = "high"             # "high" | "medium" | "guarded"
    query_ran: Optional[str] = None      # the SQL or query-spec that produced the result
    declined: bool = False
    decline_reason: Optional[str] = None
