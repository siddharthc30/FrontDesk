"""
Phase 6 — Streamlit frontend.

Consumes the FastAPI SSE stream and shows live progress steps while the
pipeline runs.  Renders the final answer, insights, hotel table, and a
Folium map.

Run with:
  streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os

import folium
import requests
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from streamlit_folium import st_folium

CITY_CENTERS: dict[str, tuple[float, float]] = {
    "London":    (51.5074, -0.1278),
    "Paris":     (48.8566,  2.3522),
    "Barcelona": (41.3851,  2.1734),
    "Milan":     (45.4642,  9.1900),
    "Vienna":    (48.2082, 16.3738),
    "Amsterdam": (52.3676,  4.9041),
}


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_address(address: str) -> tuple[float, float, dict] | None:
    """Geocode `address` via Nominatim. Returns (lat, lng, address_components) or None."""
    geolocator = Nominatim(user_agent="frontdesk-hotel-search")
    try:
        result = geolocator.geocode(address, addressdetails=True, timeout=10)
    except (GeocoderServiceError, GeocoderTimedOut):
        return None
    if result is None:
        return None
    return result.latitude, result.longitude, result.raw.get("address", {})

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hotel NL Search", layout="wide", page_icon="🏨")
st.title("🏨 Hotel Natural Language Search")
st.caption("Ask any question about 1,000 hotels across 6 European cities — London, Paris, Barcelona, Milan, Vienna, Amsterdam")

API_URL = os.environ.get("API_URL", "http://localhost:8000")


# ── SSE helper ─────────────────────────────────────────────────────────────────

def stream_question(
    question: str,
    user_lat: float | None,
    user_lng: float | None,
    api_url: str,
):
    """Yield (event_type, data_dict) tuples from the SSE stream."""
    resp = requests.post(
        f"{api_url}/ask",
        json={"question": question, "user_lat": user_lat, "user_lng": user_lng},
        stream=True,
        timeout=120,
        headers={"Accept": "text/event-stream"},
    )
    resp.raise_for_status()

    event_type: str | None = None
    data_buffer: str = ""

    for raw_line in resp.iter_lines(decode_unicode=True):
        line = raw_line or ""
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_buffer = line[len("data:"):].strip()
        elif line == "":  # blank line = end of one SSE event
            if event_type and data_buffer:
                try:
                    yield event_type, json.loads(data_buffer)
                except json.JSONDecodeError:
                    pass
                event_type = None
                data_buffer = ""


# ── Input area ─────────────────────────────────────────────────────────────────
st.markdown("---")
question = st.text_input(
    "Ask a question about hotels:",
    placeholder='e.g. "Top 5 hotels in Paris", "How many hotels have a pool?", "Hotels near me with score above 8"',
)

col1, col2 = st.columns([1, 2])
with col1:
    city_options = ["None (no location)"] + list(CITY_CENTERS.keys())
    selected_city_label = st.selectbox(
        "City (optional — for 'near me' queries)",
        options=city_options,
        index=0,
    )
with col2:
    address_input = st.text_input(
        "Address in that city (optional — overrides city center)",
        placeholder="e.g. 10 Downing Street",
    )

selected_city = None if selected_city_label == "None (no location)" else selected_city_label

search_clicked = st.button("🔍 Search", type="primary", disabled=not question)

# ── Resolve user_lat / user_lng from city + address ────────────────────────────
user_lat: float | None = None
user_lng: float | None = None

if search_clicked and question:
    if selected_city is None:
        if address_input.strip():
            st.error("Please select a city before entering an address.")
            st.stop()
    else:
        if not address_input.strip():
            user_lat, user_lng = CITY_CENTERS[selected_city]
        else:
            geo = geocode_address(address_input.strip())
            if geo is None:
                st.error("Could not find that address — try a more specific one.")
                st.stop()
            lat, lng, components = geo
            resolved_city = (
                components.get("city")
                or components.get("town")
                or components.get("village")
                or components.get("municipality")
                or components.get("county")
                or ""
            )
            if resolved_city.strip().lower() != selected_city.lower():
                shown = resolved_city or "an unknown location"
                st.error(
                    f"That address resolves to **{shown}**, not **{selected_city}**. "
                    f"Please enter an address in {selected_city}."
                )
                st.stop()
            user_lat, user_lng = lat, lng

# ── Search & streaming ─────────────────────────────────────────────────────────
if search_clicked and question:
    result_data: dict | None = None

    with st.status("Processing your question...", expanded=True) as status:
        try:
            for event_type, data in stream_question(
                question,
                float(user_lat) if user_lat is not None else None,
                float(user_lng) if user_lng is not None else None,
                API_URL,
            ):
                if event_type == "step":
                    st.write(f"{data.get('message', '')}")

                elif event_type == "result":
                    result_data = data
                    if data.get("declined"):
                        status.update(label="Question declined", state="error", expanded=False)
                    else:
                        status.update(label="✅ Done!", state="complete", expanded=False)

                elif event_type == "error":
                    status.update(label="Error", state="error", expanded=True)
                    st.error(data.get("message", "Unknown error"))

        except requests.exceptions.ConnectionError:
            status.update(label="Connection error", state="error", expanded=True)
            st.error(
                "⚠️ Could not connect to the API server. "
                f"Make sure `uvicorn api.main:app --reload` is running at **{API_URL}**."
            )
            st.stop()
        except requests.exceptions.Timeout:
            status.update(label="Timeout", state="error", expanded=True)
            st.error("⚠️ The request timed out. Please try again.")
            st.stop()
        except requests.exceptions.HTTPError as exc:
            status.update(label="API error", state="error", expanded=True)
            st.error(f"⚠️ API returned an error: {exc}")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────────
    if result_data:
        if result_data.get("declined"):
            st.warning(
                f"**I can't answer that:** {result_data.get('decline_reason', 'Unknown reason')}"
            )
        else:
            import pandas as pd

            path = result_data.get("path", "unknown")
            confidence = result_data.get("confidence", "high")

            # ── Confidence banner for non-high paths ──────────────────────────
            if path == "review_search":
                search_terms = []
                query_ran_str = result_data.get("query_ran", "")
                if "terms=" in query_ran_str:
                    search_terms = query_ran_str.split("terms=")[-1]
                st.info(
                    f"**Results from searching guest reviews** for {search_terms}. "
                    "Sentiment values are computed on the fly from matching reviews, "
                    "not from precomputed scores. Mention counts indicate how many "
                    "guests wrote about this topic."
                )
            elif confidence == "guarded":
                st.warning("These results come from a generated SQL query and may be less reliable.")

            # Answer
            st.subheader("Answer")
            st.write(result_data.get("answer", ""))

            # Insights
            insights = result_data.get("insights", "")
            if insights:
                st.subheader("Insights")
                st.write(insights)

            # Hotels table + map
            hotels = result_data.get("hotels", [])
            if hotels:
                st.subheader(f"Hotels ({len(hotels)} result{'s' if len(hotels) != 1 else ''})")

                if path == "review_search":
                    # Review-search results have a different shape
                    rs_cols = ["name", "city", "country", "avg_score", "price_per_night",
                               "mention_count", "pos_count", "neg_count", "sentiment"]
                    rs_data = []
                    for h in hotels:
                        rs_data.append({c: h.get(c) for c in rs_cols if c in h})
                    df = pd.DataFrame(rs_data)
                    if "sentiment" in df.columns:
                        df["sentiment"] = df["sentiment"].apply(
                            lambda v: f"{v:.1%}" if v is not None else "—"
                        )
                    st.dataframe(df, use_container_width=True)
                else:
                    df = pd.DataFrame(hotels)

                    # Drop columns that are entirely 0 / None (e.g. all amenities off)
                    core_cols = {"hotel_id", "name", "city", "country", "avg_score",
                                 "total_reviews", "latitude", "longitude", "address",
                                 "price_per_night", "distance_km"}
                    cols_to_drop = [
                        c for c in df.columns
                        if c not in core_cols and df[c].fillna(0).eq(0).all()
                    ]
                    df = df.drop(columns=cols_to_drop, errors="ignore")

                    # Drop distance_km if all null (non-geo query)
                    if "distance_km" in df.columns and df["distance_km"].isna().all():
                        df = df.drop(columns=["distance_km"])

                    st.dataframe(df, use_container_width=True)

                # ── Folium map ─────────────────────────────────────────────────
                st.subheader("Map")
                lats = [h["latitude"] for h in hotels if h.get("latitude")]
                lngs = [h["longitude"] for h in hotels if h.get("longitude")]

                if lats and lngs:
                    center_lat = sum(lats) / len(lats)
                    center_lng = sum(lngs) / len(lngs)

                    m = folium.Map(location=[center_lat, center_lng], zoom_start=12)
                    for h in hotels:
                        if not h.get("latitude") or not h.get("longitude"):
                            continue
                        score = h.get("avg_score", "N/A")
                        city  = h.get("city", "")
                        price = h.get("price_per_night")
                        price_str = f"<br>Price: ${price}/night" if price else ""
                        # For review-search results, show mention count in popup
                        extra = ""
                        if path == "review_search":
                            mentions = h.get("mention_count", 0)
                            sentiment = h.get("sentiment")
                            sent_str = f" ({sentiment:.0%} positive)" if sentiment is not None else ""
                            extra = f"<br>Mentions: {mentions}{sent_str}"
                        popup_html = (
                            f"<b>{h['name']}</b><br>"
                            f"Score: {score}<br>"
                            f"{city}{price_str}{extra}"
                        )
                        folium.CircleMarker(
                            location=[h["latitude"], h["longitude"]],
                            radius=8,
                            color="#3186cc",
                            weight=2,
                            fill=True,
                            fill_color="#3186cc",
                            fill_opacity=0.7,
                            popup=folium.Popup(popup_html, max_width=250),
                            tooltip=h["name"],
                        ).add_to(m)

                    st_folium(m, width=700, height=500, returned_objects=[])

        # ── Transparency expander ──────────────────────────────────────────────
        with st.expander("🔍 Query details (transparency)"):
            path = result_data.get("path", "unknown")
            confidence = result_data.get("confidence", "high")
            confidence_icon = {"high": "✅", "medium": "⚠️", "guarded": "🔶"}.get(confidence, "")
            st.write(f"**Path used:** `{path}`")
            st.write(f"**Confidence:** {confidence_icon} `{confidence}`")
            query_ran = result_data.get("query_ran")
            if query_ran:
                lang = "sql" if query_ran.strip().upper().startswith("SELECT") else "text"
                st.code(query_ran, language=lang)
