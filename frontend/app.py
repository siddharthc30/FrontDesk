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
from streamlit_folium import st_folium

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

col1, col2 = st.columns(2)
with col1:
    user_lat = st.number_input(
        "Your latitude (optional — for 'near me' queries)",
        value=None,
        format="%.6f",
    )
with col2:
    user_lng = st.number_input(
        "Your longitude (optional)",
        value=None,
        format="%.6f",
    )

search_clicked = st.button("🔍 Search", type="primary", disabled=not question)

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

                # Clean up the dataframe: drop zero/null amenity/distance cols if unset
                import pandas as pd

                df = pd.DataFrame(hotels)

                # Drop columns that are entirely 0 / None (e.g. amenities all off)
                cols_to_drop = [
                    c for c in df.columns
                    if c not in ("id", "name", "city", "country", "avg_score", "total_reviews",
                                 "latitude", "longitude", "address", "price_per_night", "distance_km")
                    and df[c].fillna(0).eq(0).all()
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
                        popup_html = (
                            f"<b>{h['name']}</b><br>"
                            f"Score: {score}<br>"
                            f"{city}{price_str}"
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
            st.write(f"**Path used:** `{path}`")
            query_ran = result_data.get("query_ran")
            if query_ran:
                # Show as SQL if it looks like SQL, otherwise as plain text/JSON
                lang = "sql" if query_ran.strip().upper().startswith("SELECT") else "json"
                st.code(query_ran, language=lang)
