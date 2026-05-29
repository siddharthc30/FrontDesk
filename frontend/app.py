"""
Phase 6 — Streamlit frontend.

Consumes the FastAPI SSE stream and shows live progress steps while the
pipeline runs.  Renders the final answer, insights, hotel table, and a
Folium map.

Voice mode (Phase 8c): mic button → server-side STT → results + audio playback.
Text mode (default): existing question box + city/address fields.

Run with:
  streamlit run frontend/app.py
"""

from __future__ import annotations

import base64
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

def _secret_or_env(key: str, default: str = "") -> str:
    """Read from st.secrets first (Streamlit Cloud convention), fall back to env."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except (FileNotFoundError, Exception):  # noqa: BLE001  — secrets.toml may not exist locally
        pass
    return os.environ.get(key, default)


API_URL = _secret_or_env("API_URL", "http://localhost:8000")
APP_TOKEN = _secret_or_env("APP_TOKEN", "")


def _auth_headers(extra: dict | None = None) -> dict:
    headers = dict(extra or {})
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN
    return headers


# ── SSE helper ─────────────────────────────────────────────────────────────────

def stream_question(
    question: str,
    user_lat: float | None,
    user_lng: float | None,
    user_city: str | None,
    api_url: str,
    voice: bool = False,
):
    """Yield (event_type, data_dict) tuples from the SSE stream."""
    resp = requests.post(
        f"{api_url}/ask",
        json={
            "question": question,
            "user_lat": user_lat,
            "user_lng": user_lng,
            "user_city": user_city,
            "voice": voice,
        },
        stream=True,
        timeout=120,
        headers=_auth_headers({"Accept": "text/event-stream"}),
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


# ── Mode toggle ────────────────────────────────────────────────────────────────
st.markdown("---")
voice_mode = st.toggle("🎙️ Voice mode", value=False, help="Speak your question instead of typing")

# Clear voice state when switching to text mode
if not voice_mode:
    for key in ("voice_audio_bytes", "voice_transcript", "voice_recorder_generation"):
        st.session_state.pop(key, None)


# ── Input area ─────────────────────────────────────────────────────────────────

question: str = ""
user_lat: float | None = None
user_lng: float | None = None
selected_city: str | None = None
search_clicked: bool = False

if not voice_mode:
    # ── Text mode (unchanged) ──────────────────────────────────────────────────
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

    # ── Resolve user_lat / user_lng from city + address ───────────────────────
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

else:
    # ── Voice mode ─────────────────────────────────────────────────────────────
    try:
        from audio_recorder_streamlit import audio_recorder
        _recorder_available = True
    except ImportError:
        _recorder_available = False

    if not _recorder_available:
        st.error(
            "Voice mode requires `audio-recorder-streamlit`. "
            "Run: `pip install audio-recorder-streamlit`"
        )
        st.stop()

    st.markdown("### 🎙️ Speak your question")
    st.caption(
        "Tap the mic button and ask naturally — include the city, any filters, everything. "
        "For example: *'What are the top 5 hotels in Paris with a pool and good breakfast?'*"
    )

    # The key must change when the user re-records so Streamlit mounts a fresh
    # component instance (returning None) rather than replaying the old audio.
    recorder_key = f"voice_input_{st.session_state.get('voice_recorder_generation', 0)}"
    raw_audio = audio_recorder(
        text="",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="3x",
        pause_threshold=2.0,
        key=recorder_key,
    )

    # Detect new recording (ignore noise / empty clips)
    if raw_audio is not None and len(raw_audio) > 1000:
        if raw_audio != st.session_state.get("voice_audio_bytes"):
            st.session_state["voice_audio_bytes"] = raw_audio
            st.session_state.pop("voice_transcript", None)  # reset so we re-transcribe

    # Transcribe if we have audio but no transcript yet
    if st.session_state.get("voice_audio_bytes") and "voice_transcript" not in st.session_state:
        with st.spinner("Transcribing your question..."):
            encoded = base64.b64encode(st.session_state["voice_audio_bytes"]).decode()
            try:
                stt_resp = requests.post(
                    f"{API_URL}/api/stt",
                    json={"audio_b64": encoded, "content_type": "audio/wav"},
                    timeout=30,
                    headers=_auth_headers(),
                )
                stt_resp.raise_for_status()
                st.session_state["voice_transcript"] = stt_resp.json().get("text") or ""
            except requests.exceptions.ConnectionError:
                st.error(
                    "⚠️ Could not connect to the API for transcription. "
                    f"Make sure `uvicorn api.main:app --reload` is running at **{API_URL}**."
                )
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Transcription failed: {exc}. Please try recording again.")
                st.session_state["voice_transcript"] = ""

    transcript = st.session_state.get("voice_transcript")

    if transcript:
        st.info(f"**Heard:** {transcript}")
        question = transcript
        col_submit, col_retry = st.columns([1, 4])
        with col_submit:
            search_clicked = st.button("🔍 Search", type="primary", key="voice_search")
        with col_retry:
            if st.button("🔄 Re-record", key="voice_retry"):
                st.session_state.pop("voice_audio_bytes", None)
                st.session_state.pop("voice_transcript", None)
                st.session_state["voice_recorder_generation"] = (
                    st.session_state.get("voice_recorder_generation", 0) + 1
                )
                st.rerun()
    elif transcript == "":
        st.warning("Could not hear anything. Please tap the mic and try again.")
        if st.button("🔄 Try again", key="voice_retry_empty"):
            st.session_state.pop("voice_audio_bytes", None)
            st.session_state.pop("voice_transcript", None)
            st.session_state["voice_recorder_generation"] = (
                st.session_state.get("voice_recorder_generation", 0) + 1
            )
            st.rerun()


# ── Search & streaming ─────────────────────────────────────────────────────────
if search_clicked and question:
    result_data: dict | None = None

    with st.status("Processing your question...", expanded=True) as status:
        try:
            for event_type, data in stream_question(
                question,
                float(user_lat) if user_lat is not None else None,
                float(user_lng) if user_lng is not None else None,
                selected_city,
                API_URL,
                voice=voice_mode,
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

            # Audio playback (voice mode only) — fetched via side-channel
            # /api/tts AFTER the result arrives, so the large base64 MP3
            # blob doesn't have to travel inside an SSE event (proxies on
            # Streamlit Cloud / Render were dropping it).
            #
            # ⚠️ DIAGNOSTIC MODE: errors are surfaced verbatim in the UI and
            # printed to stdout so they appear in the deploy logs. Tighten
            # this back to a quiet caption once the root cause is fixed.
            if voice_mode:
                import traceback as _tb

                tts_text = result_data.get("answer", "")
                if insights:
                    tts_text = f"{tts_text}  {insights}" if tts_text else insights
                tts_text = tts_text.strip()

                if tts_text:
                    tts_url = f"{API_URL}/api/tts"
                    print(f"[TTS] POST {tts_url}  text_chars={len(tts_text)}", flush=True)
                    try:
                        tts_resp = requests.post(
                            tts_url,
                            json={"text": tts_text},
                            timeout=60,
                            headers=_auth_headers(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        tb = _tb.format_exc()
                        print(f"[TTS] request raised:\n{tb}", flush=True)
                        st.error(f"🔇 TTS request failed: {exc}")
                        st.code(tb, language="text")
                    else:
                        # Always show what came back — status, headers, body preview.
                        print(
                            f"[TTS] response status={tts_resp.status_code} "
                            f"content_type={tts_resp.headers.get('content-type')} "
                            f"body_bytes={len(tts_resp.content)}",
                            flush=True,
                        )
                        if tts_resp.status_code != 200:
                            body = tts_resp.text[:1000]
                            print(f"[TTS] non-200 body: {body}", flush=True)
                            st.error(
                                f"🔇 TTS HTTP {tts_resp.status_code} from {tts_url}"
                            )
                            st.code(body or "(empty body)", language="text")
                        else:
                            try:
                                payload = tts_resp.json()
                            except Exception as exc:  # noqa: BLE001
                                tb = _tb.format_exc()
                                print(f"[TTS] json decode failed:\n{tb}", flush=True)
                                st.error(f"🔇 TTS response was not JSON: {exc}")
                                st.code(tts_resp.text[:1000], language="text")
                            else:
                                audio_b64 = payload.get("audio_b64")
                                if audio_b64:
                                    print(
                                        f"[TTS] ok, audio_b64_chars={len(audio_b64)}",
                                        flush=True,
                                    )
                                    st.audio(
                                        base64.b64decode(audio_b64), format="audio/mp3"
                                    )
                                else:
                                    print(
                                        f"[TTS] 200 OK but no audio_b64 field. "
                                        f"keys={list(payload.keys())}",
                                        flush=True,
                                    )
                                    st.error(
                                        "🔇 TTS returned 200 OK but no audio_b64 field."
                                    )
                                    st.code(str(payload)[:1000], language="text")

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

    # After a voice search, clear the pending recording so mic is ready for next question
    if voice_mode:
        st.session_state.pop("voice_audio_bytes", None)
        st.session_state.pop("voice_transcript", None)
