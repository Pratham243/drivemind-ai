"""DriveMind AI — Streamlit dashboard.

Talks to the FastAPI backend over HTTP (API_BASE_URL), so this file has
zero direct dependency on the agent/NLU internals — exactly the same
contract any other client (mobile app, voice assistant) would use.

Run with:
    streamlit run app/streamlit_app.py
(the FastAPI server must be running separately — see README)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voice.stt import get_stt_provider  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="DriveMind AI", page_icon="🚗", layout="wide")


def call_api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", timeout=10, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, (
            f"Could not connect to the DriveMind API at {API_BASE_URL}. "
            "Start it with: uvicorn src.api.main:app --reload"
        )
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, detail
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

st.title("🚗 DriveMind AI")
st.caption("Agentic NLP & LLM-Based Intelligent In-Car Assistant")

col_chat, col_vehicle = st.columns([2, 1])

# ---------------------------------------------------------------------
# Vehicle status panel
# ---------------------------------------------------------------------
with col_vehicle:
    st.subheader("Vehicle Status")
    status, err = call_api("GET", "/vehicle/status")
    if err:
        st.error(err)
    else:
        c1, c2 = st.columns(2)
        c1.metric("Battery", f"{status['battery_pct']}%")
        c2.metric("Range", f"{status['range_km']} km")
        c1.metric("Temperature", f"{status['temperature_c']}°C")
        c2.metric("Speed", f"{status['speed_kmh']} km/h")

        with st.expander("Windows"):
            st.json(status["windows"])
        with st.expander("Tire pressure"):
            st.json(status["tire_pressure_bar"])
        with st.expander("Media"):
            st.json(status["media"])
        st.caption(f"Ambient light: {status['ambient_light_color']}")
        if status["navigation_active"]:
            st.info(f"Navigating to {status['navigation_destination']}")

    st.divider()
    health, herr = call_api("GET", "/health")
    if not herr:
        st.subheader("Model Info")
        st.write(f"**NLU model:** `{health['nlu_model']}`")
        st.write(f"**LLM provider:** `{health['llm_provider']}`")
        if health["llm_provider"] == "mock":
            st.caption("No OPENAI_API_KEY configured — using offline MockLLMProvider.")

    st.divider()
    if st.button("Refresh"):
        st.rerun()

# ---------------------------------------------------------------------
# Conversation panel
# ---------------------------------------------------------------------
with col_chat:
    st.subheader("Conversation")

    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.write(text)

    stt = get_stt_provider()
    if not stt.is_available():
        st.caption("🎤 Voice input not configured in this environment — text input below works fully.")

    user_input = st.chat_input("Ask DriveMind AI (e.g. 'Find a charging station near Berlin')")
    if user_input:
        st.session_state.history.append(("user", user_input))
        with st.chat_message("user"):
            st.write(user_input)

        result, err = call_api("POST", "/chat", json={"message": user_input})
        with st.chat_message("assistant"):
            if err:
                st.error(err)
                reply = f"Error: {err}"
            else:
                reply = result["response"]
                st.write(reply)
                st.session_state.last_trace = result
        st.session_state.history.append(("assistant", reply))

    st.divider()
    st.subheader("Technical Trace")
    trace = st.session_state.last_trace
    if trace is None:
        st.caption("Send a message to see the agent's reasoning trace here.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Intent", trace["intent"] or "—")
        t2.metric("Tool", trace["tool"] or "—")
        safety_status = (trace.get("safety") or {}).get("status", "—")
        t3.metric("Safety", safety_status.upper())

        st.write("**Entities**")
        st.json(trace["entities"])

        st.write("**Tool result**")
        st.json(trace["tool_result"])

        if trace.get("errors"):
            st.error(f"Errors: {trace['errors']}")

        if len(trace.get("steps", [])) > 1:
            st.write("**Multi-step execution**")
            for i, step in enumerate(trace["steps"], start=1):
                st.write(f"Step {i}: `{step['sub_message']}` → intent=`{step['intent']}`, "
                         f"tool=`{step['selected_tool']}`, safety=`{step['safety']['status']}`")

        with st.expander("Full raw trace (JSON)"):
            st.json(trace)

st.divider()
st.caption(
    "DriveMind AI — Master's portfolio project. NLU: TF-IDF/Transformer baseline + "
    "rule-based entity extraction + LLM fallback. Agent: multi-step tool-calling with "
    "a mandatory safety validation layer in front of every vehicle-state mutation."
)
