import json
import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"
DB_PATH = "vanguard.db"
# Groq retired llama-3.3-70b-versatile on this account; this is what tests confirmed active.
DEFAULT_MODEL = "openai/gpt-oss-20b"


def stream_completion(messages: list[dict]):
    """Yield assistant text from the gateway's SSE stream, one delta at a time."""
    payload = {"model": DEFAULT_MODEL, "messages": messages, "stream": True}
    try:
        with requests.post(GATEWAY_URL, json=payload, stream=True, timeout=(5, 180)) as response:
            if response.status_code != 200:
                yield f"Gateway returned {response.status_code}: {response.text[:300]}"
                return
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
    except requests.RequestException as exc:
        yield f"Cannot reach the gateway at {GATEWAY_URL} - is uvicorn running? ({exc})"


# Guarded so tests can import stream_completion without rendering the page.
if __name__ == "__main__":
    st.set_page_config(page_title="VanguardAI", layout="wide")

    tab1, tab2 = st.tabs(["💬 Live Chat", "📊 Gateway Telemetry"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask the gateway anything"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                reply = st.write_stream(stream_completion(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": reply})

    with tab2:
        if not Path(DB_PATH).exists():
            st.info(f"{DB_PATH} does not exist yet. Send a chat message first.")
        else:
            connection = sqlite3.connect(DB_PATH)
            try:
                logs = pd.read_sql_query(
                    "SELECT * FROM request_logs ORDER BY id DESC", connection
                )
            finally:
                connection.close()

            if logs.empty:
                st.info("No requests logged yet.")
            else:
                total, hit_rate, avg_latency = st.columns(3)
                total.metric("Total Requests Intercepted", len(logs))
                hit_rate.metric("Cache Hit Rate (%)", f"{logs['cache_hit'].mean() * 100:.1f}")
                avg_latency.metric("Average Latency (ms)", f"{logs['ttft_ms'].mean():.1f}")
                st.dataframe(logs, width="stretch")
