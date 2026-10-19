"""Small Streamlit client for the FastAPI RAG and guardrail service."""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

st.set_page_config(page_title="Trustworthy AI Guardrails", layout="centered")


def _post_chat(api_url: str, session_id: str, message: str) -> dict[str, object]:
    response = httpx.post(
        f"{api_url.rstrip('/')}/api/v1/chat",
        json={"session_id": session_id, "message": message},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def _render_response(payload: dict[str, object]) -> None:
    st.markdown(str(payload.get("answer", "")))

    guardrails = payload.get("guardrails")
    if isinstance(guardrails, dict):
        status = str(guardrails.get("status", "unknown"))
        flags = guardrails.get("flags", [])
        flag_text = ", ".join(str(flag) for flag in flags) if isinstance(flags, list) else ""
        label = f"Guardrails: {status}"
        if flag_text:
            label += f" | {flag_text}"
        st.caption(label)

    sources = payload.get("sources")
    if isinstance(sources, list) and sources:
        with st.expander("Sources"):
            for source in sources:
                if isinstance(source, dict):
                    st.write(
                        f"Document `{source.get('document_id', 'unknown')}`, "
                        f"chunk `{source.get('chunk_index', '?')}`"
                    )

    booking = payload.get("booking")
    if isinstance(booking, dict):
        with st.expander("Booking state"):
            st.json(booking)


def main() -> None:
    st.title("Trustworthy AI")
    st.caption("Document questions with input and output guardrails")

    with st.sidebar:
        st.header("Connection")
        default_api_url = os.getenv("TRUSTWORTHY_AI_API_URL", "http://localhost:8000")
        api_url = st.text_input("FastAPI URL", value=default_api_url)
        if "session_id" not in st.session_state:
            st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        session_id = st.text_input("Session ID", value=st.session_state.session_id)
        st.session_state.session_id = session_id
        if st.button("New conversation", use_container_width=True):
            st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_response(message["content"])
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about the uploaded documents")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking and answering..."):
            try:
                payload = _post_chat(api_url, session_id, prompt)
            except httpx.HTTPStatusError as exc:
                st.error(f"The API returned HTTP {exc.response.status_code}.")
                return
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the API: {exc}")
                return

            _render_response(payload)
            st.session_state.messages.append({"role": "assistant", "content": payload})


if __name__ == "__main__":
    main()

