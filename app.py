"""Streamlit UI for the cold-email multi-agent system.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from agents.graph import build_graph
from agents.state import EmailState, SenderInfo
from agents import extract_text_content
from config import Settings

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cold Email AI",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── cached graph (keyed on API key so swapping keys invalidates the cache) ────
@st.cache_resource(show_spinner=False)
def _get_graph(api_key: str):
    settings = Settings(google_api_key=api_key)
    settings.configure_ssl()
    settings.configure_langsmith()
    return build_graph(settings), settings


# ── session-state bootstrap ───────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "stage": "input",       # input | running | awaiting_approval | done | error
    "thread_id": None,
    "email_draft": "",
    "review_result": {},
    "final_email": "",
    "error": None,
    "log": [],
    "sender_info": None,
    "target_url": "",
}


def _init_session() -> None:
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_session() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ── sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> tuple[str, SenderInfo | None]:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        api_key: str = st.text_input(
            "Google API Key",
            type="password",
            key="api_key",
            help="Get yours at https://aistudio.google.com",
        )

        st.divider()
        st.markdown("## 👤 Sender Information")
        name = st.text_input("Your Name", key="s_name")
        role = st.text_input("Your Role / Title", key="s_role")
        company = st.text_input("Your Company", key="s_company")
        value_prop = st.text_area(
            "Value Proposition",
            key="s_value_prop",
            help="What unique value do you deliver? (1-2 sentences)",
            height=80,
        )
        website = st.text_input("Your Website (optional)", key="s_website")

    sender_info: SenderInfo | None = None
    if name and role and company and value_prop:
        sender_info = {
            "name": name,
            "role": role,
            "company": company,
            "value_proposition": value_prop,
            "website": website or None,
        }
    return api_key, sender_info


# ── activity log expander ─────────────────────────────────────────────────────
def _render_log() -> None:
    if st.session_state.log:
        with st.expander("📋 Agent Activity Log", expanded=False):
            for entry in st.session_state.log:
                st.markdown(f"- {entry}")


# ── stream helper ─────────────────────────────────────────────────────────────
def _collect_stream(graph, input_or_none, config: dict) -> None:
    """Run graph.stream and append messages to the session log."""
    for chunk in graph.stream(input_or_none, config, stream_mode="values"):
        msgs = chunk.get("messages", [])
        if msgs:
            content = extract_text_content(msgs[-1].content)
            st.session_state.log.append(content)


# ── stages ────────────────────────────────────────────────────────────────────

def _stage_input(api_key: str, sender_info: SenderInfo | None) -> None:
    col_url, col_btn = st.columns([3, 1])
    with col_url:
        url: str = st.text_input(
            "🌐 Target Company Website URL",
            placeholder="https://acme.com",
            key="target_url_input",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("🚀 Generate Email", type="primary", use_container_width=True)

    if generate:
        # Normalise URL — add scheme if missing
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url.strip()

        errors: list[str] = []
        if not api_key:
            errors.append("Google API Key is required (sidebar).")
        if not url:
            errors.append("Target URL is required.")
        if not sender_info:
            errors.append("Complete all sender fields in the sidebar.")
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            return

        st.session_state.target_url = url
        st.session_state.sender_info = sender_info
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.log = []
        st.session_state.stage = "running"
        st.rerun()


def _stage_running(api_key: str) -> None:
    st.info("⚙️ Agents are running… please wait.")

    try:
        graph, _ = _get_graph(api_key)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        initial: EmailState = {
            "url": st.session_state.target_url,
            "sender_info": st.session_state.sender_info,
            "robots_allowed": True,
            "raw_content": "",
            "page_title": "",
            "company_info": {},          # type: ignore[typeddict-item]
            "email_draft": "",
            "review_result": {},         # type: ignore[typeddict-item]
            "review_iterations": 0,
            "human_feedback": "",
            "human_approved": False,
            "final_email": "",
            "error": None,
            "messages": [],
        }

        with st.spinner("Running agents…"):
            _collect_stream(graph, initial, config)

        gstate = graph.get_state(config)

        if "human_approval" in (gstate.next or ()):
            st.session_state.email_draft = gstate.values.get("email_draft", "")
            st.session_state.review_result = gstate.values.get("review_result") or {}
            st.session_state.stage = "awaiting_approval"
        elif gstate.values.get("error"):
            st.session_state.error = gstate.values["error"]
            st.session_state.stage = "error"
        else:
            st.session_state.final_email = gstate.values.get("final_email", "")
            st.session_state.stage = "done"

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.stage = "error"

    st.rerun()


def _stage_awaiting_approval(api_key: str) -> None:
    st.subheader("👤 Human Review Required")
    _render_log()

    review: dict = st.session_state.review_result
    if review:
        score = review.get("quality_score", 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("AI Quality Score", f"{score} / 10")
        c2.metric(
            "Compliance Issues",
            len(review.get("compliance_issues") or []),
            delta_color="inverse",
        )
        c3.metric("AI Pre-approved", "✅ Yes" if review.get("approved") else "⚠️  No")

        if review.get("compliance_issues"):
            st.warning("**Compliance issues to fix:**\n- " + "\n- ".join(review["compliance_issues"]))

        if review.get("suggestions"):
            with st.expander("💡 AI Suggestions", expanded=True):
                for s in review["suggestions"]:
                    st.markdown(f"- {s}")

    st.markdown("### ✏️ Email Draft")
    edited: str = st.text_area(
        "Review and edit before approving:",
        value=st.session_state.email_draft,
        height=320,
        key="edited_draft",
    )

    st.markdown("---")
    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button("✅ Approve & Finalise", type="primary", use_container_width=True):
            try:
                graph, _ = _get_graph(api_key)
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                # Apply any manual edits made in the text area
                if edited != st.session_state.email_draft:
                    graph.update_state(config, {"email_draft": edited})
                graph.update_state(
                    config, {"human_approved": True, "human_feedback": ""}
                )
                with st.spinner("Finalising…"):
                    _collect_stream(graph, None, config)
                gstate = graph.get_state(config)
                st.session_state.final_email = gstate.values.get(
                    "final_email", edited
                )
                st.session_state.stage = "done"
                st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

    with col_reject:
        with st.form("revision_form"):
            feedback: str = st.text_area(
                "Revision Notes",
                placeholder="What should be changed?",
                height=100,
                key="revision_feedback",
            )
            submitted = st.form_submit_button("🔄 Request Revision", use_container_width=True)

        if submitted:
            if not feedback.strip():
                st.warning("Please enter revision feedback before submitting.")
            else:
                try:
                    graph, _ = _get_graph(api_key)
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    graph.update_state(
                        config,
                        {"human_approved": False, "human_feedback": feedback.strip()},
                    )
                    with st.spinner("Revising…"):
                        _collect_stream(graph, None, config)
                    gstate = graph.get_state(config)
                    if "human_approval" in (gstate.next or ()):
                        st.session_state.email_draft = gstate.values.get("email_draft", "")
                        st.session_state.review_result = gstate.values.get("review_result") or {}
                    else:
                        st.session_state.final_email = gstate.values.get("final_email", "")
                        st.session_state.stage = "done"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")


def _stage_done() -> None:
    _render_log()
    st.success("✅ Email finalised and ready to send!")

    raw: str = st.session_state.final_email
    st.markdown("### 📧 Final Email")

    lines = raw.split("\n", 2)
    if lines and lines[0].upper().startswith("SUBJECT:"):
        subject = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
        st.markdown(f"**Subject:** {subject}")
        st.divider()
        st.markdown(body)
    else:
        st.markdown(raw)

    st.divider()
    col_copy, col_download, col_restart = st.columns(3)
    with col_copy:
        st.code(raw, language="text")
    with col_download:
        st.download_button(
            "⬇️ Download as .txt",
            data=raw,
            file_name="cold_email.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_restart:
        if st.button("🔄 Start Over", use_container_width=True):
            _reset_session()
            st.rerun()


def _stage_error() -> None:
    _render_log()
    st.error(f"❌ Pipeline Error: {st.session_state.error}")
    if st.button("🔄 Start Over"):
        _reset_session()
        st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    st.title("✉️ Cold Email AI")
    st.caption("Multi-agent cold email generator powered by Google Gemini")

    api_key, sender_info = _render_sidebar()

    stage = st.session_state.stage

    if stage == "input":
        _stage_input(api_key, sender_info)
    elif stage == "running":
        _stage_running(api_key)
    elif stage == "awaiting_approval":
        _stage_awaiting_approval(api_key)
    elif stage == "done":
        _stage_done()
    elif stage == "error":
        _stage_error()


if __name__ == "__main__":
    main()
