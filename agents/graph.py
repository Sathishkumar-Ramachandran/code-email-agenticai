"""LangGraph StateGraph — wires all agents into a deterministic pipeline.

Graph topology
──────────────
START
  └─► crawl ──[error / robots blocked]──► END
         └─► research ──► draft ──► review
                                       ├─[approved OR max iterations]──► human_approval
                                       └─[not approved, iterations < max]──► draft  (loop)
                                                human_approval
                                                  ├─[human approved]──► finalize ──► END
                                                  └─[human rejected]──► draft  (loop)

Human-in-the-loop
─────────────────
The graph is compiled with interrupt_before=["human_approval"].
The pipeline pauses BEFORE running that node so the caller can:
  1. Read state.values["email_draft"] and state.values["review_result"].
  2. Update the state with the human's decision via graph.update_state(...).
  3. Resume by calling graph.stream(None, config).
The human_approval node itself is a lightweight pass-through.

Durable state / checkpointing
──────────────────────────────
MemorySaver is used for in-process persistence (suitable for dev/demo).
Swap it for SqliteSaver or a Redis/Postgres saver for production deployments.
"""
from __future__ import annotations

from functools import partial
from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.crawler import crawl_node
from agents.drafter import draft_node
from agents.researcher import research_node
from agents.reviewer import review_node
from agents.state import EmailState
from config import Settings


# ── Node factories (inject Settings without polluting the node signatures) ─────

def _make_crawl(settings: Settings):
    def node(state: EmailState) -> dict:
        return crawl_node(state, settings)
    node.__name__ = "crawl"
    return node


def _make_research(settings: Settings):
    def node(state: EmailState) -> dict:
        return research_node(state, settings)
    node.__name__ = "research"
    return node


def _make_draft(settings: Settings):
    def node(state: EmailState) -> dict:
        return draft_node(state, settings)
    node.__name__ = "draft"
    return node


def _make_review(settings: Settings):
    def node(state: EmailState) -> dict:
        return review_node(state, settings)
    node.__name__ = "review"
    return node


# ── Pass-through human-approval node ─────────────────────────────────────────

def human_approval_node(state: EmailState) -> dict:
    """No-op node; the pipeline is interrupted BEFORE this node executes.
    The caller updates state (human_approved, human_feedback) via update_state()
    and then resumes — this node just emits a log message.
    """
    approved = state.get("human_approved", False)
    feedback = state.get("human_feedback", "")
    return {
        "messages": [
            SystemMessage(
                content=(
                    f"[HumanApproval] Decision: {'APPROVED' if approved else 'REJECTED'}"
                    + (f" | Feedback: {feedback}" if feedback else "")
                )
            )
        ]
    }


def finalize_node(state: EmailState) -> dict:
    """Package the approved draft as the final deliverable."""
    return {
        "final_email": state.get("email_draft", ""),
        "messages": [
            SystemMessage(content="[Finalizer] Email approved and ready to send.")
        ],
    }


# ── Conditional routing ───────────────────────────────────────────────────────

def _route_after_crawl(
    state: EmailState,
) -> Literal["research", "__end__"]:
    if state.get("error") or not state.get("robots_allowed", True):
        return "__end__"
    if not state.get("raw_content", "").strip():
        return "__end__"
    return "research"


def _route_after_review(
    state: EmailState,
    max_iterations: int,
) -> Literal["human_approval", "draft", "__end__"]:
    if state.get("error"):
        return "__end__"
    review = state.get("review_result") or {}
    iterations = state.get("review_iterations", 0)
    if review.get("approved", False):
        return "human_approval"
    if iterations >= max_iterations:
        # Force hand-off to human after hitting the max auto-revision limit
        return "human_approval"
    return "draft"


def _route_after_human(
    state: EmailState,
) -> Literal["finalize", "draft"]:
    return "finalize" if state.get("human_approved", False) else "draft"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(settings: Settings | None = None):
    """Build and compile the cold-email StateGraph.

    Returns a compiled LangGraph app with MemorySaver checkpointing and
    an interrupt before the human_approval node.
    """
    if settings is None:
        settings = Settings()

    settings.configure_langsmith()

    builder = StateGraph(EmailState)

    # Nodes
    builder.add_node("crawl", _make_crawl(settings))
    builder.add_node("research", _make_research(settings))
    builder.add_node("draft", _make_draft(settings))
    builder.add_node("review", _make_review(settings))
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("finalize", finalize_node)

    # Edges
    builder.add_edge(START, "crawl")
    builder.add_conditional_edges(
        "crawl",
        _route_after_crawl,
        {"research": "research", "__end__": END},
    )
    builder.add_edge("research", "draft")
    builder.add_edge("draft", "review")
    builder.add_conditional_edges(
        "review",
        partial(_route_after_review, max_iterations=settings.max_review_iterations),
        {"human_approval": "human_approval", "draft": "draft", "__end__": END},
    )
    builder.add_conditional_edges(
        "human_approval",
        _route_after_human,
        {"finalize": "finalize", "draft": "draft"},
    )
    builder.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )
