"""LangGraph state schema for the cold-email multi-agent pipeline."""
from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SenderInfo(TypedDict):
    name: str
    role: str
    company: str
    value_proposition: str
    website: Optional[str]


class CompanyInfo(TypedDict):
    name: str
    industry: str
    products_services: list[str]
    pain_points: list[str]
    recent_news: list[str]
    tone: str                  # "formal" | "casual" | "technical" | etc.
    decision_makers: list[str]
    unique_insights: str       # 1-2 sentence personalisation hook


class ReviewResult(TypedDict):
    quality_score: int         # 1-10
    compliance_issues: list[str]
    suggestions: list[str]
    approved: bool             # True when score >= 7 and no compliance issues


class EmailState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    url: str
    sender_info: SenderInfo

    # ── Crawler outputs ───────────────────────────────────────────────────────
    robots_allowed: bool
    raw_content: str
    page_title: str

    # ── Researcher output ─────────────────────────────────────────────────────
    company_info: CompanyInfo

    # ── Draft / review cycle ──────────────────────────────────────────────────
    email_draft: str
    review_result: ReviewResult
    review_iterations: int

    # ── Human-in-the-loop ─────────────────────────────────────────────────────
    human_feedback: str
    human_approved: bool

    # ── Pipeline output ───────────────────────────────────────────────────────
    final_email: str

    # ── Error propagation ─────────────────────────────────────────────────────
    error: Optional[str]

    # ── Agent message log (append-only via add_messages reducer) ─────────────
    messages: Annotated[list[AnyMessage], add_messages]
