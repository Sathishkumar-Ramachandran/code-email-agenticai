"""Shared pytest fixtures for the cold-email test suite."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from config import Settings


# ── Settings fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def settings() -> Settings:
    return Settings(google_api_key="test-api-key-1234")


# ── Minimal TypedDict state helpers ───────────────────────────────────────────

@pytest.fixture
def sender_info() -> dict:
    return {
        "name": "Alice",
        "role": "Head of Sales",
        "company": "Acme",
        "value_proposition": "We cut churn by 30% in 90 days.",
        "website": "https://acme.com",
    }


@pytest.fixture
def company_info() -> dict:
    return {
        "name": "TechCorp",
        "industry": "SaaS",
        "products_services": ["CRM platform", "Analytics dashboard"],
        "pain_points": ["high customer churn", "poor data visibility"],
        "recent_news": ["Launched v2.0 of their platform"],
        "tone": "professional",
        "decision_makers": ["CTO", "VP Engineering"],
        "unique_insights": "TechCorp just released a major platform update.",
    }


@pytest.fixture
def base_state(sender_info: dict, company_info: dict) -> dict:
    return {
        "url": "https://techcorp.example.com",
        "sender_info": sender_info,
        "robots_allowed": True,
        "raw_content": "TechCorp builds amazing SaaS products for enterprise customers.",
        "page_title": "TechCorp — Enterprise SaaS",
        "company_info": company_info,
        "email_draft": "",
        "review_result": {},
        "review_iterations": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_email": "",
        "error": None,
        "messages": [],
    }


# ── LLM mock factory ─────────────────────────────────────────────────────────

def make_llm_mock(response_text: str) -> MagicMock:
    """Return a MagicMock that behaves like a ChatGoogleGenerativeAI instance."""
    mock = MagicMock()
    ai_message = MagicMock()
    ai_message.content = response_text
    mock.invoke.return_value = ai_message
    return mock
