"""Unit tests for agents/drafter.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.drafter import _build_feedback, draft_node
from config import Settings


# ── _build_feedback ───────────────────────────────────────────────────────────

class TestBuildFeedback:
    def test_no_feedback_returns_fresh_draft_message(self):
        state = {"review_result": {}, "human_feedback": ""}
        assert "fresh draft" in _build_feedback(state)

    def test_includes_ai_suggestions(self):
        state = {
            "review_result": {
                "approved": False,
                "suggestions": ["Be more specific"],
                "compliance_issues": [],
            },
            "human_feedback": "",
        }
        result = _build_feedback(state)
        assert "Be more specific" in result

    def test_includes_compliance_issues(self):
        state = {
            "review_result": {
                "approved": False,
                "suggestions": [],
                "compliance_issues": ["Missing opt-out"],
            },
            "human_feedback": "",
        }
        result = _build_feedback(state)
        assert "Missing opt-out" in result

    def test_includes_human_feedback(self):
        state = {
            "review_result": {},
            "human_feedback": "Make it shorter",
        }
        result = _build_feedback(state)
        assert "Make it shorter" in result

    def test_combines_all_feedback(self):
        state = {
            "review_result": {
                "approved": False,
                "suggestions": ["Fix CTA"],
                "compliance_issues": ["No sender ID"],
            },
            "human_feedback": "Also change the tone",
        }
        result = _build_feedback(state)
        assert "Fix CTA" in result
        assert "No sender ID" in result
        assert "Also change the tone" in result


# ── draft_node ────────────────────────────────────────────────────────────────

class TestDraftNode:
    def _settings(self) -> Settings:
        return Settings(google_api_key="test-key")

    def test_returns_empty_when_error_set(self):
        state = {"error": "crawl failed"}
        result = draft_node(state, self._settings())
        assert result == {}

    def test_produces_draft_from_fresh_state(self):
        mock_msg = MagicMock()
        mock_msg.content = "SUBJECT: Quick question\n\nHi there, ..."

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg

        state = {
            "url": "https://example.com",
            "sender_info": {
                "name": "Alice",
                "role": "Sales Lead",
                "company": "Acme",
                "value_proposition": "Cut churn by 30%",
                "website": None,
            },
            "company_info": {
                "name": "TargetCo",
                "industry": "SaaS",
                "products_services": ["CRM"],
                "pain_points": ["churn"],
                "recent_news": [],
                "tone": "professional",
                "decision_makers": ["CTO"],
                "unique_insights": "Just raised Series B.",
            },
            "email_draft": "",
            "review_result": {},
            "human_feedback": "",
            "error": None,
        }

        with patch("agents.drafter.ChatGoogleGenerativeAI", return_value=mock_llm):
            result = draft_node(state, self._settings())

        assert "SUBJECT" in result["email_draft"]
        assert result["human_feedback"] == ""
        assert len(result["messages"]) == 1

    def test_revision_uses_revise_prompt(self):
        mock_msg = MagicMock()
        mock_msg.content = "SUBJECT: Revised\n\nRevised body"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg

        state = {
            "sender_info": {"name": "A", "role": "B", "company": "C", "value_proposition": "D", "website": None},
            "company_info": {"name": "X", "industry": "Y", "products_services": [], "pain_points": [], "recent_news": [], "tone": "casual", "decision_makers": [], "unique_insights": ""},
            "email_draft": "SUBJECT: Old\n\nOld body",
            "review_result": {"approved": False, "suggestions": ["Fix CTA"], "compliance_issues": []},
            "human_feedback": "",
            "error": None,
        }

        with patch("agents.drafter.ChatGoogleGenerativeAI", return_value=mock_llm):
            result = draft_node(state, self._settings())

        # The LLM should have received the revise prompt (existing draft + feedback)
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "ORIGINAL EMAIL" in prompt_text or "FEEDBACK" in prompt_text
        assert result["email_draft"] == "SUBJECT: Revised\n\nRevised body"
