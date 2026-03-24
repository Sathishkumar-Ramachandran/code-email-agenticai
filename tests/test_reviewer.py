"""Unit tests for agents/reviewer.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.reviewer import _parse_review, review_node
from config import Settings


# ── _parse_review ─────────────────────────────────────────────────────────────

class TestParseReview:
    def test_parses_valid_json(self):
        data = {
            "quality_score": 8,
            "compliance_issues": [],
            "suggestions": ["Minor tone tweak"],
            "approved": True,
        }
        result = _parse_review(json.dumps(data))
        assert result["quality_score"] == 8
        assert result["approved"] is True

    def test_strips_markdown_fences(self):
        inner = '{"quality_score": 6, "compliance_issues": ["no opt-out"], "suggestions": [], "approved": false}'
        raw = f"```json\n{inner}\n```"
        result = _parse_review(raw)
        assert result["quality_score"] == 6
        assert result["approved"] is False

    def test_fallback_on_invalid_json(self):
        result = _parse_review("This is not JSON!")
        assert result["quality_score"] == 5
        assert result["approved"] is True  # fail-open


# ── review_node ───────────────────────────────────────────────────────────────

class TestReviewNode:
    def _settings(self) -> Settings:
        return Settings(google_api_key="test-key")

    def test_returns_empty_when_error_set(self):
        state = {"error": "something broke"}
        result = review_node(state, self._settings())
        assert result == {}

    def test_returns_not_approved_when_no_draft(self):
        state = {"error": None, "email_draft": "", "review_iterations": 0}
        result = review_node(state, self._settings())
        assert result["review_result"]["approved"] is False
        assert result["review_result"]["quality_score"] == 0
        assert result["review_iterations"] == 1

    def test_reviews_draft_and_returns_result(self):
        review_json = json.dumps({
            "quality_score": 9,
            "compliance_issues": [],
            "suggestions": [],
            "approved": True,
        })
        mock_msg = MagicMock()
        mock_msg.content = review_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg

        state = {
            "error": None,
            "email_draft": "SUBJECT: Test\n\nHello World",
            "review_iterations": 1,
        }

        with patch("agents.reviewer.ChatGoogleGenerativeAI", return_value=mock_llm):
            result = review_node(state, self._settings())

        assert result["review_result"]["quality_score"] == 9
        assert result["review_result"]["approved"] is True
        assert result["review_iterations"] == 2
        assert len(result["messages"]) == 1

    def test_increments_review_iterations(self):
        review_json = json.dumps({
            "quality_score": 4,
            "compliance_issues": ["Missing opt-out"],
            "suggestions": ["Add unsubscribe"],
            "approved": False,
        })
        mock_msg = MagicMock()
        mock_msg.content = review_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg

        state = {
            "error": None,
            "email_draft": "SUBJECT: Bad\n\nBad email",
            "review_iterations": 2,
        }

        with patch("agents.reviewer.ChatGoogleGenerativeAI", return_value=mock_llm):
            result = review_node(state, self._settings())

        assert result["review_iterations"] == 3
