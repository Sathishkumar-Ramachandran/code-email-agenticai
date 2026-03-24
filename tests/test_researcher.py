"""Unit tests for agents/researcher.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.researcher import _parse_json_response, research_node
from config import Settings


# ── _parse_json_response ──────────────────────────────────────────────────────

class TestParseJsonResponse:
    def test_parses_valid_json(self):
        data = {
            "name": "Acme",
            "industry": "SaaS",
            "products_services": ["CRM"],
            "pain_points": ["churn"],
            "recent_news": ["Launched v2"],
            "tone": "professional",
            "decision_makers": ["CTO"],
            "unique_insights": "Acme just launched v2.",
        }
        result = _parse_json_response(json.dumps(data), "fallback")
        assert result["name"] == "Acme"
        assert result["industry"] == "SaaS"

    def test_strips_markdown_fences(self):
        inner = '{"name": "Corp", "industry": "Fintech", "products_services": [], "pain_points": [], "recent_news": [], "tone": "formal", "decision_makers": [], "unique_insights": "hook"}'
        raw = f"```json\n{inner}\n```"
        result = _parse_json_response(raw, "fallback")
        assert result["name"] == "Corp"

    def test_fallback_on_invalid_json(self):
        result = _parse_json_response("Not valid JSON at all!", "FallbackCo")
        assert result["name"] == "FallbackCo"
        assert result["industry"] == "Unknown"
        assert "Not valid JSON" in result["unique_insights"]


# ── research_node ─────────────────────────────────────────────────────────────

class TestResearchNode:
    def _settings(self) -> Settings:
        return Settings(google_api_key="test-key")

    def test_returns_empty_when_error_set(self):
        state = {"error": "previous failure"}
        result = research_node(state, self._settings())
        assert result == {}

    def test_invokes_llm_and_returns_company_info(self):
        company_json = json.dumps({
            "name": "TestCo",
            "industry": "HealthTech",
            "products_services": ["EHR system"],
            "pain_points": ["compliance costs"],
            "recent_news": ["FDA approval"],
            "tone": "formal",
            "decision_makers": ["CTO", "VP Eng"],
            "unique_insights": "Just got FDA approval for their new device.",
        })
        mock_msg = MagicMock()
        mock_msg.content = company_json

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg

        state = {
            "url": "https://testco.com",
            "page_title": "TestCo",
            "raw_content": "TestCo is a HealthTech company...",
            "error": None,
        }

        with patch("agents.researcher.ChatGoogleGenerativeAI", return_value=mock_llm):
            result = research_node(state, self._settings())

        assert result["company_info"]["name"] == "TestCo"
        assert result["company_info"]["industry"] == "HealthTech"
        assert len(result["messages"]) == 2
