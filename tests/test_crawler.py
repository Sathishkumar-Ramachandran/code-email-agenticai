"""Unit tests for agents/crawler.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.crawler import (
    _check_robots,
    _extract_title,
    _fetch_with_requests,
    crawl_node,
)
from config import Settings


# ── _check_robots ─────────────────────────────────────────────────────────────

class TestCheckRobots:
    def test_allows_when_robots_unreachable(self):
        with patch("urllib.robotparser.RobotFileParser.read", side_effect=Exception("timeout")):
            assert _check_robots("https://example.com/page") is True

    def test_allows_when_permitted(self):
        with patch("urllib.robotparser.RobotFileParser.read"):
            with patch("urllib.robotparser.RobotFileParser.can_fetch", return_value=True):
                assert _check_robots("https://example.com/page") is True

    def test_blocks_when_disallowed(self):
        with patch("urllib.robotparser.RobotFileParser.read"):
            with patch("urllib.robotparser.RobotFileParser.can_fetch", return_value=False):
                assert _check_robots("https://example.com/page") is False


# ── _extract_title ────────────────────────────────────────────────────────────

class TestExtractTitle:
    def test_extracts_title_tag(self):
        html = "<html><head><title>My Company</title></head></html>"
        assert _extract_title(html) == "My Company"

    def test_extracts_og_title_when_no_title_tag(self):
        html = '<html><head><meta property="og:title" content="OG Title"/></head></html>'
        assert _extract_title(html) == "OG Title"

    def test_extracts_h1_as_fallback(self):
        html = "<html><body><h1> About Us </h1></body></html>"
        assert _extract_title(html) == "About Us"

    def test_returns_empty_string_when_no_title(self):
        html = "<html><body><p>No title here</p></body></html>"
        assert _extract_title(html) == ""


# ── crawl_node ────────────────────────────────────────────────────────────────

class TestCrawlNode:
    def _make_settings(self) -> Settings:
        return Settings(google_api_key="test-key", rate_limit_delay=0.0, crawl_timeout=5)

    def _base_state(self) -> dict:
        return {
            "url": "https://example.com",
            "sender_info": {},
            "robots_allowed": True,
            "raw_content": "",
            "page_title": "",
            "company_info": {},
            "email_draft": "",
            "review_result": {},
            "review_iterations": 0,
            "human_feedback": "",
            "human_approved": False,
            "final_email": "",
            "error": None,
            "messages": [],
        }

    def test_blocks_when_robots_disallowed(self):
        state = self._base_state()
        settings = self._make_settings()

        with patch("agents.crawler._check_robots", return_value=False):
            result = crawl_node(state, settings)

        assert result["robots_allowed"] is False
        assert result["error"] is not None
        assert "robots.txt" in result["error"]
        assert len(result["messages"]) == 1

    def test_successful_crawl_with_requests(self):
        state = self._base_state()
        settings = self._make_settings()

        fake_html = "<html><head><title>Acme Corp</title></head><body>Great products</body></html>"
        fake_text = "A " * 300  # > 200 chars so playwright not triggered

        with patch("agents.crawler._check_robots", return_value=True), \
             patch("agents.crawler._fetch_with_requests", return_value=(fake_text, fake_html)):
            result = crawl_node(state, settings)

        assert result["robots_allowed"] is True
        assert result["error"] is None
        assert result["page_title"] == "Acme Corp"
        assert len(result["raw_content"]) > 0

    def test_falls_back_to_playwright_when_content_short(self):
        state = self._base_state()
        settings = self._make_settings()

        short_text = "short"
        full_text = "Enterprise content " * 200
        fake_html = "<html><title>Company</title></html>"

        with patch("agents.crawler._check_robots", return_value=True), \
             patch("agents.crawler._fetch_with_requests", return_value=(short_text, fake_html)), \
             patch("agents.crawler._fetch_with_playwright", return_value=(full_text, fake_html)) as pw_mock:
            result = crawl_node(state, settings)

        pw_mock.assert_called_once()
        assert result["robots_allowed"] is True
        assert len(result["raw_content"]) > 0

    def test_crawl_uses_playwright_when_forced(self):
        state = self._base_state()
        settings = self._make_settings()
        settings.use_playwright = True  # type: ignore[assignment]

        full_text = "Playwright content " * 200
        fake_html = "<html><title>JS Site</title></html>"

        with patch("agents.crawler._check_robots", return_value=True), \
             patch("agents.crawler._fetch_with_playwright", return_value=(full_text, fake_html)) as pw_mock:
            result = crawl_node(state, settings)

        pw_mock.assert_called_once()
        assert result["page_title"] == "JS Site"

    def test_crawl_error_propagates(self):
        state = self._base_state()
        settings = self._make_settings()

        with patch("agents.crawler._check_robots", return_value=True), \
             patch("agents.crawler._fetch_with_requests", side_effect=Exception("Connection refused")), \
             patch("agents.crawler._fetch_with_playwright", side_effect=Exception("Playwright also failed")):
            result = crawl_node(state, settings)

        assert result["error"] is not None
        assert "Crawl failed" in result["error"]
        assert result["raw_content"] == ""

    def test_content_truncated_to_8000_chars(self):
        state = self._base_state()
        settings = self._make_settings()

        huge_text = "x" * 20_000
        fake_html = "<html><title>Big Page</title></html>"

        with patch("agents.crawler._check_robots", return_value=True), \
             patch("agents.crawler._fetch_with_requests", return_value=(huge_text, fake_html)):
            result = crawl_node(state, settings)

        assert len(result["raw_content"]) <= 8_000
