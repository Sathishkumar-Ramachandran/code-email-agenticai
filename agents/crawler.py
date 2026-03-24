"""CrawlerAgent — fetches and extracts content from a company website.

Strategy:
  1. Check robots.txt (fail-safe: allow if unreachable).
  2. Fetch with requests + trafilatura.
  3. If extraction fails or yields empty content, auto-fallback to Playwright.
  4. Respect rate-limit delay between requests.
"""
from __future__ import annotations

import concurrent.futures
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests
import trafilatura
import urllib3
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage

from agents.state import EmailState
from config import Settings

_BOT_UA = "ColdEmailResearchBot/1.0 (non-commercial research; contact: admin@example.com)"


# ── robots.txt helper ─────────────────────────────────────────────────────────

def _check_robots(url: str, verify_ssl: bool = True) -> bool:
    """Return True if crawling *url* is permitted by the site's robots.txt."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(robots_url, headers={"User-Agent": _BOT_UA},
                            timeout=10, verify=verify_ssl)
        resp.raise_for_status()
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp.can_fetch(_BOT_UA, url)
    except Exception:
        # Cannot reach robots.txt — fail-open so valid sites aren't blocked.
        return True


# ── fetchers ──────────────────────────────────────────────────────────────────

def _fetch_with_requests(url: str, timeout: int, verify_ssl: bool = True) -> tuple[str, str]:
    """Return (extracted_text, raw_html) using requests + trafilatura."""
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    headers = {"User-Agent": _BOT_UA}
    resp = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
    resp.raise_for_status()
    html = resp.text
    text = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
    )
    return (text or ""), html


def _playwright_available() -> bool:
    """Check whether Playwright + browser binaries are actually usable."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        # Actually test that browser binaries exist by examining executable path
        import subprocess
        result = subprocess.run(
            ["python", "-c", "from playwright.sync_api import sync_playwright; p=sync_playwright().start(); p.stop()"],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


# Cache the result so we don't re-check every call
_PW_OK: bool | None = None


def _is_playwright_ok() -> bool:
    global _PW_OK
    if _PW_OK is None:
        _PW_OK = _playwright_available()
    return _PW_OK


def _fetch_with_playwright(url: str, timeout: int) -> tuple[str, str]:
    """Return (extracted_text, raw_html) using a headless Chromium browser.

    Runs in a separate thread to avoid conflicts with Streamlit's
    (or any host framework's) running asyncio event loop on Windows.
    """
    if not _is_playwright_ok():
        raise RuntimeError("Playwright is not installed or browser binaries are missing")

    def _run() -> tuple[str, str]:
        from playwright.sync_api import sync_playwright  # lazy import
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=_BOT_UA)
            page.goto(url, timeout=timeout * 1_000, wait_until="domcontentloaded")
            page.wait_for_load_state("networkidle", timeout=timeout * 1_000)
            html = page.content()
            browser.close()
        text = trafilatura.extract(html, include_tables=True, no_fallback=False) or ""
        return text, html

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result(timeout=timeout + 30)


def _extract_title(html: str) -> str:
    """Best-effort page title from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    # 1. <title> tag
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # 2. Open Graph title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    # 3. <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return ""


# ── LangGraph node ────────────────────────────────────────────────────────────

def _normalise_url(url: str) -> str:
    """Ensure the URL has a scheme (default to https)."""
    url = url.strip()
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
    return url


def crawl_node(state: EmailState, settings: Settings) -> dict:
    """LangGraph node: validate URL, check robots.txt, crawl, extract text."""
    url = _normalise_url(state["url"])

    # ── robots.txt check ──────────────────────────────────────────────────────
    if not _check_robots(url, verify_ssl=settings.verify_ssl):
        msg = f"robots.txt disallows crawling {url}"
        return {
            "robots_allowed": False,
            "error": msg,
            "messages": [SystemMessage(content=f"[Crawler] BLOCKED — {msg}")],
        }

    # ── polite delay ──────────────────────────────────────────────────────────
    time.sleep(settings.rate_limit_delay)

    try:
        html = ""
        pw_ok = _is_playwright_ok()

        if settings.use_playwright and pw_ok:
            text, html = _fetch_with_playwright(url, settings.crawl_timeout)
        else:
            try:
                text, html = _fetch_with_requests(url, settings.crawl_timeout, verify_ssl=settings.verify_ssl)
                # Auto-fallback when requests yields very little content
                if len(text.strip()) < 200 and pw_ok:
                    text, html = _fetch_with_playwright(url, settings.crawl_timeout)
            except Exception:
                if pw_ok:
                    text, html = _fetch_with_playwright(url, settings.crawl_timeout)
                else:
                    raise

        title = _extract_title(html)
        # Truncate to avoid bloating the LLM context window
        raw_content = text.strip()[:8_000]

        return {
            "robots_allowed": True,
            "raw_content": raw_content,
            "page_title": title,
            "messages": [
                SystemMessage(
                    content=(
                        f"[Crawler] Extracted {len(raw_content)} chars from {url} "
                        f"(title: '{title}')"
                    )
                )
            ],
        }

    except Exception as exc:
        return {
            "robots_allowed": True,
            "raw_content": "",
            "page_title": "",
            "error": f"Crawl failed: {exc}",
            "messages": [SystemMessage(content=f"[Crawler] ERROR crawling {url}: {exc}")],
        }
