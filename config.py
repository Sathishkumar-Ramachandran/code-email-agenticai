"""Application configuration using Pydantic Settings (v2)."""
from __future__ import annotations

import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    google_api_key: str = Field(..., description="Google AI Studio API key")
    gemini_pro_model: str = Field(
        default="gemini-3.1-pro-preview",
        description="Gemini Pro model for deep company analysis",
    )
    gemini_flash_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini Flash model for fast drafting and review",
    )

    # ── Crawler ──────────────────────────────────────────────────────────────
    crawl_timeout: int = Field(default=30, description="HTTP request timeout (seconds)")
    rate_limit_delay: float = Field(
        default=1.5, description="Polite delay between requests (seconds)"
    )
    use_playwright: bool = Field(
        default=False, description="Force Playwright for JS-heavy sites"
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates (set False behind corporate proxies)",
    )

    # ── Quality control ──────────────────────────────────────────────────────
    max_review_iterations: int = Field(
        default=3,
        description="Max auto-review loops before forcing human handoff",
    )

    # ── LangSmith observability ──────────────────────────────────────────────
    langsmith_api_key: str = Field(default="", description="LangSmith API key (optional)")
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_project: str = Field(
        default="cold-email-agenticai", description="LangSmith project name"
    )

    def configure_langsmith(self) -> None:
        """Set LangChain env vars when tracing is enabled."""
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", self.langsmith_api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", self.langsmith_project)

    def configure_ssl(self) -> None:
        """Disable SSL verification globally when behind a corporate proxy."""
        if not self.verify_ssl:
            # google-generativeai / httpx rely on GRPC_DEFAULT_SSL_ROOTS_FILE_PATH
            # and httpx's default SSL context.  Setting these env vars disables
            # verification at the transport layer.
            os.environ["GRPC_SSL_TARGET_NAME_OVERRIDE"] = "generativelanguage.googleapis.com"
            # httpx / urllib3 / aiohttp respect SSL_CERT_FILE; point to empty
            # is not ideal, so instead we patch ssl globally for this process.
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
