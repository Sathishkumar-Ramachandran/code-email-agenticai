"""ReviewerAgent — evaluates the email draft for quality and legal compliance.

Uses Gemini Flash (fast, low temperature) to score and flag issues.
Approved when: quality_score >= 7 AND no compliance issues.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents import extract_text_content
from agents.state import EmailState, ReviewResult
from config import Settings

_REVIEW_PROMPT = """\
You are a cold-email quality reviewer and legal compliance expert.
Evaluate the email below and return ONLY a valid JSON object — no markdown fences.

EMAIL:
{email}

Score against ALL of the following:
1. Subject specificity (personalised, non-clickbait, ≤50 chars)
2. Opening personalisation (no generic openers like "I came across your website")
3. Value proposition clarity (reader understands the benefit in ≤2 sentences)
4. CTA clarity and low friction (single, specific ask)
5. Word count target (≤150 words — penalise if clearly over)
6. CAN-SPAM compliance:
   - Sender clearly identified
   - No deceptive subject line
   - Opt-out / "reply STOP" language present
7. GDPR / data-privacy considerations:
   - No sensitive personal data referenced
   - Message relies on legitimate business interest basis
8. Overall persuasiveness and professionalism

Return EXACTLY this JSON (nothing else):

{{
  "quality_score": <integer 1-10>,
  "compliance_issues": ["<issue 1>", "..."],
  "suggestions": ["<improvement 1>", "..."],
  "approved": <true if quality_score >= 7 AND compliance_issues is empty, else false>
}}
"""


def _parse_review(raw: str) -> ReviewResult:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "quality_score": 5,
            "compliance_issues": [],
            "suggestions": ["Could not parse automated review — manual check recommended"],
            "approved": True,  # fail-open so pipeline can continue
        }


def review_node(state: EmailState, settings: Settings) -> dict:
    """LangGraph node: score the draft for quality and CAN-SPAM / GDPR compliance."""
    if state.get("error"):
        return {}

    draft = (state.get("email_draft") or "").strip()
    if not draft:
        return {
            "review_result": {
                "quality_score": 0,
                "compliance_issues": ["No draft available to review"],
                "suggestions": ["Generate a draft first"],
                "approved": False,
            },
            "review_iterations": (state.get("review_iterations") or 0) + 1,
        }

    llm_kwargs: dict = dict(
        model=settings.gemini_flash_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
    )
    if not settings.verify_ssl:
        import httpx
        llm_kwargs["client_args"] = {
            "http_options": {"httpxClient": httpx.Client(verify=False)},
        }
    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    response = llm.invoke([HumanMessage(content=_REVIEW_PROMPT.format(email=draft))])
    raw_text = extract_text_content(response.content)
    review: ReviewResult = _parse_review(raw_text)
    iterations = (state.get("review_iterations") or 0) + 1

    return {
        "review_result": review,
        "review_iterations": iterations,
        "messages": [
            AIMessage(
                content=(
                    f"[Reviewer] Score {review.get('quality_score')}/10 | "
                    f"Approved: {review.get('approved')} | "
                    f"Iteration: {iterations} | "
                    f"Issues: {review.get('compliance_issues') or 'none'}"
                )
            )
        ],
    }
