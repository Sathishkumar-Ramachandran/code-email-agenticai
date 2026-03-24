"""ResearcherAgent — extracts structured B2B intelligence from crawled content.

Uses Gemini Pro (deep understanding model) with JSON function-calling style output.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents import extract_text_content
from agents.state import CompanyInfo, EmailState
from config import Settings

_RESEARCH_PROMPT = """\
You are a senior B2B sales-intelligence analyst.
Analyse the website content below and return structured intelligence that will be used
to personalise a cold outreach email.

Website URL : {url}
Page Title  : {title}
Content     :
{content}

Return ONLY a valid JSON object — no markdown fences, no extra text — with exactly these keys:

{{
  "name": "<official company name>",
  "industry": "<industry / sector>",
  "products_services": ["<product or service 1>", "..."],
  "pain_points": ["<likely business challenge 1>", "..."],
  "recent_news": ["<notable update / launch / milestone mentioned on the page>"],
  "tone": "<formal | casual | technical | innovative | mixed>",
  "decision_makers": ["<typical buyer title, e.g. CTO, VP Engineering>"],
  "unique_insights": "<1-2 sentence personalisation hook grounded in the page content>"
}}
"""


def _parse_json_response(raw: str, fallback_name: str) -> CompanyInfo:
    """Strip optional markdown fences and parse JSON; fall back gracefully."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] may start with "json\n"
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "name": fallback_name,
            "industry": "Unknown",
            "products_services": [],
            "pain_points": [],
            "recent_news": [],
            "tone": "professional",
            "decision_makers": ["CEO", "VP Sales", "Founder"],
            "unique_insights": text[:300],
        }


def research_node(state: EmailState, settings: Settings) -> dict:
    """LangGraph node: analyse company website and return CompanyInfo."""
    if state.get("error"):
        return {}

    llm_kwargs: dict = dict(
        model=settings.gemini_pro_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    if not settings.verify_ssl:
        import httpx
        llm_kwargs["client_args"] = {
            "http_options": {"httpxClient": httpx.Client(verify=False)},
        }
    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    prompt = _RESEARCH_PROMPT.format(
        url=state["url"],
        title=state.get("page_title", ""),
        content=state.get("raw_content", "")[:6_000],
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_text = extract_text_content(response.content)
    company_info = _parse_json_response(
        raw_text,
        fallback_name=state.get("page_title") or "the company",
    )

    return {
        "company_info": company_info,
        "messages": [
            HumanMessage(content=f"[Researcher] Analysing {state['url']}"),
            AIMessage(
                content=(
                    f"[Researcher] Company: {company_info.get('name')} | "
                    f"Industry: {company_info.get('industry')} | "
                    f"Insight: {company_info.get('unique_insights', '')[:120]}"
                )
            ),
        ],
    }
