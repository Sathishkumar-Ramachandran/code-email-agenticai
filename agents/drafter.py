"""DrafterAgent — writes or revises the cold email using Gemini Flash.

On the first call it writes a fresh draft.
On subsequent calls (when review_result.approved is False, or after human feedback)
it revises the existing draft using targeted feedback.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agents import extract_text_content
from agents.state import EmailState
from config import Settings

_DRAFT_PROMPT = """\
You are an elite B2B cold-email copywriter.
Write a compelling, highly personalised cold email using the intelligence below.

── SENDER ─────────────────────────────────────────────────────────────────────
Name            : {sender_name}
Role            : {sender_role}
Company         : {sender_company}
Value Prop      : {value_prop}

── PROSPECT INTELLIGENCE ──────────────────────────────────────────────────────
Company         : {prospect_name}
Industry        : {industry}
Products/Svcs   : {products}
Pain Points     : {pain_points}
Communication   : {tone}
Decision Makers : {decision_makers}
Unique Insight  : {unique_insights}

── AI / HUMAN FEEDBACK (apply if present) ─────────────────────────────────────
{feedback}

── EMAIL WRITING RULES ────────────────────────────────────────────────────────
• Subject : specific + personalised, under 50 chars, zero clickbait
• Opening : reference a concrete fact about the prospect (never "I came across your website")
• Body    : 3–5 short sentences; bridge their pain point to your value proposition
• CTA     : one low-friction ask (15-min call, brief reply, a single question)
• Closing : professional sign-off with sender name, role, company
• Tone    : match the prospect's communication style ({tone})
• Length  : ≤150 words total
• Compliance: include one-line opt-out hint, no misleading statements (CAN-SPAM / GDPR)

Return the email in EXACTLY this format (no extra text):

SUBJECT: <subject line>

<full email body including sign-off>
"""

_REVISE_PROMPT = """\
You are revising the cold email below to address specific feedback.

── ORIGINAL EMAIL ─────────────────────────────────────────────────────────────
{original}

── FEEDBACK TO ADDRESS ────────────────────────────────────────────────────────
{feedback}

Apply every point of feedback and return the improved email in the EXACT same format:

SUBJECT: <subject line>

<full email body including sign-off>
"""


def _build_feedback(state: EmailState) -> str:
    parts: list[str] = []

    review = state.get("review_result") or {}
    if not review.get("approved", True):
        if review.get("suggestions"):
            parts.append("AI suggestions: " + "; ".join(review["suggestions"]))
        if review.get("compliance_issues"):
            parts.append("Compliance fixes needed: " + "; ".join(review["compliance_issues"]))

    human_fb = (state.get("human_feedback") or "").strip()
    if human_fb:
        parts.append(f"Human reviewer: {human_fb}")

    return "\n".join(parts) if parts else "None — write a fresh draft."


def draft_node(state: EmailState, settings: Settings) -> dict:
    """LangGraph node: produce or revise the cold-email draft."""
    if state.get("error"):
        return {}

    llm_kwargs: dict = dict(
        model=settings.gemini_flash_model,
        google_api_key=settings.google_api_key,
        temperature=0.7,
    )
    if not settings.verify_ssl:
        import httpx
        llm_kwargs["client_args"] = {
            "http_options": {"httpxClient": httpx.Client(verify=False)},
        }
    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    sender = state.get("sender_info") or {}
    company = state.get("company_info") or {}
    existing_draft = (state.get("email_draft") or "").strip()
    feedback = _build_feedback(state)

    if existing_draft and "None — write a fresh draft." not in feedback:
        prompt = _REVISE_PROMPT.format(original=existing_draft, feedback=feedback)
    else:
        prompt = _DRAFT_PROMPT.format(
            sender_name=sender.get("name", ""),
            sender_role=sender.get("role", ""),
            sender_company=sender.get("company", ""),
            value_prop=sender.get("value_proposition", ""),
            prospect_name=company.get("name", "the company"),
            industry=company.get("industry", ""),
            products=", ".join(company.get("products_services", [])),
            pain_points=", ".join(company.get("pain_points", [])),
            tone=company.get("tone", "professional"),
            decision_makers=", ".join(company.get("decision_makers", [])),
            unique_insights=company.get("unique_insights", ""),
            feedback=feedback,
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    draft = extract_text_content(response.content).strip()

    return {
        "email_draft": draft,
        "human_feedback": "",   # clear after use
        "messages": [
            AIMessage(
                content=f"[Drafter] Draft ready ({len(draft.split())} words)"
            )
        ],
    }
