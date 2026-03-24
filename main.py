"""CLI entry point for the cold-email multi-agent system.

Usage examples
──────────────
Interactive:
    python main.py https://example.com

Non-interactive (CI / scripts):
    python main.py https://example.com \\
        --sender-name "Alice" \\
        --sender-role "Head of Sales" \\
        --sender-company "Acme" \\
        --value-prop "We cut churn by 30 %% in 90 days"
"""
from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any

from agents.graph import build_graph
from agents.state import EmailState, SenderInfo
from agents import extract_text_content
from config import Settings


# ── helpers ───────────────────────────────────────────────────────────────────

def _empty_state(url: str, sender_info: SenderInfo) -> EmailState:
    return {
        "url": url,
        "sender_info": sender_info,
        "robots_allowed": True,
        "raw_content": "",
        "page_title": "",
        "company_info": {},           # type: ignore[typeddict-item]
        "email_draft": "",
        "review_result": {},          # type: ignore[typeddict-item]
        "review_iterations": 0,
        "human_feedback": "",
        "human_approved": False,
        "final_email": "",
        "error": None,
        "messages": [],
    }


def _print_stream_chunk(chunk: dict[str, Any]) -> None:
    msgs = chunk.get("messages", [])
    if msgs:
        content = extract_text_content(msgs[-1].content)
        print(f"  {content}")


def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _collect_sender_info_interactively() -> SenderInfo:
    print()
    _hr("─")
    print("  Sender Information")
    _hr("─")
    return {
        "name": input("  Your name              : ").strip(),
        "role": input("  Your role / title      : ").strip(),
        "company": input("  Your company           : ").strip(),
        "value_proposition": input("  Value proposition (1-2s): ").strip(),
        "website": input("  Your website (optional): ").strip() or None,
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(url: str, sender_info: SenderInfo, settings: Settings) -> int:
    """Run the full pipeline. Returns 0 on success, 1 on error."""
    graph = build_graph(settings)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    _hr("═")
    print(f"  Cold Email AI  |  target: {url}")
    _hr("═")

    # ── Phase 1: run pipeline until the human-approval interrupt ─────────────
    for chunk in graph.stream(_empty_state(url, sender_info), config, stream_mode="values"):
        _print_stream_chunk(chunk)

    state = graph.get_state(config)

    # Pipeline may have ended early (crawl error, robots block, etc.)
    if not state.next:
        err = state.values.get("error")
        if err:
            print(f"\n[ERROR] {err}")
            return 1
        # Completed without human approval (shouldn't normally happen)
        _print_final_email(state.values.get("final_email", ""))
        return 0

    # ── Phase 2: human-in-the-loop loop ─────────────────────────────────────
    while state.next and "human_approval" in state.next:
        draft = state.values.get("email_draft", "")
        review = state.values.get("review_result") or {}

        _hr("═")
        print("  HUMAN REVIEW REQUIRED")
        _hr("═")
        _hr("─")
        print("  Draft Email")
        _hr("─")
        print(draft)
        _hr("─")
        score = review.get("quality_score", "?")
        approved_by_ai = review.get("approved", False)
        print(f"  AI Score : {score}/10   Auto-approved: {approved_by_ai}")
        issues = review.get("compliance_issues") or []
        if issues:
            print(f"  Compliance issues: {', '.join(issues)}")
        suggestions = review.get("suggestions") or []
        if suggestions:
            print("  Suggestions:")
            for s in suggestions:
                print(f"    • {s}")
        _hr("─")

        choice = input("  [a]pprove / [r]evise / [e]dit+approve : ").strip().lower()

        if choice == "a":
            graph.update_state(config, {"human_approved": True, "human_feedback": ""})
        elif choice == "e":
            print("  Paste your edited email (type END on a new line when done):")
            lines: list[str] = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            edited = "\n".join(lines)
            graph.update_state(
                config,
                {"email_draft": edited, "human_approved": True, "human_feedback": ""},
            )
        else:
            feedback = input("  Revision notes: ").strip()
            graph.update_state(
                config, {"human_approved": False, "human_feedback": feedback}
            )

        # Resume pipeline
        for chunk in graph.stream(None, config, stream_mode="values"):
            _print_stream_chunk(chunk)

        state = graph.get_state(config)

    # ── Final output ─────────────────────────────────────────────────────────
    err = state.values.get("error")
    if err:
        print(f"\n[ERROR] {err}")
        return 1

    _print_final_email(state.values.get("final_email", ""))
    return 0


def _print_final_email(email: str) -> None:
    if not email:
        print("\n[!] No final email was produced.")
        return
    _hr("═")
    print("  FINAL EMAIL")
    _hr("═")
    print(email)
    _hr("═")


# ── CLI argument parser ───────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cold Email AI — multi-agent cold email generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("url", nargs="?", help="Target company website URL")
    p.add_argument("--sender-name", metavar="NAME")
    p.add_argument("--sender-role", metavar="ROLE")
    p.add_argument("--sender-company", metavar="COMPANY")
    p.add_argument("--value-prop", metavar="TEXT")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    try:
        settings = Settings()
    except Exception as exc:
        print(f"[ERROR] Configuration problem: {exc}")
        print("Create a .env file from .env.example and set GOOGLE_API_KEY.")
        sys.exit(1)

    settings.configure_ssl()
    settings.configure_langsmith()

    url = args.url or input("Target company URL: ").strip()
    if not url:
        print("[ERROR] URL is required.")
        sys.exit(1)

    has_all_sender = all(
        [args.sender_name, args.sender_role, args.sender_company, args.value_prop]
    )
    if has_all_sender:
        sender_info: SenderInfo = {
            "name": args.sender_name,
            "role": args.sender_role,
            "company": args.sender_company,
            "value_proposition": args.value_prop,
            "website": None,
        }
    else:
        sender_info = _collect_sender_info_interactively()

    sys.exit(run(url, sender_info, settings))


if __name__ == "__main__":
    main()
