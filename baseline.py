"""
Baseline inference script for the Support Ticket Triage environment.

Runs all 3 tasks using either:
  1. Gemini 2.0 Flash (if GEMINI_API_KEY is set)
  2. A deterministic rule-based agent (fallback for reproducibility)

Usage:
    # With Gemini
    export GEMINI_API_KEY=your_key_here
    python baseline.py

    # Deterministic fallback (no API key needed)
    python baseline.py --deterministic

    # Against a remote server
    python baseline.py --base-url https://your-space.hf.space
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:7860"
GEMINI_MODEL = "gemini-2.0-flash"

# ── Deterministic rule-based agent ──────────────────────────────────────────

# Maps keywords in ticket text to (category, priority, department)
_KEYWORD_RULES = [
    # High-priority security
    (["unauthorized", "security", "breach", "hack"],
     "security", "P1", "security"),
    # Legal / compliance
    (["gdpr", "legal", "erasure", "right to", "compliance"],
     "account", "P1", "legal"),
    # Production outage
    (["500 error", "production down", "api returning", "outage"],
     "it_support", "P1", "engineering"),
    # VIP / SLA
    (["sla", "enterprise premium", "premium customer"],
     "complaint", "P1", "engineering"),
    # Billing
    (["overcharged", "invoice", "charged", "refund", "payment"],
     "billing", "P2", "billing"),
    # Product complaints
    (["crash", "bug", "not working", "broken", "disappointed"],
     "complaint", "P2", "engineering"),
    # Sales
    (["demo", "enterprise plan", "pricing", "volume discount"],
     "sales", "P3", "sales"),
    # Password / login
    (["password", "log in", "login", "credentials", "can't log"],
     "it_support", "P3", "it_support"),
    # Multi-issue / general
    (["feature request", "dark mode", "three things"],
     "general", "P3", "customer_success"),
]

# Tickets whose blueprints include "escalate" in the required workflow
_ESCALATION_KEYWORDS = [
    "unauthorized", "gdpr", "legal", "sla", "premium",
    "production down", "500 error", "chargeback", "refund",
]


def _match_rules(text: str):
    """Match ticket text against keyword rules."""
    text_lower = text.lower()
    for keywords, category, priority, department in _KEYWORD_RULES:
        if any(kw in text_lower for kw in keywords):
            return category, priority, department
    return "general", "P3", "customer_success"


def _needs_escalation(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in _ESCALATION_KEYWORDS)


def deterministic_agent(tickets: List[Dict]) -> List[Dict[str, Any]]:
    """
    Generate a sequence of actions for all open tickets using rules.

    Returns a list of action dicts to execute sequentially.
    """
    actions = []
    # Sort by inferred priority (P1 first)
    ticket_infos = []
    for t in tickets:
        text = f"{t['subject']} {t['body']}"
        cat, prio, dept = _match_rules(text)
        escalate = _needs_escalation(text)
        ticket_infos.append((t, cat, prio, dept, escalate))

    # Sort by priority (P1 < P2 < P3 < P4)
    prio_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    ticket_infos.sort(key=lambda x: prio_order.get(x[2], 3))

    for t, cat, prio, dept, escalate in ticket_infos:
        tid = t["id"]
        # Classify
        actions.append({"ticket_id": tid, "action_type": "classify", "target": cat})
        # Prioritize
        actions.append({"ticket_id": tid, "action_type": "prioritize", "target": prio})
        # Route
        actions.append({"ticket_id": tid, "action_type": "route", "target": dept})
        # Escalate if warranted
        if escalate:
            actions.append({"ticket_id": tid, "action_type": "escalate", "target": "management"})
        # Respond (for medium/hard tickets)
        actions.append({
            "ticket_id": tid,
            "action_type": "respond",
            "message": f"We've received your ticket and it has been assigned to {dept}.",
        })
        # Resolve
        actions.append({"ticket_id": tid, "action_type": "resolve"})

    return actions


# ── Gemini-powered agent ───────────────────────────────────────────────────

def gemini_agent(tickets: List[Dict], api_key: str) -> List[Dict[str, Any]]:
    """
    Use Gemini 2.0 Flash to generate actions for all open tickets.

    Sends tickets as context and asks the model to produce a JSON array
    of actions for each ticket.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai not installed, falling back to deterministic.")
        return deterministic_agent(tickets)

    genai.configure(api_key=api_key)

    ticket_descriptions = "\n".join(
        f"- [{t['id']}] Subject: {t['subject']}\n  Body: {t['body']}\n  Sentiment: {t.get('customer_sentiment', 'unknown')}"
        for t in tickets
    )

    prompt = f"""You are an expert customer support triage AI agent. You must process the following support tickets by generating a JSON array of actions.

For EACH ticket, you must decide the correct sequence of actions. Available action types:
- "classify": Set the ticket category. target must be one of: it_support, billing, sales, complaint, security, account, general
- "prioritize": Set priority. target must be one of: P1, P2, P3, P4 (P1=critical, P4=low)
- "route": Route to department. target must be one of: it_support, billing, sales, customer_success, security, engineering, management, legal
- "escalate": Escalate to management/security. Only use for security breaches, legal issues, VIP complaints, production outages, or refund disputes.
- "respond": Send a response message. Include a "message" field.
- "resolve": Mark the ticket as resolved. Do this LAST, after classify+route at minimum.

IMPORTANT RULES:
1. Process higher-priority tickets first (P1 before P2, etc.)
2. Every ticket needs at minimum: classify → prioritize → route → resolve
3. Complex tickets (security, legal, VIP, outages) also need: escalate and respond
4. For "resolve", do NOT include a target — just the ticket_id and action_type

Here are the tickets:
{ticket_descriptions}

Respond with ONLY a JSON array of action objects. Each object has:
- "ticket_id": string
- "action_type": string
- "target": string (optional, required for classify/prioritize/route/escalate)
- "message": string (optional, for respond actions)

JSON array:"""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Extract JSON from response (handle ```json ... ``` blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        actions = json.loads(text)
        if isinstance(actions, list):
            return actions
        else:
            print("Gemini returned non-list, falling back to deterministic.")
            return deterministic_agent(tickets)

    except Exception as e:
        print(f"Gemini error: {e}, falling back to deterministic.")
        return deterministic_agent(tickets)


# ── Runner ──────────────────────────────────────────────────────────────────

def run_baseline(
    base_url: str = DEFAULT_BASE_URL,
    use_deterministic: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the baseline agent across all 3 tasks.

    Returns a dict with scores for each task and the average.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not use_deterministic:
        print("No GEMINI_API_KEY found. Using deterministic agent.")
        use_deterministic = True

    client = httpx.Client(base_url=base_url, timeout=30.0)
    results = {}

    for task_id in range(3):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Task {task_id} ({'easy' if task_id == 0 else 'medium' if task_id == 1 else 'hard'})")
            print(f"{'='*60}")

        # Reset
        resp = client.post(f"/reset?task_idx={task_id}")
        if resp.status_code != 200:
            print(f"  Reset failed: {resp.text}")
            results[f"task_{task_id}"] = {"score": 0.0, "error": resp.text}
            continue

        obs = resp.json()
        open_tickets = obs.get("open_tickets", [])

        if verbose:
            print(f"  Loaded {len(open_tickets)} ticket(s)")

        # Generate actions
        if use_deterministic:
            actions = deterministic_agent(open_tickets)
        else:
            actions = gemini_agent(open_tickets, api_key)

        if verbose:
            print(f"  Generated {len(actions)} action(s)")

        # Execute actions
        step_count = 0
        for action in actions:
            # Clean action for API
            payload = {
                "ticket_id": action.get("ticket_id", ""),
                "action_type": action.get("action_type", ""),
            }
            if action.get("target"):
                payload["target"] = action["target"]
            if action.get("message"):
                payload["message"] = action["message"]

            resp = client.post("/step", json=payload)
            if resp.status_code != 200:
                if verbose:
                    print(f"  Step error: {resp.text}")
                continue

            result = resp.json()
            step_count += 1

            if verbose:
                info = result.get("info", {})
                feedback = info.get("feedback", "")
                reward = result.get("reward", 0)
                print(f"  Step {step_count}: {payload['action_type']} on {payload['ticket_id']}"
                      f" → reward={reward:+.2f} | {feedback[:80]}")

            if result.get("done", False):
                break

        # Get final score
        resp = client.get("/grader")
        grader_data = resp.json()
        score = grader_data.get("score", 0.0)

        results[f"task_{task_id}"] = {
            "score": score,
            "difficulty": grader_data.get("task_difficulty", "unknown"),
            "steps_taken": grader_data.get("steps_taken", step_count),
            "cumulative_reward": grader_data.get("cumulative_reward", 0.0),
            "max_possible_reward": grader_data.get("max_possible_reward", 0.0),
        }

        if verbose:
            print(f"\n  ► Score: {score:.4f}")

    client.close()

    # Summary
    scores = [v["score"] for v in results.values() if isinstance(v, dict)]
    avg = sum(scores) / len(scores) if scores else 0.0
    results["average_score"] = round(avg, 4)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  BASELINE RESULTS SUMMARY")
        print(f"{'='*60}")
        for key, val in results.items():
            if key == "average_score":
                print(f"  Average Score: {val:.4f}")
            elif isinstance(val, dict):
                print(f"  {key} ({val.get('difficulty', '?')}): {val['score']:.4f}")
        print(f"{'='*60}\n")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline agent on Support Ticket Triage environment"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the environment server (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic rule-based agent (no Gemini API call)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step output",
    )
    args = parser.parse_args()

    results = run_baseline(
        base_url=args.base_url,
        use_deterministic=args.deterministic,
        verbose=not args.quiet,
    )

    # Write results to file
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results written to baseline_results.json")


if __name__ == "__main__":
    main()
