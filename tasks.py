"""
Task definitions and ticket corpus for the Support Ticket Triage environment.

Provides 3 escalating difficulty levels:
  Task 0 (easy)   — 2 clear-cut tickets, 8 max steps
  Task 1 (medium) — 4 mixed tickets, 15 max steps
  Task 2 (hard)   — 6 complex/ambiguous tickets, 25 max steps

Each ticket carries ground-truth labels used by the reward engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TicketBlueprint:
    """
    Ground-truth definition for a ticket.

    Attributes:
        id:                 Unique ticket identifier
        subject:            Subject line of the ticket
        body:               Full ticket body text
        correct_category:   The true category label
        correct_priority:   The true priority level (P1–P4)
        correct_department: The department that should handle this
        sentiment:          Customer's emotional state
        required_workflow:  Ordered list of action_types needed for full resolution
        partial_credit_map: Maps action types to departments that earn partial credit
    """
    id: str
    subject: str
    body: str
    correct_category: str
    correct_priority: str
    correct_department: str
    sentiment: str = "neutral"
    required_workflow: List[str] = field(default_factory=lambda: [
        "classify", "prioritize", "route", "resolve"
    ])
    partial_credit_depts: List[str] = field(default_factory=list)


@dataclass
class TaskDefinition:
    """A task is a set of tickets with configured difficulty."""
    id: int
    difficulty: str
    description: str
    max_steps: int
    ticket_blueprints: List[TicketBlueprint]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TICKET CORPUS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Easy tickets ────────────────────────────────────────────────────────────

_PASSWORD_RESET = TicketBlueprint(
    id="T001",
    subject="Can't log in to my account",
    body=(
        "Hi, I've been trying to log in for the past 30 minutes but it keeps "
        "saying 'Invalid credentials'. I'm sure my password is correct. "
        "Can you please help me reset it?"
    ),
    correct_category="it_support",
    correct_priority="P3",
    correct_department="it_support",
    sentiment="frustrated",
    required_workflow=["classify", "prioritize", "route", "resolve"],
)

_BILLING_OVERCHARGE = TicketBlueprint(
    id="T002",
    subject="Overcharged on my last invoice",
    body=(
        "I was charged $149.99 on my last invoice but my plan is $79.99/month. "
        "This has happened before and I need it corrected immediately. "
        "Please issue a refund for the difference."
    ),
    correct_category="billing",
    correct_priority="P2",
    correct_department="billing",
    sentiment="frustrated",
    required_workflow=["classify", "prioritize", "route", "resolve"],
)

# ── Medium tickets ──────────────────────────────────────────────────────────

_SALES_INQUIRY = TicketBlueprint(
    id="T003",
    subject="Interested in enterprise plan",
    body=(
        "Our company (500+ employees) is evaluating your platform. We'd like "
        "to schedule a demo and understand the enterprise pricing structure. "
        "Who should we talk to about volume discounts?"
    ),
    correct_category="sales",
    correct_priority="P3",
    correct_department="sales",
    sentiment="neutral",
    required_workflow=["classify", "prioritize", "route", "respond", "resolve"],
    partial_credit_depts=["customer_success"],
)

_PRODUCT_COMPLAINT = TicketBlueprint(
    id="T004",
    subject="App crashes every time I open reports",
    body=(
        "For the past week the mobile app crashes whenever I try to view "
        "my weekly reports. I've reinstalled it twice. This is really affecting "
        "my productivity and I'm considering switching to a competitor. "
        "Very disappointed."
    ),
    correct_category="complaint",
    correct_priority="P2",
    correct_department="engineering",
    sentiment="angry",
    required_workflow=["classify", "prioritize", "route", "respond", "resolve"],
    partial_credit_depts=["it_support", "customer_success"],
)

# ── Hard tickets ────────────────────────────────────────────────────────────

_SECURITY_BREACH = TicketBlueprint(
    id="T005",
    subject="URGENT: Unauthorized access to our account",
    body=(
        "We detected multiple unauthorized logins to our company account from "
        "IP addresses in regions where we have no employees. Several admin-level "
        "changes were made including new API keys being generated. We need this "
        "investigated IMMEDIATELY — our compliance team is involved."
    ),
    correct_category="security",
    correct_priority="P1",
    correct_department="security",
    sentiment="angry",
    required_workflow=["classify", "prioritize", "route", "escalate", "respond", "resolve"],
    partial_credit_depts=["engineering", "management"],
)

_LEGAL_THREAT = TicketBlueprint(
    id="T006",
    subject="Data deletion request — GDPR compliance",
    body=(
        "Under GDPR Article 17, I am exercising my right to erasure. Please "
        "delete ALL personal data associated with account #A-8812 within 30 days. "
        "If this is not completed, our legal department will take further action. "
        "Please confirm receipt and provide a timeline."
    ),
    correct_category="account",
    correct_priority="P1",
    correct_department="legal",
    sentiment="neutral",
    required_workflow=["classify", "prioritize", "route", "escalate", "respond", "resolve"],
    partial_credit_depts=["management", "security"],
)

_VIP_CUSTOMER = TicketBlueprint(
    id="T007",
    subject="Service degradation on premium plan",
    body=(
        "We're an Enterprise Premium customer paying $25,000/month and our "
        "dashboard has been loading extremely slowly for 3 days. Our SLA "
        "guarantees 99.9% uptime and sub-2s response times. We're tracking "
        "this for potential SLA credit claims. Please prioritize."
    ),
    correct_category="complaint",
    correct_priority="P1",
    correct_department="engineering",
    sentiment="frustrated",
    required_workflow=["classify", "prioritize", "route", "escalate", "respond", "resolve"],
    partial_credit_depts=["customer_success", "management"],
)

_CASCADING_FAILURE = TicketBlueprint(
    id="T008",
    subject="API returning 500 errors — production down",
    body=(
        "Our entire production pipeline is down because your API endpoint "
        "/v2/sync has been returning HTTP 500 since 3:42 AM UTC. This is "
        "blocking 200+ downstream jobs. We need an immediate status update "
        "and an ETA for resolution."
    ),
    correct_category="it_support",
    correct_priority="P1",
    correct_department="engineering",
    sentiment="angry",
    required_workflow=["classify", "prioritize", "route", "escalate", "respond", "resolve"],
    partial_credit_depts=["it_support"],
)

_REFUND_ABUSE = TicketBlueprint(
    id="T009",
    subject="I want a full refund for the last 6 months",
    body=(
        "Your product has been useless. Nothing works the way it's advertised. "
        "I want a complete refund for the last 6 months of payments ($479.94) "
        "or I'll be filing a chargeback with my bank and leaving a review "
        "on every platform I can find."
    ),
    correct_category="billing",
    correct_priority="P2",
    correct_department="billing",
    sentiment="angry",
    required_workflow=["classify", "prioritize", "route", "escalate", "respond", "resolve"],
    partial_credit_depts=["customer_success", "management"],
)

_EDGE_CASE = TicketBlueprint(
    id="T010",
    subject="Feature request + billing question + bug report",
    body=(
        "Three things: (1) Can you add dark mode to the dashboard? It's 2026. "
        "(2) Why was I charged twice this month? I see two charges of $39.99. "
        "(3) The export-to-PDF button generates blank PDFs on Firefox. "
        "Please address all three."
    ),
    correct_category="general",
    correct_priority="P3",
    correct_department="customer_success",
    sentiment="frustrated",
    required_workflow=["classify", "prioritize", "route", "respond", "resolve"],
    partial_credit_depts=["billing", "engineering", "it_support"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TASK DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASKS: List[TaskDefinition] = [
    # ── Task 0: Easy ────────────────────────────────────────────────────────
    TaskDefinition(
        id=0,
        difficulty="easy",
        description=(
            "Simple ticket routing — classify and route 2 straightforward "
            "tickets (IT password reset + billing overcharge)."
        ),
        max_steps=8,
        ticket_blueprints=[_PASSWORD_RESET, _BILLING_OVERCHARGE],
    ),
    # ── Task 1: Medium ──────────────────────────────────────────────────────
    TaskDefinition(
        id=1,
        difficulty="medium",
        description=(
            "Mixed priority triage — handle 4 tickets across IT, billing, "
            "sales, and complaints. Requires correct priority assessment "
            "and multi-department routing."
        ),
        max_steps=15,
        ticket_blueprints=[
            _PASSWORD_RESET,
            _BILLING_OVERCHARGE,
            _SALES_INQUIRY,
            _PRODUCT_COMPLAINT,
        ],
    ),
    # ── Task 2: Hard ────────────────────────────────────────────────────────
    TaskDefinition(
        id=2,
        difficulty="hard",
        description=(
            "Crisis management — handle 6 complex tickets including a "
            "security breach, legal compliance request, VIP SLA violation, "
            "production outage, refund dispute, and a multi-issue edge case. "
            "Requires escalation paths, priority ordering, and nuanced responses."
        ),
        max_steps=25,
        ticket_blueprints=[
            _SECURITY_BREACH,
            _LEGAL_THREAT,
            _VIP_CUSTOMER,
            _CASCADING_FAILURE,
            _REFUND_ABUSE,
            _EDGE_CASE,
        ],
    ),
]


def get_task(task_idx: int) -> TaskDefinition:
    """Retrieve a task definition by index."""
    if task_idx < 0 or task_idx >= len(TASKS):
        raise ValueError(
            f"task_idx must be 0–{len(TASKS) - 1}, got {task_idx}"
        )
    return TASKS[task_idx]


def get_blueprint(task_idx: int, ticket_id: str) -> TicketBlueprint:
    """Look up the ground-truth blueprint for a ticket within a task."""
    task = get_task(task_idx)
    for bp in task.ticket_blueprints:
        if bp.id == ticket_id:
            return bp
    raise ValueError(f"No blueprint for ticket '{ticket_id}' in task {task_idx}")
