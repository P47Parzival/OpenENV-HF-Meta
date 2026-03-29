"""
Reward engine for the Support Ticket Triage environment.

Provides partial-credit scoring so agents get signal for incremental progress,
not just binary success/failure.  Final scores are normalized to [0.0, 1.0].

Reward table:
    classify  — correct: +0.20, wrong: −0.10
    prioritize — correct: +0.15, wrong: −0.10
    route     — correct dept: +0.25, related dept: +0.05, wrong: −0.15
    respond   — +0.10 (always helpful if ticket is in progress)
    escalate  — warranted: +0.20, unwarranted: −0.15
    resolve   — full workflow done: +0.30, partial: +0.10, premature: −0.20

Bonuses / penalties applied globally:
    +0.05  correct priority ordering (handling P1 before P4)
    −0.05  redundant / repeated action on same ticket
    −0.10  per step when avg wait time exceeds threshold
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from models import Action, ActionType, Ticket, TicketStatus
from tasks import TicketBlueprint


# ── Reward constants ────────────────────────────────────────────────────────

_R = {
    # (action_type, outcome) → reward
    ("classify", "correct"):     0.20,
    ("classify", "wrong"):      -0.10,
    ("prioritize", "correct"):   0.15,
    ("prioritize", "wrong"):    -0.10,
    ("route", "correct"):        0.25,
    ("route", "partial"):        0.05,
    ("route", "wrong"):         -0.15,
    ("respond", "ok"):           0.10,
    ("escalate", "warranted"):   0.20,
    ("escalate", "unwarranted"):-0.15,
    ("resolve", "full"):         0.30,
    ("resolve", "partial"):      0.10,
    ("resolve", "premature"):   -0.20,
}

REDUNDANCY_PENALTY = -0.05
PRIORITY_ORDER_BONUS = 0.05
WAIT_TIME_PENALTY = -0.10
WAIT_TIME_THRESHOLD = 5  # steps before penalty kicks in


def _needs_escalation(bp: TicketBlueprint) -> bool:
    """Check if a ticket's ground-truth workflow includes escalation."""
    return "escalate" in bp.required_workflow


def _workflow_completion_ratio(ticket: Ticket, bp: TicketBlueprint) -> float:
    """Fraction of required workflow steps that have been completed."""
    if not bp.required_workflow:
        return 1.0
    done = set(ticket.resolution_steps_done)
    required = set(bp.required_workflow)
    return len(done & required) / len(required)


# ── Main reward function ────────────────────────────────────────────────────

def calculate_step_reward(
    action: Action,
    ticket: Ticket,
    blueprint: TicketBlueprint,
    action_history: List[Dict],
    all_tickets: List[Ticket],
    current_step: int,
) -> Tuple[float, str]:
    """
    Compute the reward for a single step.

    Returns:
        (reward, feedback_message)
    """
    reward = 0.0
    feedback_parts: List[str] = []
    act = action.action_type

    # ── Check for redundancy ────────────────────────────────────────────
    same_actions = [
        h for h in action_history
        if h.get("ticket_id") == action.ticket_id
        and h.get("action_type") == act
    ]
    if same_actions:
        reward += REDUNDANCY_PENALTY
        feedback_parts.append(
            f"Redundant '{act}' on {action.ticket_id} (−0.05)."
        )

    # ── Classify ────────────────────────────────────────────────────────
    if act == ActionType.CLASSIFY or act == "classify":
        if action.target == blueprint.correct_category:
            reward += _R[("classify", "correct")]
            ticket.category = action.target
            ticket.status = TicketStatus.CLASSIFIED
            ticket.resolution_steps_done.append("classify")
            feedback_parts.append(
                f"Correctly classified {ticket.id} as '{action.target}' (+0.20)."
            )
        else:
            reward += _R[("classify", "wrong")]
            ticket.category = action.target  # still apply it
            ticket.status = TicketStatus.CLASSIFIED
            ticket.resolution_steps_done.append("classify")
            feedback_parts.append(
                f"Misclassified {ticket.id} as '{action.target}' "
                f"(expected '{blueprint.correct_category}', −0.10)."
            )

    # ── Prioritize ──────────────────────────────────────────────────────
    elif act == ActionType.PRIORITIZE or act == "prioritize":
        if action.target == blueprint.correct_priority:
            reward += _R[("prioritize", "correct")]
            ticket.priority = action.target
            ticket.status = TicketStatus.PRIORITIZED
            ticket.resolution_steps_done.append("prioritize")
            feedback_parts.append(
                f"Correctly prioritized {ticket.id} as {action.target} (+0.15)."
            )
        else:
            reward += _R[("prioritize", "wrong")]
            ticket.priority = action.target
            ticket.status = TicketStatus.PRIORITIZED
            ticket.resolution_steps_done.append("prioritize")
            feedback_parts.append(
                f"Wrong priority for {ticket.id}: set {action.target}, "
                f"expected {blueprint.correct_priority} (−0.10)."
            )

    # ── Route ───────────────────────────────────────────────────────────
    elif act == ActionType.ROUTE or act == "route":
        if action.target == blueprint.correct_department:
            reward += _R[("route", "correct")]
            ticket.assigned_to = action.target
            ticket.status = TicketStatus.ROUTED
            ticket.resolution_steps_done.append("route")
            feedback_parts.append(
                f"Correctly routed {ticket.id} to '{action.target}' (+0.25)."
            )
        elif action.target in blueprint.partial_credit_depts:
            reward += _R[("route", "partial")]
            ticket.assigned_to = action.target
            ticket.status = TicketStatus.ROUTED
            ticket.resolution_steps_done.append("route")
            feedback_parts.append(
                f"Partially correct route for {ticket.id} to '{action.target}' "
                f"(best: '{blueprint.correct_department}', +0.05)."
            )
        else:
            reward += _R[("route", "wrong")]
            ticket.assigned_to = action.target
            ticket.status = TicketStatus.ROUTED
            ticket.resolution_steps_done.append("route")
            feedback_parts.append(
                f"Wrong route for {ticket.id}: sent to '{action.target}', "
                f"expected '{blueprint.correct_department}' (−0.15)."
            )

    # ── Respond ─────────────────────────────────────────────────────────
    elif act == ActionType.RESPOND or act == "respond":
        reward += _R[("respond", "ok")]
        ticket.status = TicketStatus.IN_PROGRESS
        ticket.resolution_steps_done.append("respond")
        feedback_parts.append(
            f"Responded to {ticket.id} (+0.10)."
        )

    # ── Escalate ────────────────────────────────────────────────────────
    elif act == ActionType.ESCALATE or act == "escalate":
        if _needs_escalation(blueprint):
            reward += _R[("escalate", "warranted")]
            ticket.status = TicketStatus.ESCALATED
            ticket.resolution_steps_done.append("escalate")
            if action.target:
                ticket.assigned_to = action.target
            feedback_parts.append(
                f"Escalation of {ticket.id} was warranted (+0.20)."
            )
        else:
            reward += _R[("escalate", "unwarranted")]
            ticket.status = TicketStatus.ESCALATED
            ticket.resolution_steps_done.append("escalate")
            feedback_parts.append(
                f"Unnecessary escalation of {ticket.id} (−0.15)."
            )

    # ── Resolve ─────────────────────────────────────────────────────────
    elif act == ActionType.RESOLVE or act == "resolve":
        ratio = _workflow_completion_ratio(ticket, blueprint)
        # Need at least classify + route done before resolving
        min_steps_done = (
            "classify" in ticket.resolution_steps_done
            and "route" in ticket.resolution_steps_done
        )
        if ratio >= 0.8 and min_steps_done:
            reward += _R[("resolve", "full")]
            ticket.status = TicketStatus.RESOLVED
            ticket.resolution_steps_done.append("resolve")
            feedback_parts.append(
                f"Fully resolved {ticket.id} after complete workflow (+0.30)."
            )
        elif min_steps_done:
            reward += _R[("resolve", "partial")]
            ticket.status = TicketStatus.RESOLVED
            ticket.resolution_steps_done.append("resolve")
            feedback_parts.append(
                f"Resolved {ticket.id} with partial workflow "
                f"({ratio:.0%} complete, +0.10)."
            )
        else:
            reward += _R[("resolve", "premature")]
            ticket.resolution_steps_done.append("resolve")
            feedback_parts.append(
                f"Premature resolve of {ticket.id} — "
                f"missing required workflow steps (−0.20)."
            )

    # ── Priority ordering bonus ─────────────────────────────────────────
    # Award a bonus if the agent handles higher-priority tickets first
    _PRIO_ORDER = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    ticket_prio = _PRIO_ORDER.get(blueprint.correct_priority, 3)

    unhandled_higher = any(
        _PRIO_ORDER.get(
            _find_prio(t.id, all_tickets), 3
        ) < ticket_prio
        and t.status in (TicketStatus.OPEN, "open")
        for t in all_tickets
        if t.id != ticket.id
    )
    if not unhandled_higher and ticket_prio <= 1 and act in ("classify", "route", ActionType.CLASSIFY, ActionType.ROUTE):
        reward += PRIORITY_ORDER_BONUS
        feedback_parts.append("Priority ordering bonus (+0.05).")

    feedback = " ".join(feedback_parts) if feedback_parts else "Action processed."
    return reward, feedback


def _find_prio(ticket_id: str, tickets: List[Ticket]) -> str:
    """Get the priority string of a ticket, defaulting to P4."""
    for t in tickets:
        if t.id == ticket_id and t.priority:
            return t.priority
    return "P4"


def compute_max_possible_reward(blueprints: list) -> float:
    """
    Calculate the theoretical maximum reward for a set of tickets,
    assuming the agent performs every required action correctly.
    """
    total = 0.0
    for bp in blueprints:
        for step_name in bp.required_workflow:
            if step_name == "classify":
                total += _R[("classify", "correct")]
            elif step_name == "prioritize":
                total += _R[("prioritize", "correct")]
            elif step_name == "route":
                total += _R[("route", "correct")]
            elif step_name == "respond":
                total += _R[("respond", "ok")]
            elif step_name == "escalate":
                total += _R[("escalate", "warranted")]
            elif step_name == "resolve":
                total += _R[("resolve", "full")]
    return total


def normalize_score(cumulative: float, max_possible: float) -> float:
    """Normalize cumulative reward to [0.0, 1.0]."""
    if max_possible <= 0:
        return 0.0
    return max(0.0, min(1.0, cumulative / max_possible))
