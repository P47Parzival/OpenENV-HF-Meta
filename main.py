"""
OpenEnv Support Ticket Triage — FastAPI Environment Server

Implements the full OpenEnv spec with endpoints:
    POST /reset?task_idx=0  — reset environment, load a task
    POST /step              — execute an agent action
    GET  /state             — retrieve current environment state
    GET  /tasks             — list available tasks with metadata
    GET  /grader            — get normalized score [0.0–1.0]
    GET  /health            — health check for Docker / HF Spaces
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    Action,
    ActionType,
    EnvState,
    Observation,
    StepResult,
    Ticket,
    TicketStatus,
)
from rewards import (
    calculate_step_reward,
    compute_max_possible_reward,
    normalize_score,
)
from tasks import TASKS, get_blueprint, get_task

load_dotenv()

# ── FastAPI application ─────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv — Support Ticket Triage",
    description=(
        "A real-world OpenEnv environment where AI agents learn to classify, "
        "prioritize, route, and resolve customer support tickets."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment singleton ───────────────────────────────────────────────────

_env = EnvState()


def _build_observation() -> Observation:
    """Construct the agent-visible observation from current state."""
    open_tix = [t for t in _env.tickets if t.status != TicketStatus.RESOLVED and t.status != "resolved"]
    resolved_count = len(_env.tickets) - len(open_tix)
    pending = sum(
        1 for t in open_tix
        if t.status in (TicketStatus.ROUTED, TicketStatus.IN_PROGRESS, "routed", "in_progress")
    )
    # Average wait = mean steps since each open ticket was created
    if open_tix:
        avg_wait = sum(_env.step_count - t.created_at for t in open_tix) / len(open_tix)
    else:
        avg_wait = 0.0

    return Observation(
        open_tickets=open_tix,
        resolved_count=resolved_count,
        pending_response_count=pending,
        avg_wait_steps=round(avg_wait, 2),
        step_number=_env.step_count,
        max_steps=_env.max_steps,
    )


def _load_task(task_idx: int) -> None:
    """Reset state and populate tickets from a task definition."""
    global _env
    task = get_task(task_idx)

    tickets = []
    for bp in task.ticket_blueprints:
        tickets.append(
            Ticket(
                id=bp.id,
                subject=bp.subject,
                body=bp.body,
                customer_sentiment=bp.sentiment,
                created_at=0,
            )
        )

    max_reward = compute_max_possible_reward(task.ticket_blueprints)

    _env = EnvState(
        episode_id=str(uuid4()),
        task_id=task_idx,
        task_difficulty=task.difficulty,
        step_count=0,
        max_steps=task.max_steps,
        tickets=tickets,
        cumulative_reward=0.0,
        max_possible_reward=max_reward,
        action_history=[],
        done=False,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(task_idx: int = 0):
    """
    Reset the environment and load a task.

    Args:
        task_idx: Task index (0=easy, 1=medium, 2=hard)

    Returns:
        Initial observation with open tickets.
    """
    try:
        _load_task(task_idx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _build_observation()


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """
    Execute a single agent action.

    The action must target an existing, non-resolved ticket with a valid
    action_type and (where applicable) a target value.

    Returns:
        StepResult with observation, reward, done flag, and info.
    """
    if _env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call /reset to start a new one.",
        )

    # Find the ticket
    ticket = next((t for t in _env.tickets if t.id == action.ticket_id), None)
    if ticket is None:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket '{action.ticket_id}' not found.",
        )

    if ticket.status in (TicketStatus.RESOLVED, "resolved"):
        # Acting on a resolved ticket — small penalty
        _env.step_count += 1
        _env.cumulative_reward -= 0.10
        _env.action_history.append({
            "step": _env.step_count,
            "ticket_id": action.ticket_id,
            "action_type": action.action_type,
            "target": action.target,
            "reward": -0.10,
            "feedback": f"Ticket {action.ticket_id} is already resolved.",
        })
        done = _env.step_count >= _env.max_steps
        _env.done = done
        return StepResult(
            observation=_build_observation(),
            reward=-0.10,
            done=done,
            info={
                "step": _env.step_count,
                "feedback": f"Ticket {action.ticket_id} is already resolved.",
                "cumulative_reward": round(_env.cumulative_reward, 4),
                "normalized_score": normalize_score(
                    _env.cumulative_reward, _env.max_possible_reward
                ),
            },
        )

    # Look up the blueprint for ground-truth scoring
    try:
        blueprint = get_blueprint(_env.task_id, action.ticket_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"No blueprint for ticket '{action.ticket_id}' in task {_env.task_id}.",
        )

    # Calculate reward
    _env.step_count += 1
    step_reward, feedback = calculate_step_reward(
        action=action,
        ticket=ticket,
        blueprint=blueprint,
        action_history=_env.action_history,
        all_tickets=_env.tickets,
        current_step=_env.step_count,
    )
    _env.cumulative_reward += step_reward

    # Record in history
    _env.action_history.append({
        "step": _env.step_count,
        "ticket_id": action.ticket_id,
        "action_type": action.action_type,
        "target": action.target,
        "message": action.message,
        "reward": round(step_reward, 4),
        "feedback": feedback,
    })

    # Check termination
    all_resolved = all(
        t.status in (TicketStatus.RESOLVED, "resolved") for t in _env.tickets
    )
    done = all_resolved or _env.step_count >= _env.max_steps
    _env.done = done

    score = normalize_score(_env.cumulative_reward, _env.max_possible_reward)

    return StepResult(
        observation=_build_observation(),
        reward=round(step_reward, 4),
        done=done,
        info={
            "step": _env.step_count,
            "feedback": feedback,
            "cumulative_reward": round(_env.cumulative_reward, 4),
            "normalized_score": round(score, 4),
            "all_resolved": all_resolved,
        },
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the full environment state (for debugging / grading)."""
    return {
        "episode_id": _env.episode_id,
        "task_id": _env.task_id,
        "task_difficulty": _env.task_difficulty,
        "step_count": _env.step_count,
        "max_steps": _env.max_steps,
        "done": _env.done,
        "tickets": [t.model_dump() for t in _env.tickets],
        "cumulative_reward": round(_env.cumulative_reward, 4),
        "max_possible_reward": round(_env.max_possible_reward, 4),
        "normalized_score": round(
            normalize_score(_env.cumulative_reward, _env.max_possible_reward), 4
        ),
        "action_history": _env.action_history,
    }


@app.get("/tasks")
def tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": t.id,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
                "num_tickets": len(t.ticket_blueprints),
            }
            for t in TASKS
        ],
        "action_types": [e.value for e in ActionType],
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
def grader():
    """
    Return the normalized score for the current episode.

    Score is in [0.0, 1.0], where 1.0 = perfect performance.
    """
    score = normalize_score(_env.cumulative_reward, _env.max_possible_reward)
    return {
        "score": round(score, 4),
        "cumulative_reward": round(_env.cumulative_reward, 4),
        "max_possible_reward": round(_env.max_possible_reward, 4),
        "task_id": _env.task_id,
        "task_difficulty": _env.task_difficulty,
        "steps_taken": _env.step_count,
        "done": _env.done,
    }


@app.get("/health")
def health():
    """Health check endpoint for Docker / Hugging Face Spaces."""
    return {"status": "healthy", "environment": "support_ticket_triage"}


# ── Run directly ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)