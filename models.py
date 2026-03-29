"""
Typed Pydantic models for the Support Ticket Triage OpenEnv environment.

Defines the Action space, Observation space, State, and supporting types
following the OpenEnv specification for type-safe environment interactions.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """All possible agent actions."""
    CLASSIFY = "classify"
    PRIORITIZE = "prioritize"
    ROUTE = "route"
    RESPOND = "respond"
    ESCALATE = "escalate"
    RESOLVE = "resolve"


class TicketCategory(str, Enum):
    """Ticket classification categories."""
    IT_SUPPORT = "it_support"
    BILLING = "billing"
    SALES = "sales"
    COMPLAINT = "complaint"
    SECURITY = "security"
    ACCOUNT = "account"
    GENERAL = "general"


class Priority(str, Enum):
    """Ticket priority levels (P1 is most urgent)."""
    P1_CRITICAL = "P1"
    P2_HIGH = "P2"
    P3_MEDIUM = "P3"
    P4_LOW = "P4"


class Department(str, Enum):
    """Departments tickets can be routed to."""
    IT_SUPPORT = "it_support"
    BILLING = "billing"
    SALES = "sales"
    CUSTOMER_SUCCESS = "customer_success"
    SECURITY = "security"
    ENGINEERING = "engineering"
    MANAGEMENT = "management"
    LEGAL = "legal"


class TicketStatus(str, Enum):
    """Lifecycle status of a ticket."""
    OPEN = "open"
    CLASSIFIED = "classified"
    PRIORITIZED = "prioritized"
    ROUTED = "routed"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class Sentiment(str, Enum):
    """Customer sentiment levels."""
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


# ── Core Models ─────────────────────────────────────────────────────────────

class Ticket(BaseModel):
    """A customer support ticket with full metadata."""
    id: str
    subject: str
    body: str
    status: TicketStatus = TicketStatus.OPEN
    category: Optional[TicketCategory] = None
    priority: Optional[Priority] = None
    assigned_to: Optional[Department] = None
    customer_sentiment: Sentiment = Sentiment.NEUTRAL
    created_at: int = 0  # step number when created
    resolution_steps_done: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class Action(BaseModel):
    """
    An agent action in the triage environment.

    Actions target a specific ticket and perform an operation such as
    classifying, prioritizing, routing, responding, escalating, or resolving.
    """
    ticket_id: str = Field(..., description="ID of the ticket to act on")
    action_type: ActionType = Field(..., description="Type of action to perform")
    target: Optional[str] = Field(
        None,
        description="Target value — category for classify, priority for prioritize, "
                    "department for route/escalate"
    )
    message: Optional[str] = Field(
        None,
        description="Optional response message for 'respond' actions"
    )

    class Config:
        use_enum_values = True


class Observation(BaseModel):
    """
    What the agent observes about the current environment state.

    Includes open tickets, aggregate metrics, and step progress.
    """
    open_tickets: List[Ticket] = Field(default_factory=list)
    resolved_count: int = 0
    pending_response_count: int = 0
    avg_wait_steps: float = 0.0
    satisfaction_score: float = 1.0
    step_number: int = 0
    max_steps: int = 0
    last_action_feedback: str = "Environment reset. Awaiting first action."


class StepResult(BaseModel):
    """Result returned from a step() call."""
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    """Full internal environment state (returned by state())."""
    episode_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: int = 0
    task_difficulty: str = "easy"
    step_count: int = 0
    max_steps: int = 8
    tickets: List[Ticket] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    max_possible_reward: float = 0.0
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False
