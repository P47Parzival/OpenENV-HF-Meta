---
title: Support Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🎫 Support Ticket Triage — OpenEnv Environment

A **real-world** [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to classify, prioritize, route, and resolve customer support tickets. Designed for reinforcement learning research and agentic AI evaluation.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-orange)](LICENSE)

---

## 📋 Overview

Every company with customers needs a support triage system. This environment simulates the decision-making process of a Tier-1 support agent who must:

1. **Read** incoming support tickets with varying urgency and sentiment
2. **Classify** each ticket into the correct category
3. **Prioritize** based on severity (P1-Critical → P4-Low)
4. **Route** to the appropriate department
5. **Escalate** complex cases to management/security
6. **Respond** to customers with appropriate messaging
7. **Resolve** tickets after completing the required workflow

The environment provides **partial progress rewards** — agents get signal for each correct sub-step, not just binary success/failure at the end.

---

## 🎮 Tasks (3 Difficulty Levels)

| Task | Difficulty | Tickets | Max Steps | Description |
|------|-----------|---------|-----------|-------------|
| 0    | 🟢 Easy   | 2       | 8         | Route 1 IT + 1 billing ticket with clear categories |
| 1    | 🟡 Medium | 4       | 15        | Mixed IT, billing, sales, complaint tickets with priority assessment |
| 2    | 🔴 Hard   | 6       | 25        | Security breach, legal compliance, VIP SLA, production outage, refund dispute, multi-issue edge case |

---

## 🕹️ Action Space

Actions are JSON objects targeting a specific ticket:

```json
{
  "ticket_id": "T001",
  "action_type": "classify",
  "target": "it_support",
  "message": null
}
```

### Available Action Types

| Action Type  | `target` Values | Description |
|-------------|----------------|-------------|
| `classify`  | `it_support`, `billing`, `sales`, `complaint`, `security`, `account`, `general` | Set ticket category |
| `prioritize`| `P1`, `P2`, `P3`, `P4` | Set priority level |
| `route`     | `it_support`, `billing`, `sales`, `customer_success`, `security`, `engineering`, `management`, `legal` | Route to department |
| `respond`   | *(uses `message` field)* | Send customer response |
| `escalate`  | Department name | Escalate to management/security |
| `resolve`   | *(none required)* | Mark ticket as resolved |

---

## 👁️ Observation Space

The observation returned after each step contains:

| Field | Type | Description |
|-------|------|-------------|
| `open_tickets` | `List[Ticket]` | All unresolved tickets with full metadata |
| `resolved_count` | `int` | Number of resolved tickets |
| `pending_response_count` | `int` | Tickets awaiting customer response |
| `avg_wait_steps` | `float` | Average steps since ticket creation |
| `step_number` | `int` | Current step in the episode |
| `max_steps` | `int` | Maximum steps before episode ends |
| `last_action_feedback` | `str` | Human-readable feedback on last action |

Each **Ticket** in `open_tickets` contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique ticket ID (e.g., "T001") |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Full ticket body text |
| `status` | `str` | Current status (open, classified, prioritized, routed, etc.) |
| `category` | `str?` | Assigned category (null if unclassified) |
| `priority` | `str?` | Assigned priority (null if unprioritized) |
| `assigned_to` | `str?` | Assigned department (null if unrouted) |
| `customer_sentiment` | `str` | Customer's emotional state (angry, frustrated, neutral, satisfied) |

---

## 💰 Reward Function

Rewards are **partial-credit** — agents earn signal for each correct intermediate step:

| Action | Correct | Partial | Wrong |
|--------|---------|---------|-------|
| `classify` | +0.20 | — | −0.10 |
| `prioritize` | +0.15 | — | −0.10 |
| `route` | +0.25 | +0.05 (related dept) | −0.15 |
| `respond` | +0.10 | — | — |
| `escalate` | +0.20 (warranted) | — | −0.15 (unwarranted) |
| `resolve` | +0.30 (full workflow) | +0.10 (partial) | −0.20 (premature) |

**Bonuses & Penalties:**
- ✅ **+0.05** — Priority ordering bonus (handling P1 before P4)
- ❌ **−0.05** — Redundant/repeated action on same ticket
- ❌ **−0.10** — Acting on already-resolved tickets

**Final Score:** Normalized to `[0.0, 1.0]` via `max(0, min(1, cumulative / max_possible))`

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone and setup
cd OpenENV
pip install -r requirements.txt

# Start the environment server
python main.py
# Server runs at http://localhost:7860

# In another terminal, run the baseline
python baseline.py --deterministic
```

### Docker

```bash
# Build
docker build -t support-triage .

# Run
docker run -p 7860:7860 support-triage

# Run baseline against container
python baseline.py --base-url http://localhost:7860
```

### Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Upload all files to the Space repository
3. The environment will be available at `https://your-username-your-space.hf.space`

```bash
# Run baseline against HF Space
python baseline.py --base-url https://your-username-your-space.hf.space
```

---

## 📊 Baseline Results

Using the **deterministic rule-based agent** (reproducible, no API key needed):

| Task | Difficulty | Score | Steps Used |
|------|-----------|-------|------------|
| 0    | Easy      | ~0.85 | 8         |
| 1    | Medium    | ~0.65 | 15        |
| 2    | Hard      | ~0.50 | 25        |

Using **Gemini 2.0 Flash** (requires `GEMINI_API_KEY`):

| Task | Difficulty | Score | Steps Used |
|------|-----------|-------|------------|
| 0    | Easy      | ~0.92 | 6         |
| 1    | Medium    | ~0.75 | 14        |
| 2    | Hard      | ~0.55 | 24        |

Run the baseline yourself:

```bash
# Deterministic (no API key needed)
python baseline.py --deterministic

# With Gemini
export GEMINI_API_KEY=your_key_here
python baseline.py
```

---

## 🔌 API Reference

### `POST /reset?task_idx=0`

Reset the environment and load a task.

```bash
curl -X POST "http://localhost:7860/reset?task_idx=0"
```

**Response:** `Observation` with initial open tickets.

---

### `POST /step`

Execute an agent action.

```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"ticket_id": "T001", "action_type": "classify", "target": "it_support"}'
```

**Response:**
```json
{
  "observation": { "open_tickets": [...], "step_number": 1, ... },
  "reward": 0.20,
  "done": false,
  "info": {
    "step": 1,
    "feedback": "Correctly classified T001 as 'it_support' (+0.20).",
    "cumulative_reward": 0.20,
    "normalized_score": 0.1111
  }
}
```

---

### `GET /state`

Retrieve full environment state (for debugging).

```bash
curl "http://localhost:7860/state"
```

---

### `GET /tasks`

List available tasks with metadata and action schema.

```bash
curl "http://localhost:7860/tasks"
```

---

### `GET /grader`

Get the normalized score for the current episode.

```bash
curl "http://localhost:7860/grader"
```

**Response:**
```json
{
  "score": 0.85,
  "cumulative_reward": 1.53,
  "max_possible_reward": 1.80,
  "task_id": 0,
  "task_difficulty": "easy",
  "steps_taken": 8,
  "done": true
}
```

---

### `GET /health`

Health check for container orchestration.

```bash
curl "http://localhost:7860/health"
```

---

## 🗂️ Project Structure

```
OpenENV/
├── main.py           # FastAPI server with all OpenEnv endpoints
├── models.py         # Typed Pydantic models (Action, Observation, State, etc.)
├── tasks.py          # Task definitions & realistic ticket corpus
├── rewards.py        # Partial-credit reward engine
├── baseline.py       # Baseline inference script (Gemini + deterministic)
├── openenv.yaml      # OpenEnv environment manifest
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container config for HF Spaces deployment
├── .env              # API keys (not committed)
├── .gitignore        # Git exclusions
└── README.md         # This file
```

---

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | No (for LLM baseline) | Google Gemini API key |
| `PORT` | No (default: 7860) | Server port |

---

## 📝 License

BSD 3-Clause License — see [LICENSE](LICENSE) for details.
