# Architecture Notes

## Overview

Revenue Recovery Autopilot is a **modular monolith** with a background worker.
It is deliberately NOT built as microservices.

## Core Loop

```
Failed Payment
    │
    ▼
Revenue-at-Risk Detection        ← deterministic: identify eligible cases
    │
    ▼
ML Recovery Scoring              ← model: P(recovery | transaction, action)
    │
    ▼
LLM Reasoning                    ← bounded: pick action from allowed set + explain
    │
    ▼
Policy Engine                    ← deterministic: is this action allowed?
    │
    ├─ NO  → STOP / ESCALATE
    └─ YES ─▶ Action Executor
                  │
                  ├─ PAYMENT_LINK  → Razorpay Test API
                  ├─ REMINDER      → simulated notification
                  └─ ESCALATE      → flag for human review
                        │
                        ▼
                  Razorpay Webhook → update outcome → metrics → audit log
```

## Key Design Decisions

### 1. ML and LLM are separate

- **ML model** answers: *"What is P(recovery) for each action?"*
- **LLM** answers: *"Given this context, what action should we take and why?"*
- **Policy engine** answers: *"Are we allowed to take that action?"*

The LLM never executes financial actions. It only produces a structured proposal.

### 2. Policy engine is deterministic

All guardrails are explicit Python code — no fuzzy AI decisions about whether an action is allowed.

### 3. Webhook handling is idempotent

Duplicate webhooks must not produce duplicate recovery outcomes.
The event ID is stored and deduplicated before processing.

### 4. No Celery in Phase 0

Redis is present for the planned architecture. Celery workers will be introduced in Phase 5 when background recovery jobs are implemented.

## Module Map

```
backend/app/
├── api/           HTTP routers
├── models/        SQLAlchemy ORM models        (Phase 1)
├── schemas/       Pydantic request/response     (Phase 1)
├── services/      Business logic               (Phase 1+)
├── integrations/  Razorpay client wrapper      (Phase 2)
├── ml/            Feature engineering + model  (Phase 3)
├── agent/         LLM reasoning layer          (Phase 4)
├── policies/      Deterministic policy engine  (Phase 4)
├── workers/       Celery task definitions      (Phase 5)
├── audit/         Audit log writer             (Phase 4)
└── utils/         Shared helpers
```
