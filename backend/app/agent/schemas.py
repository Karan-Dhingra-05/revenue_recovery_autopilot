"""
Pydantic schemas for the recovery-agent input, output, and validation.

Two main models:
  - AgentInput:  structured context passed to the LLM
  - AgentProposal: validated structured output from the LLM

The schemas enforce:
  - allowed action enum (PAYMENT_LINK / REMINDER / DELAYED_RETRY / ESCALATE)
  - priority enum (LOW / MEDIUM / HIGH)
  - confidence in [0, 1]
  - non-empty reason
  - no arbitrary fields

STOP is intentionally excluded — it is a deterministic policy-engine
decision, not an LLM action.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class AllowedAction(str, enum.Enum):
    """Actions the LLM may propose.  STOP is NOT included."""

    PAYMENT_LINK = "PAYMENT_LINK"
    REMINDER = "REMINDER"
    DELAYED_RETRY = "DELAYED_RETRY"
    ESCALATE = "ESCALATE"


class Priority(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ── Agent Input ──────────────────────────────────────────────────────────────


class ActionScore(BaseModel):
    """ML prediction + deterministic expected-recovery for one action."""

    action: AllowedAction
    probability: float = Field(..., ge=0.0, le=1.0)
    expected_recovery: float = Field(..., ge=0.0)


class AgentInput(BaseModel):
    """
    Structured context sent to the LLM.

    All numerical values (probabilities, expected recoveries, amounts)
    are computed by application code — the LLM must not recalculate them.
    """

    transaction_id: str
    amount: float = Field(..., gt=0)
    currency: str = "INR"
    payment_method: str
    failure_reason: str
    failure_pattern: str
    attempt_number: int = Field(..., ge=1)

    # Customer context (pre-computed from DB / dataset features).
    customer_success_rate: float = Field(..., ge=0.0, le=1.0)
    customer_previous_failures: int = Field(..., ge=0)
    customer_previous_recoveries: int = Field(..., ge=0)
    hours_since_last_success: float
    subscription_flag: bool = False

    # ML scores (computed by LightGBM, NOT the LLM).
    action_scores: list[ActionScore]

    # Merchant policy summary (human-readable, for LLM context only).
    merchant_policy_summary: str = "Default policy: max 3 recovery attempts, standard cooldown."


# ── Agent Output (LLM proposal) ─────────────────────────────────────────────


class AgentProposal(BaseModel):
    """
    Validated structured output from the LLM.

    Every field has strict validation so malformed LLM responses are
    rejected before they reach the policy engine.
    """

    recommended_action: AllowedAction
    priority: Priority
    reason: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must contain non-whitespace text")
        return v.strip()


# ── Failure result ───────────────────────────────────────────────────────────


class AgentFailure(BaseModel):
    """
    Returned when the LLM call fails or produces invalid output.

    The application can inspect `error_type` to decide fallback behaviour.
    """

    error_type: str  # e.g. "timeout", "api_error", "validation_error", "malformed_json"
    error_message: str
    raw_response: str | None = None
