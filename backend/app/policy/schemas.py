"""
Schemas and configurations for the Policy Engine.

These classes strictly enforce the boundaries of the policy engine:
ALLOW, MODIFY, or BLOCK, and structured outputs for deterministic rules.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field

from app.agent.schemas import AllowedAction


# ── Decision Types ───────────────────────────────────────────────────────────

class DecisionType(str, enum.Enum):
    ALLOW = "ALLOW"
    MODIFY = "MODIFY"
    BLOCK = "BLOCK"


class DecisionSource(str, enum.Enum):
    GEMINI = "GEMINI"
    ML_FALLBACK = "ML_FALLBACK"


# ── Merchant Policy Configuration ────────────────────────────────────────────

class ActionConfig(BaseModel):
    """
    Synthetic product assumptions for action cost and risk penalty.
    """
    action_cost: float = Field(..., ge=0)
    risk_penalty: float = Field(..., ge=0)
    is_enabled: bool = True


class MerchantPolicy(BaseModel):
    """
    Configurable deterministic rules and assumptions for a merchant.
    """
    # Limits
    max_automatic_attempts: int = 3
    cooldown_hours: float = 24.0
    max_amount_eligible: float = 100000.0  # ₹1 Lakh
    max_customer_actions_per_month: int = 5
    recovery_window_days: int = 14

    # Action Costs & Risk Penalties (Synthetic Assumptions)
    action_configs: dict[AllowedAction, ActionConfig] = Field(
        default_factory=lambda: {
            AllowedAction.PAYMENT_LINK: ActionConfig(action_cost=0, risk_penalty=0),
            AllowedAction.REMINDER: ActionConfig(action_cost=1, risk_penalty=0),
            AllowedAction.DELAYED_RETRY: ActionConfig(action_cost=3, risk_penalty=10),
            AllowedAction.ESCALATE: ActionConfig(action_cost=100, risk_penalty=50),
        }
    )


# ── Rule and Decision Outputs ────────────────────────────────────────────────

class RuleResult(BaseModel):
    """Output of a single deterministic rule evaluation."""
    rule_name: str
    passed: bool
    message: str


class PolicyDecision(BaseModel):
    """
    The final deterministic output of the Policy Engine.
    """
    decision: DecisionType
    decision_source: DecisionSource
    
    proposed_action: AllowedAction
    final_action: AllowedAction | None
    
    reason: str
    rule_results: list[RuleResult]
    
    # Economics (App-calculated, never trusted from LLM)
    expected_recovery: dict[AllowedAction, float]
    expected_net_recovery: dict[AllowedAction, float]
