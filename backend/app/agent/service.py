"""
Recovery agent service — orchestrates ML scoring + LLM reasoning.

This is the main entry point for generating a recovery-action proposal.

Flow:
  1. Receive transaction context (features dict or AgentInput).
  2. Score all four candidate actions using LightGBM.
  3. Calculate expected recovery = probability × amount (deterministic).
  4. Build an AgentInput with all context and ML scores.
  5. Call Gemini 2.5 Flash for reasoning/proposal.
  6. Return a validated AgentProposal or AgentFailure.

The service does NOT:
  - execute any payment action
  - modify the database
  - bypass policy rules
"""

from __future__ import annotations

import logging

from app.agent.gemini_client import call_gemini
from app.agent.schemas import (
    ActionScore,
    AgentFailure,
    AgentInput,
    AgentProposal,
    AllowedAction,
)

logger = logging.getLogger(__name__)

# The four candidate actions (matches ml/predict.py and the schema).
ACTIONS = ["PAYMENT_LINK", "REMINDER", "DELAYED_RETRY", "ESCALATE"]


def build_agent_input(
    txn_features: dict,
    ml_scores: dict[str, float],
    merchant_policy_summary: str = "Default policy: max 3 recovery attempts, standard cooldown.",
) -> AgentInput:
    """
    Build an AgentInput from raw transaction features and ML scores.

    Expected recovery is calculated HERE — never by the LLM.

    Parameters
    ----------
    txn_features : dict
        Transaction features (same dict used for ML scoring).
    ml_scores : dict[str, float]
        Action → P(recovery) mapping from LightGBM.
    merchant_policy_summary : str
        Human-readable policy for LLM context.
    """
    amount = float(txn_features["amount"])

    action_scores = []
    for action in ACTIONS:
        prob = ml_scores.get(action, 0.0)
        expected_recovery = round(prob * amount, 2)
        action_scores.append(
            ActionScore(
                action=AllowedAction(action),
                probability=prob,
                expected_recovery=expected_recovery,
            )
        )

    return AgentInput(
        transaction_id=str(txn_features.get("transaction_id", "unknown")),
        amount=amount,
        currency=str(txn_features.get("currency", "INR")),
        payment_method=str(txn_features.get("payment_method", "UNKNOWN")),
        failure_reason=str(txn_features.get("failure_reason", "UNKNOWN")),
        failure_pattern=str(txn_features.get("failure_pattern", "UNKNOWN")),
        attempt_number=int(txn_features.get("attempt_number", 1)),
        customer_success_rate=float(txn_features.get("customer_success_rate", 0.0)),
        customer_previous_failures=int(txn_features.get("customer_previous_failures", 0)),
        customer_previous_recoveries=int(txn_features.get("customer_previous_recoveries", 0)),
        hours_since_last_success=float(txn_features.get("hours_since_last_success", -1.0)),
        subscription_flag=bool(txn_features.get("subscription_flag", False)),
        action_scores=action_scores,
        merchant_policy_summary=merchant_policy_summary,
    )


def propose_recovery_action(
    txn_features: dict,
    ml_scores: dict[str, float],
    merchant_policy_summary: str = "Default policy: max 3 recovery attempts, standard cooldown.",
) -> tuple[AgentInput, AgentProposal | AgentFailure]:
    """
    End-to-end recovery-action proposal.

    1. Build agent input (with deterministic expected-recovery values).
    2. Call Gemini for reasoning.
    3. Return (input, proposal_or_failure).

    The caller (or a future policy engine) decides what to do with the
    proposal — this function never executes an action.
    """
    agent_input = build_agent_input(txn_features, ml_scores, merchant_policy_summary)

    logger.info(
        "Requesting recovery proposal for txn=%s amount=%.2f",
        agent_input.transaction_id,
        agent_input.amount,
    )

    result = call_gemini(agent_input)

    if isinstance(result, AgentProposal):
        logger.info(
            "Proposal for txn=%s: action=%s confidence=%.2f",
            agent_input.transaction_id,
            result.recommended_action.value,
            result.confidence,
        )
    else:
        logger.warning(
            "Agent failure for txn=%s: %s — %s",
            agent_input.transaction_id,
            result.error_type,
            result.error_message,
        )

    return agent_input, result
