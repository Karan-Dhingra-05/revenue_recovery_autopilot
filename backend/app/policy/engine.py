"""
Policy Engine Orchestrator.

Combines the expected net recovery scoring with the deterministic rules
to output a final ALLOW, MODIFY, or BLOCK decision. Generates the structured
PolicyDecision that can be stored in the audit log.
"""

from __future__ import annotations

import logging

from app.agent.schemas import AgentFailure, AgentProposal, AllowedAction
from app.policy.rules import ALL_RULES, RuleContext
from app.policy.schemas import (
    DecisionSource,
    DecisionType,
    MerchantPolicy,
    PolicyDecision,
    RuleResult,
)
from app.policy.scoring import calculate_economics, get_ml_fallback_action

logger = logging.getLogger(__name__)


def evaluate_action(
    action: AllowedAction, 
    ctx: RuleContext, 
    policy: MerchantPolicy
) -> tuple[bool, list[RuleResult], str]:
    """
    Evaluate all rules for a given action.
    Returns (is_allowed, rule_results, failure_reason_if_any).
    """
    results = []
    failed_reason = ""
    for rule in ALL_RULES:
        res = rule(ctx, action, policy)
        results.append(res)
        if not res.passed and not failed_reason:
            failed_reason = res.message

    is_allowed = all(r.passed for r in results)
    return is_allowed, results, failed_reason


def make_policy_decision(
    txn_features: dict,
    ml_scores: dict[str, float],
    agent_result: AgentProposal | AgentFailure,
    rule_context: RuleContext,
    policy: MerchantPolicy | None = None,
) -> PolicyDecision:
    """
    Make the final, authoritative policy decision.
    
    1. Calculate economics based on ML probabilities and policy costs.
    2. Determine the proposed action and decision source (LLM or ML fallback).
    3. Evaluate the proposed action.
    4. If it fails, attempt to find a safe alternative (MODIFY) or else BLOCK.
    """
    if policy is None:
        policy = MerchantPolicy()
        
    amount = float(txn_features.get("amount", 0.0))
    expected_recovery, expected_net_recovery = calculate_economics(amount, ml_scores, policy)
    
    # ── Determine Proposed Action & Source ──
    if isinstance(agent_result, AgentProposal):
        proposed_action = agent_result.recommended_action
        decision_source = DecisionSource.GEMINI
    else:
        # LLM Failed — Fallback to ML's highest net recovery
        proposed_action = get_ml_fallback_action(expected_net_recovery)
        decision_source = DecisionSource.ML_FALLBACK
        logger.warning(
            "AgentFailure detected (%s), using ML fallback: %s",
            agent_result.error_type,
            proposed_action.value
        )
        
    # ── Evaluate Proposed Action ──
    is_allowed, results, failed_reason = evaluate_action(proposed_action, rule_context, policy)
    
    if is_allowed:
        return PolicyDecision(
            decision=DecisionType.ALLOW,
            decision_source=decision_source,
            proposed_action=proposed_action,
            final_action=proposed_action,
            reason="All recovery policies passed.",
            rule_results=results,
            expected_recovery=expected_recovery,
            expected_net_recovery=expected_net_recovery,
        )
        
    # ── Proposed Action Failed — Can we MODIFY? ──
    # Check alternatives ordered by expected net recovery
    sorted_alternatives = sorted(
        expected_net_recovery.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    
    for alt_action, _ in sorted_alternatives:
        if alt_action == proposed_action:
            continue
            
        alt_is_allowed, alt_results, _ = evaluate_action(alt_action, rule_context, policy)
        if alt_is_allowed:
            return PolicyDecision(
                decision=DecisionType.MODIFY,
                decision_source=decision_source,
                proposed_action=proposed_action,
                final_action=alt_action,
                reason=f"Proposed action {proposed_action.value} blocked ({failed_reason}). Modified to safe alternative {alt_action.value}.",
                rule_results=results,  # Return the results that failed the original proposal
                expected_recovery=expected_recovery,
                expected_net_recovery=expected_net_recovery,
            )
            
    # ── No alternative is safe — BLOCK ──
    return PolicyDecision(
        decision=DecisionType.BLOCK,
        decision_source=decision_source,
        proposed_action=proposed_action,
        final_action=None,
        reason=f"No automated recovery action is permitted. Primary block reason: {failed_reason}",
        rule_results=results,
        expected_recovery=expected_recovery,
        expected_net_recovery=expected_net_recovery,
    )
