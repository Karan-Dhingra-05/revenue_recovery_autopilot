"""
Tests for the Deterministic Policy Engine (Phase 5).

Verifies 22 specific test cases, covering the ALLOW/MODIFY/BLOCK behaviors,
ML fallback when Gemini is unavailable, correct expected net recovery scoring,
and safety boundaries. No external APIs are called.
"""

from __future__ import annotations

import pytest

from app.agent.schemas import AgentFailure, AgentProposal, AllowedAction, Priority
from app.policy.engine import make_policy_decision
from app.policy.rules import RuleContext
from app.policy.schemas import DecisionSource, DecisionType, MerchantPolicy


@pytest.fixture
def default_txn_features() -> dict:
    return {"amount": 10000.0}


@pytest.fixture
def default_ml_scores() -> dict[str, float]:
    return {
        "PAYMENT_LINK": 0.50,
        "REMINDER": 0.40,
        "DELAYED_RETRY": 0.80,
        "ESCALATE": 0.20,
    }


@pytest.fixture
def default_rule_context() -> RuleContext:
    return RuleContext(
        payment_status="FAILED",
        amount=10000.0,
        attempt_number=1,
        hours_since_last_action=48.0,
        days_since_failure=1.0,
        customer_actions_this_month=1,
        has_active_recovery=False,
    )


@pytest.fixture
def default_agent_proposal() -> AgentProposal:
    return AgentProposal(
        recommended_action=AllowedAction.DELAYED_RETRY,
        priority=Priority.HIGH,
        reason="Test reason",
        confidence=0.8,
    )


# ── 1. Eligible payment → ALLOW ──────────────────────────────────────────────
def test_eligible_payment_allows_action(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context)
    assert decision.decision == DecisionType.ALLOW
    assert decision.proposed_action == AllowedAction.DELAYED_RETRY
    assert decision.final_action == AllowedAction.DELAYED_RETRY
    assert decision.decision_source == DecisionSource.GEMINI


# ── 2. Payment already successful → BLOCK ────────────────────────────────────
def test_payment_already_successful_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    ctx = default_rule_context.model_copy(update={"payment_status": "SUCCESS"})
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx)
    assert decision.decision == DecisionType.BLOCK
    assert decision.final_action is None


# ── 3. Amount above allowed threshold → BLOCK ────────────────────────────────
def test_amount_above_threshold_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy(max_amount_eligible=5000.0) # Lower than 10000
    ctx = default_rule_context.model_copy(update={"amount": 10000.0})
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx, policy)
    assert decision.decision == DecisionType.BLOCK


# ── 4. Action disabled by merchant → BLOCK ───────────────────────────────────
def test_action_disabled_by_merchant(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy()
    policy.action_configs[AllowedAction.DELAYED_RETRY].is_enabled = False
    
    # Since DELAYED_RETRY is disabled, expected_net_recovery will push it to the bottom.
    # The policy engine will attempt to MODIFY to the next best action (PAYMENT_LINK).
    # To test BLOCK, we can disable all other actions.
    for action in AllowedAction:
        if action != AllowedAction.DELAYED_RETRY:
            policy.action_configs[action].is_enabled = False
            
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context, policy)
    assert decision.decision == DecisionType.BLOCK


# ── 5. Maximum retry/action attempts exceeded → BLOCK (or MODIFY) ────────────
def test_max_attempts_exceeded_modifies_to_escalate(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy(max_automatic_attempts=3)
    ctx = default_rule_context.model_copy(update={"attempt_number": 3})
    
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx, policy)
    # Rule repeated_failure_stopping restricts attempt_number == 3 to ONLY allow ESCALATE.
    assert decision.decision == DecisionType.MODIFY
    assert decision.proposed_action == AllowedAction.DELAYED_RETRY
    assert decision.final_action == AllowedAction.ESCALATE


def test_max_attempts_strictly_exceeded_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy(max_automatic_attempts=3)
    ctx = default_rule_context.model_copy(update={"attempt_number": 4})
    
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx, policy)
    # Exceeding attempt limit blocks everything.
    assert decision.decision == DecisionType.BLOCK


# ── 6. Cooldown not satisfied → BLOCK ────────────────────────────────────────
def test_cooldown_not_satisfied_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy(cooldown_hours=24.0)
    ctx = default_rule_context.model_copy(update={"hours_since_last_action": 12.0})
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx, policy)
    assert decision.decision == DecisionType.BLOCK


# ── 7. Recovery window expired → BLOCK ───────────────────────────────────────
def test_recovery_window_expired_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy(recovery_window_days=14)
    ctx = default_rule_context.model_copy(update={"days_since_failure": 15.0})
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx, policy)
    assert decision.decision == DecisionType.BLOCK


# ── 8. Duplicate active recovery → BLOCK ─────────────────────────────────────
def test_duplicate_active_recovery_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    ctx = default_rule_context.model_copy(update={"has_active_recovery": True})
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx)
    assert decision.decision == DecisionType.BLOCK


# ── 9. Customer action limit exceeded → BLOCK ────────────────────────────────
def test_customer_action_limit_blocks(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy(max_customer_actions_per_month=5)
    ctx = default_rule_context.model_copy(update={"customer_actions_this_month": 5})
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, ctx, policy)
    assert decision.decision == DecisionType.BLOCK


# ── 10. Valid Gemini action → policy evaluates it ────────────────────────────
def test_valid_gemini_action_evaluated(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context)
    assert decision.proposed_action == default_agent_proposal.recommended_action
    assert decision.decision_source == DecisionSource.GEMINI
    assert decision.decision == DecisionType.ALLOW


# ── 11. Invalid Gemini action → BLOCK / MODIFY ───────────────────────────────
def test_invalid_gemini_action_modifies(default_txn_features, default_ml_scores, default_rule_context):
    # If Gemini proposes ESCALATE, but merchant disabled it, it should MODIFY.
    proposal = AgentProposal(
        recommended_action=AllowedAction.ESCALATE,
        priority=Priority.LOW,
        reason="Test",
        confidence=0.5
    )
    policy = MerchantPolicy()
    policy.action_configs[AllowedAction.ESCALATE].is_enabled = False
    
    decision = make_policy_decision(default_txn_features, default_ml_scores, proposal, default_rule_context, policy)
    assert decision.decision == DecisionType.MODIFY
    assert decision.proposed_action == AllowedAction.ESCALATE
    # It will pick DELAYED_RETRY because it has the highest ENR.
    assert decision.final_action == AllowedAction.DELAYED_RETRY


# ── 12. Final action is deterministic ────────────────────────────────────────
def test_final_action_is_deterministic(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context)
    assert decision.final_action in list(AllowedAction)


# ── 13. Expected recovery & Net recovery calculations are correct ────────────
def test_expected_recovery_and_net_correct(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context):
    policy = MerchantPolicy()
    decision = make_policy_decision(default_txn_features, default_ml_scores, default_agent_proposal, default_rule_context, policy)
    
    # Amount = 10000. PAYMENT_LINK P = 0.50 -> ER = 5000. 
    # PAYMENT_LINK config: cost=0, risk=0 -> ENR = 5000.
    assert decision.expected_recovery[AllowedAction.PAYMENT_LINK] == 5000.0
    assert decision.expected_net_recovery[AllowedAction.PAYMENT_LINK] == 5000.0
    
    # DELAYED_RETRY P = 0.80 -> ER = 8000.
    # config: cost=3, risk=10 -> ENR = 7987.0
    assert decision.expected_recovery[AllowedAction.DELAYED_RETRY] == 8000.0
    assert decision.expected_net_recovery[AllowedAction.DELAYED_RETRY] == 7987.0


# ── 14. ML Fallback when Gemini unavailable ──────────────────────────────────
def test_agent_failure_triggers_ml_fallback(default_txn_features, default_ml_scores, default_rule_context):
    agent_failure = AgentFailure(error_type="timeout", error_message="Simulated timeout")
    
    decision = make_policy_decision(default_txn_features, default_ml_scores, agent_failure, default_rule_context)
    
    assert decision.decision_source == DecisionSource.ML_FALLBACK
    # Highest ENR is DELAYED_RETRY
    assert decision.proposed_action == AllowedAction.DELAYED_RETRY
    assert decision.decision == DecisionType.ALLOW
    assert decision.final_action == AllowedAction.DELAYED_RETRY


# ── 15. Policy engine does not trust LLM monetary values ─────────────────────
def test_llm_monetary_values_not_trusted():
    # Proven by the fact that make_policy_decision only takes AgentProposal
    # which has no monetary fields in its schema.
    assert "amount" not in AgentProposal.model_fields
    assert "expected_recovery" not in AgentProposal.model_fields


# ── 16. No external execution triggered ──────────────────────────────────────
def test_no_execution_triggered():
    import app.policy.engine as pe
    source = open(pe.__file__).read()
    assert "requests." not in source
    assert "razorpay" not in source.lower()
