#!/usr/bin/env python3
"""
Demo script for the Deterministic Policy Engine (Phase 5).

Demonstrates:
  1. CASE 1 — ALLOW
  2. CASE 2 — MODIFY
  3. CASE 3 — BLOCK
  4. CASE 4 — Gemini Failure -> ML Fallback
  5. Generating an Audit Record
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.schemas import AgentFailure, AgentProposal, AllowedAction, Priority
from app.policy.engine import make_policy_decision
from app.policy.rules import RuleContext
from app.policy.schemas import MerchantPolicy, PolicyDecision

def print_decision(case_name: str, decision: PolicyDecision):
    print(f"\n{'=' * 60}")
    print(f"  {case_name}")
    print(f"{'=' * 60}")
    
    print(f"Decision:         {decision.decision.value}")
    print(f"Decision Source:  {decision.decision_source.value}")
    print(f"Proposed Action:  {decision.proposed_action.value}")
    print(f"Final Action:     {decision.final_action.value if decision.final_action else 'None'}")
    print(f"Reason:           {decision.reason}")
    
    print("\nExpected Recoveries (App Calculated):")
    for action in AllowedAction:
        er = decision.expected_recovery.get(action, 0.0)
        enr = decision.expected_net_recovery.get(action, 0.0)
        print(f"  {action.value:<16}: ER=₹{er:,.2f} | ENR=₹{enr:,.2f}")
        
    print("\nPolicy Checks (Failures only):")
    failed = [r for r in decision.rule_results if not r.passed]
    if failed:
        for r in failed:
            print(f"  [X] {r.rule_name}: {r.message}")
    else:
        print("  All policy checks passed.")


def generate_audit_record(decision: PolicyDecision, agent_result) -> dict:
    """Format the decision into a structured audit log dictionary."""
    from datetime import timezone
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor_type": "POLICY_ENGINE",
        "event_type": "RECOVERY_DECISION",
        "decision_source": decision.decision_source.value,
        "ml_expected_recovery": {k.value: v for k, v in decision.expected_recovery.items()},
        "ml_expected_net_recovery": {k.value: v for k, v in decision.expected_net_recovery.items()},
        "policy_decision": decision.decision.value,
        "proposed_action": decision.proposed_action.value,
        "final_action": decision.final_action.value if decision.final_action else None,
        "deterministic_reason": decision.reason,
        "failed_rules": [{"rule": r.rule_name, "msg": r.message} for r in decision.rule_results if not r.passed]
    }
    
    if isinstance(agent_result, AgentProposal):
        record["llm_reason"] = agent_result.reason
        record["llm_confidence"] = agent_result.confidence
    else:
        record["llm_error"] = agent_result.error_message
        
    return record


def main():
    txn_features = {"amount": 12500.0}
    ml_scores = {
        "PAYMENT_LINK": 0.61,
        "REMINDER": 0.34,
        "DELAYED_RETRY": 0.79,
        "ESCALATE": 0.11,
    }
    
    # ── CASE 1: ALLOW ──────────────────────────────────────────────
    ctx_allow = RuleContext(
        payment_status="FAILED",
        amount=12500.0,
        attempt_number=1,
        hours_since_last_action=48.0,
        days_since_failure=1.0,
        customer_actions_this_month=1,
        has_active_recovery=False,
    )
    gemini_allow = AgentProposal(
        recommended_action=AllowedAction.DELAYED_RETRY,
        priority=Priority.HIGH,
        reason="Temporary bank timeout on first attempt.",
        confidence=0.89
    )
    decision1 = make_policy_decision(txn_features, ml_scores, gemini_allow, ctx_allow)
    print_decision("CASE 1 — ALLOW", decision1)


    # ── CASE 2: MODIFY ─────────────────────────────────────────────
    ctx_modify = ctx_allow.model_copy(update={"attempt_number": 3})
    # Gemini proposes DELAYED_RETRY, but attempt_number=3 restricts to ESCALATE.
    gemini_modify = AgentProposal(
        recommended_action=AllowedAction.DELAYED_RETRY,
        priority=Priority.HIGH,
        reason="Retrying again.",
        confidence=0.80
    )
    decision2 = make_policy_decision(txn_features, ml_scores, gemini_modify, ctx_modify)
    print_decision("CASE 2 — MODIFY (Exceeded automated retry limit)", decision2)


    # ── CASE 3: BLOCK ──────────────────────────────────────────────
    ctx_block = ctx_allow.model_copy(update={"payment_status": "SUCCESS"})
    gemini_block = AgentProposal(
        recommended_action=AllowedAction.PAYMENT_LINK,
        priority=Priority.MEDIUM,
        reason="Send payment link.",
        confidence=0.6
    )
    decision3 = make_policy_decision(txn_features, ml_scores, gemini_block, ctx_block)
    print_decision("CASE 3 — BLOCK (Payment already successful)", decision3)


    # ── CASE 4: ML FALLBACK ─────────────────────────────────────────
    agent_failure = AgentFailure(
        error_type="timeout", 
        error_message="Gemini API timed out"
    )
    decision4 = make_policy_decision(txn_features, ml_scores, agent_failure, ctx_allow)
    print_decision("CASE 4 — ML FALLBACK (Gemini API failed)", decision4)

    
    # ── AUDIT LOG EXAMPLE ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  EXAMPLE AUDIT RECORD (JSONB metadata for DB)")
    print(f"{'=' * 60}")
    audit = generate_audit_record(decision2, gemini_modify)
    print(json.dumps(audit, indent=2))
    print()


if __name__ == "__main__":
    main()
