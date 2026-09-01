#!/usr/bin/env python3
"""
Demo script for the recovery agent.

Demonstrates:
  1. Building structured agent input from transaction context + ML scores
  2. Calling Gemini 2.5 Flash for a recovery proposal (if API key is set)
  3. Falling back gracefully if the API key is not configured
  4. Showing the exact structured JSON at each stage

Usage
-----
    # With API key set:
    GEMINI_API_KEY=your_key python scripts/demo_agent.py

    # Without API key (shows mock fallback):
    python scripts/demo_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.gemini_client import call_gemini, parse_gemini_response
from app.agent.schemas import AgentFailure, AgentProposal
from app.agent.service import build_agent_input, propose_recovery_action
from app.config import settings


def _print_json(label: str, obj) -> None:
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
        # Convert enums to strings for clean display
        for k, v in data.items():
            if hasattr(v, "value"):
                data[k] = v.value
            if isinstance(v, list):
                data[k] = [
                    {kk: vv.value if hasattr(vv, "value") else vv for kk, vv in item.items()}
                    if isinstance(item, dict)
                    else (item.model_dump() if hasattr(item, "model_dump") else item)
                    for item in v
                ]
        print(f"\n{label}:")
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f"\n{label}:")
        print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    # ── Demo transaction: bank timeout on a high-value recurring customer ──
    txn_features = {
        "transaction_id": "txn_demo_001",
        "amount": 12500.00,
        "currency": "INR",
        "payment_method": "CARD",
        "failure_reason": "BANK_TIMEOUT",
        "failure_pattern": "A_TEMP_BANK",
        "attempt_number": 1,
        "customer_success_rate": 0.91,
        "customer_previous_failures": 1,
        "customer_previous_recoveries": 0,
        "hours_since_last_success": 48.5,
        "subscription_flag": 0,
    }

    # Simulated ML scores (from LightGBM — not computed by the LLM).
    ml_scores = {
        "PAYMENT_LINK": 0.61,
        "REMINDER": 0.34,
        "DELAYED_RETRY": 0.79,
        "ESCALATE": 0.11,
    }

    print("=" * 60)
    print("  Revenue Recovery Agent — Demo")
    print("=" * 60)

    # Step 1: Build agent input
    agent_input = build_agent_input(txn_features, ml_scores)
    _print_json("Step 1 — Agent Input (sent to LLM)", agent_input)

    # Step 2: Verify expected recovery is calculated by app code
    print("\nStep 2 — Expected Recovery (calculated by app code, NOT LLM):")
    for score in agent_input.action_scores:
        print(
            f"  {score.action.value:<16}  "
            f"P = {score.probability:.4f}  ×  ₹{agent_input.amount:,.2f}  "
            f"=  ₹{score.expected_recovery:,.2f}"
        )

    # Step 3: Call Gemini (or demonstrate fallback)
    print("\n" + "=" * 60)
    if settings.gemini_api_key:
        print("  Step 3 — Calling Gemini 2.5 Flash…")
        print(f"  Model: {settings.gemini_model}")
        print("=" * 60)

        agent_input, result = propose_recovery_action(txn_features, ml_scores)

        if isinstance(result, AgentProposal):
            _print_json("✓ Agent Proposal (from Gemini)", result)
        else:
            _print_json("⚠ Agent Failure", result)
    else:
        print("  Step 3 — GEMINI_API_KEY not set, demonstrating mock response")
        print("=" * 60)

        # Show what a valid response looks like
        mock_json = json.dumps({
            "recommended_action": "DELAYED_RETRY",
            "priority": "HIGH",
            "reason": (
                "The failure appears temporary (bank timeout) and the "
                "customer has a strong success rate (91%). The ML model "
                "assigns the highest recovery probability to DELAYED_RETRY "
                "(0.79), with an expected recovery of ₹9,875. This is the "
                "first attempt, so a retry is well justified."
            ),
            "confidence": 0.86,
        })
        result = parse_gemini_response(mock_json)
        _print_json("✓ Mock Agent Proposal (simulated)", result)

        # Also demonstrate a malformed fallback
        print("\n" + "-" * 60)
        print("  Demonstrating malformed-response fallback:")
        print("-" * 60)
        bad_result = parse_gemini_response("I cannot process this request.")
        _print_json("⚠ Fallback result (malformed LLM output)", bad_result)

    print()


if __name__ == "__main__":
    main()
