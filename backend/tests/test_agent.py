"""
Tests for the recovery agent (Phase 4).

All Gemini API calls are mocked — no real API quota is consumed.

Tests cover:
  1. Valid structured response
  2. Invalid action
  3. Missing required fields
  4. Confidence < 0
  5. Confidence > 1
  6. Empty reason
  7. Malformed JSON
  8. Gemini timeout
  9. Gemini API error
  10. Correct context passed to LLM
  11. Correct ML scores passed
  12. Expected recovery calculated by app code
  13. Agent cannot directly execute payment actions
  14. Fallback behaviour is safe
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agent.gemini_client import parse_gemini_response
from app.agent.schemas import (
    ActionScore,
    AgentFailure,
    AgentInput,
    AgentProposal,
    AllowedAction,
    Priority,
)
from app.agent.service import ACTIONS, build_agent_input, propose_recovery_action


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_txn_features() -> dict:
    """A realistic failed transaction for testing."""
    return {
        "transaction_id": "txn_test_001",
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


@pytest.fixture()
def sample_ml_scores() -> dict[str, float]:
    """ML scores matching the demo scenario."""
    return {
        "PAYMENT_LINK": 0.61,
        "REMINDER": 0.34,
        "DELAYED_RETRY": 0.79,
        "ESCALATE": 0.11,
    }


VALID_RESPONSE_JSON = json.dumps(
    {
        "recommended_action": "DELAYED_RETRY",
        "priority": "HIGH",
        "reason": "Temporary bank timeout with strong customer history favours a delayed retry.",
        "confidence": 0.86,
    }
)


# ── 1. Valid structured response ─────────────────────────────────────────────


def test_valid_response_parsed_correctly():
    result = parse_gemini_response(VALID_RESPONSE_JSON)
    assert isinstance(result, AgentProposal)
    assert result.recommended_action == AllowedAction.DELAYED_RETRY
    assert result.priority == Priority.HIGH
    assert 0 <= result.confidence <= 1
    assert len(result.reason) > 0


def test_valid_response_with_code_fences():
    """Gemini sometimes wraps JSON in markdown code fences."""
    fenced = f"```json\n{VALID_RESPONSE_JSON}\n```"
    result = parse_gemini_response(fenced)
    assert isinstance(result, AgentProposal)
    assert result.recommended_action == AllowedAction.DELAYED_RETRY


# ── 2. Invalid action ───────────────────────────────────────────────────────


def test_invalid_action_rejected():
    bad = json.dumps({
        "recommended_action": "REFUND",
        "priority": "HIGH",
        "reason": "Should refund.",
        "confidence": 0.5,
    })
    result = parse_gemini_response(bad)
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


def test_stop_action_rejected():
    """STOP must not be accepted as an LLM action."""
    bad = json.dumps({
        "recommended_action": "STOP",
        "priority": "LOW",
        "reason": "Should stop.",
        "confidence": 0.3,
    })
    result = parse_gemini_response(bad)
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


# ── 3. Missing required fields ──────────────────────────────────────────────


@pytest.mark.parametrize("missing_field", ["recommended_action", "priority", "reason", "confidence"])
def test_missing_field_rejected(missing_field: str):
    data = {
        "recommended_action": "PAYMENT_LINK",
        "priority": "MEDIUM",
        "reason": "Test reason.",
        "confidence": 0.7,
    }
    del data[missing_field]
    result = parse_gemini_response(json.dumps(data))
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


# ── 4. Confidence below 0 ───────────────────────────────────────────────────


def test_confidence_below_zero_rejected():
    bad = json.dumps({
        "recommended_action": "PAYMENT_LINK",
        "priority": "HIGH",
        "reason": "Negative confidence.",
        "confidence": -0.5,
    })
    result = parse_gemini_response(bad)
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


# ── 5. Confidence above 1 ───────────────────────────────────────────────────


def test_confidence_above_one_rejected():
    bad = json.dumps({
        "recommended_action": "PAYMENT_LINK",
        "priority": "HIGH",
        "reason": "Overconfident.",
        "confidence": 1.5,
    })
    result = parse_gemini_response(bad)
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


# ── 6. Empty reason ─────────────────────────────────────────────────────────


def test_empty_reason_rejected():
    bad = json.dumps({
        "recommended_action": "PAYMENT_LINK",
        "priority": "HIGH",
        "reason": "",
        "confidence": 0.5,
    })
    result = parse_gemini_response(bad)
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


def test_whitespace_only_reason_rejected():
    bad = json.dumps({
        "recommended_action": "PAYMENT_LINK",
        "priority": "HIGH",
        "reason": "   ",
        "confidence": 0.5,
    })
    result = parse_gemini_response(bad)
    assert isinstance(result, AgentFailure)
    assert result.error_type == "validation_error"


# ── 7. Malformed JSON ───────────────────────────────────────────────────────


def test_malformed_json_returns_failure():
    result = parse_gemini_response("this is not json at all")
    assert isinstance(result, AgentFailure)
    assert result.error_type == "malformed_json"


def test_json_array_rejected():
    result = parse_gemini_response('[{"recommended_action": "PAYMENT_LINK"}]')
    assert isinstance(result, AgentFailure)
    assert result.error_type == "malformed_json"


def test_empty_string_returns_failure():
    result = parse_gemini_response("")
    assert isinstance(result, AgentFailure)
    assert result.error_type == "malformed_json"


# ── 8. Gemini timeout ───────────────────────────────────────────────────────


def test_timeout_returns_failure(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)

    with patch("app.agent.gemini_client._get_client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = (
            Exception("Deadline exceeded timeout")
        )
        from app.agent.gemini_client import call_gemini
        result = call_gemini(agent_input)

    assert isinstance(result, AgentFailure)
    assert result.error_type == "timeout"


# ── 9. Gemini API error ─────────────────────────────────────────────────────


def test_api_error_returns_failure(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)

    with patch("app.agent.gemini_client._get_client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = (
            Exception("Internal server error")
        )
        from app.agent.gemini_client import call_gemini
        result = call_gemini(agent_input)

    assert isinstance(result, AgentFailure)
    assert result.error_type == "api_error"


def test_rate_limit_returns_failure(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)

    with patch("app.agent.gemini_client._get_client") as mock_client:
        mock_client.return_value.models.generate_content.side_effect = (
            Exception("Resource exhausted: rate limit quota exceeded")
        )
        from app.agent.gemini_client import call_gemini
        result = call_gemini(agent_input)

    assert isinstance(result, AgentFailure)
    assert result.error_type == "rate_limit"


# ── 10. Correct context passed to LLM ───────────────────────────────────────


def test_agent_input_contains_transaction_context(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)
    assert agent_input.transaction_id == "txn_test_001"
    assert agent_input.amount == 12500.00
    assert agent_input.payment_method == "CARD"
    assert agent_input.failure_reason == "BANK_TIMEOUT"
    assert agent_input.failure_pattern == "A_TEMP_BANK"
    assert agent_input.attempt_number == 1
    assert agent_input.customer_success_rate == 0.91


# ── 11. Correct ML scores passed ────────────────────────────────────────────


def test_ml_scores_correctly_embedded(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)
    scores_by_action = {s.action.value: s.probability for s in agent_input.action_scores}
    assert scores_by_action["PAYMENT_LINK"] == 0.61
    assert scores_by_action["REMINDER"] == 0.34
    assert scores_by_action["DELAYED_RETRY"] == 0.79
    assert scores_by_action["ESCALATE"] == 0.11


def test_all_four_actions_scored(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)
    scored_actions = {s.action.value for s in agent_input.action_scores}
    assert scored_actions == set(ACTIONS)


# ── 12. Expected recovery calculated by app code ────────────────────────────


def test_expected_recovery_computed_deterministically(sample_txn_features, sample_ml_scores):
    """Expected recovery = probability × amount, calculated in app code."""
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)
    for score in agent_input.action_scores:
        expected = round(score.probability * sample_txn_features["amount"], 2)
        assert score.expected_recovery == expected, (
            f"{score.action.value}: expected_recovery should be "
            f"{expected}, got {score.expected_recovery}"
        )


def test_expected_recovery_values_are_specific():
    """Verify specific expected recovery values for the demo scenario."""
    txn = {"amount": 12500.0, "transaction_id": "t1"}
    ml = {"PAYMENT_LINK": 0.61, "REMINDER": 0.34, "DELAYED_RETRY": 0.79, "ESCALATE": 0.11}
    agent_input = build_agent_input(txn, ml)
    er_map = {s.action.value: s.expected_recovery for s in agent_input.action_scores}
    assert er_map["PAYMENT_LINK"] == 7625.0
    assert er_map["REMINDER"] == 4250.0
    assert er_map["DELAYED_RETRY"] == 9875.0
    assert er_map["ESCALATE"] == 1375.0


# ── 13. Agent cannot execute payment actions ─────────────────────────────────


def test_agent_service_does_not_execute():
    """
    The service function returns (input, proposal) — it never calls
    any payment API, database write, or external action.
    """
    # The service module must not import or reference any execution module.
    import app.agent.service as svc
    source = open(svc.__file__).read()
    # These patterns should never appear in the service module.
    forbidden = [
        "create_payment_link",
        "send_notification",
        "send_reminder",
        "execute_action",
        "razorpay",
    ]
    for pattern in forbidden:
        assert pattern not in source, (
            f"agent/service.py must not reference '{pattern}' — "
            f"the agent is a proposal layer, not an execution layer."
        )


# ── 14. Fallback behaviour is safe ──────────────────────────────────────────


def test_fallback_on_invalid_json_is_agent_failure():
    result = parse_gemini_response("{broken json")
    assert isinstance(result, AgentFailure)
    assert result.error_type == "malformed_json"
    # The failure must contain the raw response for debugging.
    assert result.raw_response is not None


def test_fallback_on_missing_api_key(sample_txn_features, sample_ml_scores):
    agent_input = build_agent_input(sample_txn_features, sample_ml_scores)

    with patch("app.agent.gemini_client.settings") as mock_settings:
        mock_settings.gemini_api_key = ""
        from app.agent.gemini_client import call_gemini
        result = call_gemini(agent_input)

    assert isinstance(result, AgentFailure)
    assert result.error_type == "config_error"


def test_propose_recovery_action_with_mocked_gemini(sample_txn_features, sample_ml_scores):
    """Full end-to-end with mocked Gemini returning valid JSON."""
    mock_response = MagicMock()
    mock_response.text = VALID_RESPONSE_JSON

    with patch("app.agent.gemini_client._get_client") as mock_client:
        mock_client.return_value.models.generate_content.return_value = mock_response
        agent_input, result = propose_recovery_action(
            sample_txn_features, sample_ml_scores
        )

    assert isinstance(result, AgentProposal)
    assert result.recommended_action == AllowedAction.DELAYED_RETRY
    assert result.confidence == 0.86
    # Verify input was built correctly.
    assert agent_input.amount == 12500.00
    assert len(agent_input.action_scores) == 4


def test_propose_recovery_action_with_gemini_failure(sample_txn_features, sample_ml_scores):
    """Full end-to-end with Gemini returning malformed output."""
    mock_response = MagicMock()
    mock_response.text = "I'm sorry, I can't help with that."

    with patch("app.agent.gemini_client._get_client") as mock_client:
        mock_client.return_value.models.generate_content.return_value = mock_response
        agent_input, result = propose_recovery_action(
            sample_txn_features, sample_ml_scores
        )

    assert isinstance(result, AgentFailure)
    assert result.error_type == "malformed_json"
    # The agent input should still have been built correctly.
    assert agent_input.amount == 12500.00
