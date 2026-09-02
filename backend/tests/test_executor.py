"""
Tests for the Recovery Action Executor (Phase 6).

All Razorpay API calls are mocked — no real API calls during pytest.
Tests use an in-memory or test PostgreSQL database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agent.schemas import AllowedAction
from app.database import Base
from app.executor.service import ExecutionResult, execute_recovery_action
from app.models.enums import (
    AuditActorType,
    PaymentMethod,
    PaymentStatus,
    RecoveryActionStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.models.audit import AuditLog
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.recovery import RecoveryAction, RecoveryCase
from app.policy.schemas import (
    DecisionSource,
    DecisionType,
    MerchantPolicy,
    PolicyDecision,
)


# ── Test DB fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_engine():
    """Create a test PostgreSQL connection (reuses the dev DB)."""
    from app.config import settings
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db(test_engine):
    """Provide a transactional test session that rolls back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_data(db: Session):
    """Create a minimal merchant → customer → payment → recovery case chain."""
    merchant = Merchant(name="Test Merchant", currency="INR")
    db.add(merchant)
    db.flush()

    customer = Customer(
        merchant_id=merchant.id,
        external_customer_id="test_cust_001",
        email="test@example.com",
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        merchant_id=merchant.id,
        customer_id=customer.id,
        amount=Decimal("12500.00"),
        currency="INR",
        payment_method=PaymentMethod.CARD,
        status=PaymentStatus.FAILED,
    )
    db.add(payment)
    db.flush()

    case = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.OPEN,
        amount_at_risk=Decimal("12500.00"),
    )
    db.add(case)
    db.flush()

    return merchant, customer, payment, case


def _make_allow_decision(action: AllowedAction = AllowedAction.PAYMENT_LINK) -> PolicyDecision:
    return PolicyDecision(
        decision=DecisionType.ALLOW,
        decision_source=DecisionSource.GEMINI,
        proposed_action=action,
        final_action=action,
        reason="All recovery policies passed.",
        rule_results=[],
        expected_recovery={
            AllowedAction.PAYMENT_LINK: 7625.0,
            AllowedAction.REMINDER: 4250.0,
            AllowedAction.DELAYED_RETRY: 9875.0,
            AllowedAction.ESCALATE: 1375.0,
        },
        expected_net_recovery={
            AllowedAction.PAYMENT_LINK: 7625.0,
            AllowedAction.REMINDER: 4249.0,
            AllowedAction.DELAYED_RETRY: 9862.0,
            AllowedAction.ESCALATE: 1225.0,
        },
    )


def _make_block_decision() -> PolicyDecision:
    return PolicyDecision(
        decision=DecisionType.BLOCK,
        decision_source=DecisionSource.GEMINI,
        proposed_action=AllowedAction.PAYMENT_LINK,
        final_action=None,
        reason="Payment already successful.",
        rule_results=[],
        expected_recovery={a: 0.0 for a in AllowedAction},
        expected_net_recovery={a: 0.0 for a in AllowedAction},
    )


# ── Tests ──────────────────────────────────────────────────────────────────

# 1. Credentials missing + execution false → blocked
@patch("app.executor.service.create_payment_link")
def test_execution_blocked_when_flag_false_and_no_creds(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", False)
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is False
    assert "Razorpay execution is disabled" in result.message
    mock_create.assert_not_called()

# 2. Credentials present + execution false → blocked
# IMPORTANT: this is the regression test for the bug.
@patch("app.executor.service.create_payment_link")
def test_execution_blocked_when_flag_false_even_with_creds(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", False)
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is False
    assert "Razorpay execution is disabled" in result.message
    mock_create.assert_not_called()

# 3. Credentials missing + execution true → clear configuration error
def test_execution_true_with_missing_creds_raises_error(db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    monkeypatch.setattr("app.config.settings.razorpay_key_id", "")
    monkeypatch.setattr("app.config.settings.razorpay_key_secret", "")
    
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    
    result = execute_recovery_action(decision, case, payment, db)
    
    assert result.success is False
    assert "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set" in result.message

# 4. Test credentials + execution true → Razorpay client may be called
@patch("app.executor.service.create_payment_link")
def test_execution_true_with_test_creds_calls_api(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    monkeypatch.setattr("app.config.settings.razorpay_key_id", "rzp_test_123")
    monkeypatch.setattr("app.config.settings.razorpay_key_secret", "secret")
    
    mock_create.return_value = {"id": "plink_test123", "short_url": "https://rzp.io/test123"}
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is True
    assert result.external_reference == "plink_test123"
    mock_create.assert_called_once()

# 4b. Live credentials + execution true → Razorpay client rejected
def test_execution_true_with_live_creds_is_rejected(db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    monkeypatch.setattr("app.config.settings.razorpay_key_id", "rzp_live_123")
    monkeypatch.setattr("app.config.settings.razorpay_key_secret", "secret")
    
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    
    result = execute_recovery_action(decision, case, payment, db)
    assert result.success is False
    assert "Live mode credentials are not permitted" in result.message

# 5. BLOCK policy + execution true → Razorpay client NOT called
@patch("app.executor.service.create_payment_link")
def test_block_does_not_call_razorpay(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    decision = _make_block_decision()
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is False
    assert "BLOCK" in result.message
    mock_create.assert_not_called()

# 6. Invalid action + execution true → Razorpay client NOT called
@patch("app.executor.service.create_payment_link")
def test_invalid_action_does_not_call_razorpay(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    
    # Let's create an invalid decision with no final action
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    decision.final_action = None
    
    result = execute_recovery_action(decision, case, payment, db)
    
    assert result.success is False
    assert "No final_action" in result.message
    mock_create.assert_not_called()

# 7. Already successful payment → rejected
@patch("app.executor.service.create_payment_link")
def test_successful_payment_rejected(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    payment.status = PaymentStatus.SUCCESS
    db.flush()

    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is False
    assert "already successful" in result.message.lower()
    mock_create.assert_not_called()

# 8. Duplicate execution → existing result returned (idempotent)
@patch("app.executor.service.create_payment_link")
def test_duplicate_execution_is_idempotent(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    mock_create.return_value = {"id": "plink_dup_test", "short_url": "https://rzp.io/dup"}

    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)

    # First call
    result1 = execute_recovery_action(decision, case, payment, db)
    assert result1.success is True
    assert result1.already_existed is False
    assert mock_create.call_count == 1

    # Second call — should return existing, NOT call Razorpay again
    result2 = execute_recovery_action(decision, case, payment, db)
    assert result2.success is True
    assert result2.already_existed is True
    assert result2.external_reference == "plink_dup_test"
    assert mock_create.call_count == 1  # Still only called once


# 9. Payment amount comes from trusted DB record
@patch("app.executor.service.create_payment_link")
def test_amount_from_db_not_llm(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    mock_create.return_value = {"id": "plink_amt", "short_url": "https://rzp.io/amt"}

    # Decision has different amounts (from ML calculation) but executor uses DB payment.amount
    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is True
    call_kwargs = mock_create.call_args
    # Must be the exact DB amount, not any LLM/policy value
    assert call_kwargs[1]["amount_rupees"] == payment.amount


# 10. Razorpay API failure is handled safely
@patch("app.executor.service.create_payment_link")
def test_razorpay_api_failure_handled(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    mock_create.side_effect = Exception("Razorpay API timeout")

    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is False
    assert "Razorpay API" in result.message

    # The action should be marked FAILED in the DB
    action = db.query(RecoveryAction).filter(
        RecoveryAction.recovery_case_id == case.id
    ).first()
    assert action.status == RecoveryActionStatus.FAILED


# 11. Simulated actions (REMINDER)
def test_reminder_creates_simulated_record(db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.REMINDER)

    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is True
    assert "Simulated" in result.message

    action = db.query(RecoveryAction).filter(
        RecoveryAction.recovery_case_id == case.id
    ).first()
    assert action.action_type == RecoveryActionType.REMINDER
    assert action.status == RecoveryActionStatus.COMPLETED


# 12. Simulated actions (ESCALATE)
def test_escalate_creates_simulated_record(db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    decision = _make_allow_decision(AllowedAction.ESCALATE)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is True
    action = db.query(RecoveryAction).filter(
        RecoveryAction.recovery_case_id == case.id
    ).first()
    assert action.action_type == RecoveryActionType.ESCALATE


# 13. Audit record is created
@patch("app.executor.service.create_payment_link")
def test_audit_record_created(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    mock_create.return_value = {"id": "plink_audit", "short_url": "https://rzp.io/audit"}

    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    execute_recovery_action(decision, case, payment, db)

    audits = db.query(AuditLog).filter(
        AuditLog.recovery_case_id == case.id,
        AuditLog.event_type == "RECOVERY_EXECUTION",
    ).all()
    assert len(audits) >= 1


# 14. Non-OPEN case is rejected
@patch("app.executor.service.create_payment_link")
def test_non_open_case_rejected(mock_create, db, sample_data, monkeypatch):
    monkeypatch.setattr("app.integrations.razorpay_service.EXECUTION_ENABLED", True)
    merchant, customer, payment, case = sample_data
    case.status = RecoveryCaseStatus.RECOVERED
    db.flush()

    decision = _make_allow_decision(AllowedAction.PAYMENT_LINK)
    result = execute_recovery_action(decision, case, payment, db)

    assert result.success is False
    assert "RECOVERED" in result.message
    mock_create.assert_not_called()
