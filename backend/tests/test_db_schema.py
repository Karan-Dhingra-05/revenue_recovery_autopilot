"""
Integration tests for the PostgreSQL schema created by the initial Alembic migration.

These tests require a running PostgreSQL instance (docker compose up).
They are marked with @pytest.mark.integration and use the db_session fixture
from conftest.py, which rolls back all writes after each test.

What is tested:
  1. All expected tables exist.
  2. Monetary amounts use Numeric — no floating-point drift on round-trip.
  3. Foreign key constraints are enforced by the database.
  4. Full recovery chain can be created (Merchant→Customer→Payment→...→RecoveryAction).
  5. UNIQUE constraint prevents two RecoveryCases for the same Payment.
  6. UNIQUE constraint on PaymentFailure (one per Payment).
  7. AuditLog can be created standalone (nullable recovery_case_id).
  8. RecoveryOutcome stores recovered_amount as Numeric (no drift).
"""

import uuid
from decimal import Decimal
from typing import Generator

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.audit import AuditLog
from app.models.customer import Customer
from app.models.enums import (
    AuditActorType,
    OutcomeType,
    PaymentMethod,
    PaymentStatus,
    RecoveryActionStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.models.merchant import Merchant
from app.models.payment import Payment, PaymentFailure
from app.models.recovery import RecoveryAction, RecoveryCase, RecoveryOutcome

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_merchant(db: Session, name: str = "Test Merchant") -> Merchant:
    m = Merchant(name=name, currency="INR")
    db.add(m)
    db.flush()
    return m


def _make_customer(db: Session, merchant: Merchant, ext_id: str = "cust_x") -> Customer:
    c = Customer(
        merchant_id=merchant.id,
        external_customer_id=ext_id,
        successful_payment_count=2,
        failed_payment_count=1,
        total_paid=Decimal("10000.00"),
    )
    db.add(c)
    db.flush()
    return c


def _make_failed_payment(
    db: Session,
    merchant: Merchant,
    customer: Customer,
    amount: Decimal = Decimal("5000.00"),
) -> Payment:
    p = Payment(
        merchant_id=merchant.id,
        customer_id=customer.id,
        amount=amount,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        status=PaymentStatus.FAILED,
    )
    db.add(p)
    db.flush()
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_expected_tables_exist(db_engine) -> None:
    """
    Verify that all 8 domain tables (+ alembic_version) are present.
    This catches a missing model import in models/__init__.py.
    """
    inspector = inspect(db_engine)
    actual_tables = set(inspector.get_table_names())
    required = {
        "merchants",
        "customers",
        "payments",
        "payment_failures",
        "recovery_cases",
        "recovery_actions",
        "recovery_outcomes",
        "audit_logs",
    }
    missing = required - actual_tables
    assert not missing, f"Tables missing from database: {missing}"


def test_monetary_amount_no_float_drift(db_session: Session) -> None:
    """
    Numeric(14, 2) amounts must survive a DB round-trip without drift.
    A float column would store 9999.99 as 9999.990234375 or similar.
    """
    merchant = _make_merchant(db_session)
    customer = _make_customer(db_session, merchant)

    tricky_amount = Decimal("9999.99")
    payment = _make_failed_payment(db_session, merchant, customer, tricky_amount)
    db_session.refresh(payment)

    assert payment.amount == tricky_amount, (
        f"Amount drifted: stored {payment.amount!r}, expected {tricky_amount!r}"
    )


def test_customer_fk_enforced(db_session: Session) -> None:
    """
    Inserting a Customer with a non-existent merchant_id must raise IntegrityError.
    """
    orphan = Customer(
        merchant_id=uuid.uuid4(),  # does not exist
        external_customer_id="orphan",
        successful_payment_count=0,
        failed_payment_count=0,
        total_paid=Decimal("0.00"),
    )
    db_session.add(orphan)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_payment_fk_enforced(db_session: Session) -> None:
    """
    Inserting a Payment with a non-existent customer_id must raise IntegrityError.
    """
    merchant = _make_merchant(db_session)

    bad_payment = Payment(
        merchant_id=merchant.id,
        customer_id=uuid.uuid4(),  # does not exist
        amount=Decimal("100.00"),
        currency="INR",
        payment_method=PaymentMethod.UPI,
        status=PaymentStatus.FAILED,
    )
    db_session.add(bad_payment)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_full_recovery_chain_can_be_created(db_session: Session) -> None:
    """
    Verify the complete chain:
    Merchant → Customer → Payment → PaymentFailure → RecoveryCase → RecoveryAction

    Also verifies relationships are navigable.
    """
    merchant = _make_merchant(db_session, "Chain Merchant")
    customer = _make_customer(db_session, merchant, "cust_chain")
    payment = _make_failed_payment(db_session, merchant, customer, Decimal("10000.00"))

    failure = PaymentFailure(
        payment_id=payment.id,
        failure_code="INSUFFICIENT_FUNDS",
        failure_reason="Balance was too low.",
        failure_source="bank",
    )
    db_session.add(failure)
    db_session.flush()

    case = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.OPEN,
        amount_at_risk=payment.amount,
    )
    db_session.add(case)
    db_session.flush()

    action = RecoveryAction(
        recovery_case_id=case.id,
        action_type=RecoveryActionType.PAYMENT_LINK,
        status=RecoveryActionStatus.PROPOSED,
        probability=Decimal("0.7800"),
        expected_recovery=Decimal("7800.00"),
        expected_net_recovery=Decimal("7700.00"),
        priority=1,
        reason="High-value customer with good history.",
    )
    db_session.add(action)
    db_session.flush()

    # Verify navigation through relationships
    db_session.refresh(case)
    assert len(case.actions) == 1
    assert case.actions[0].action_type == RecoveryActionType.PAYMENT_LINK
    assert case.actions[0].probability == Decimal("0.7800")
    assert case.amount_at_risk == Decimal("10000.00")

    # Verify payment → case navigation
    db_session.refresh(payment)
    assert payment.recovery_case is not None
    assert payment.recovery_case.id == case.id


def test_unique_recovery_case_per_payment(db_session: Session) -> None:
    """
    A Payment can have at most one RecoveryCase (UNIQUE constraint on payment_id).
    """
    merchant = _make_merchant(db_session, "Unique Case Merchant")
    customer = _make_customer(db_session, merchant, "cust_unique_case")
    payment = _make_failed_payment(db_session, merchant, customer)

    case1 = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.OPEN,
        amount_at_risk=payment.amount,
    )
    db_session.add(case1)
    db_session.flush()

    # Duplicate case for the same payment → must fail
    case2 = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.OPEN,
        amount_at_risk=payment.amount,
    )
    db_session.add(case2)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_unique_failure_per_payment(db_session: Session) -> None:
    """
    A Payment can have at most one PaymentFailure (UNIQUE on payment_id).
    """
    merchant = _make_merchant(db_session, "Unique Failure Merchant")
    customer = _make_customer(db_session, merchant, "cust_unique_fail")
    payment = _make_failed_payment(db_session, merchant, customer)

    fail1 = PaymentFailure(
        payment_id=payment.id,
        failure_code="TIMEOUT",
        failure_source="gateway",
    )
    db_session.add(fail1)
    db_session.flush()

    fail2 = PaymentFailure(
        payment_id=payment.id,  # duplicate
        failure_code="ANOTHER_CODE",
        failure_source="bank",
    )
    db_session.add(fail2)

    with pytest.raises(IntegrityError):
        db_session.flush()

    db_session.rollback()


def test_audit_log_without_case(db_session: Session) -> None:
    """
    AuditLog with a null recovery_case_id is valid for system-wide events.
    """
    log = AuditLog(
        recovery_case_id=None,
        actor_type=AuditActorType.SYSTEM,
        actor_name="test_runner",
        event_type="TEST_STARTUP",
        decision_summary="System test event with no associated case.",
        metadata_json={"env": "test", "version": "0.1.0"},
    )
    db_session.add(log)
    db_session.flush()

    assert log.id is not None
    assert log.timestamp is not None


def test_recovery_outcome_amount_no_drift(db_session: Session) -> None:
    """
    RecoveryOutcome.recovered_amount uses Numeric(14, 2) — no float drift.
    """
    merchant = _make_merchant(db_session, "Outcome Merchant")
    customer = _make_customer(db_session, merchant, "cust_outcome")
    payment = _make_failed_payment(db_session, merchant, customer, Decimal("3333.33"))

    case = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.RECOVERED,
        amount_at_risk=payment.amount,
    )
    db_session.add(case)
    db_session.flush()

    recovered = Decimal("3333.33")
    outcome = RecoveryOutcome(
        recovery_case_id=case.id,
        outcome_type=OutcomeType.RECOVERED,
        recovered_amount=recovered,
    )
    db_session.add(outcome)
    db_session.flush()
    db_session.refresh(outcome)

    assert outcome.recovered_amount == recovered, (
        f"recovered_amount drifted: stored {outcome.recovered_amount!r}, "
        f"expected {recovered!r}"
    )
