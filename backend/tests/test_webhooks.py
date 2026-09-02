"""
Tests for the Razorpay webhook endpoint (Phase 6).

All tests use FastAPI's TestClient with dependency override for the DB session.
No real Razorpay webhooks are sent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base, get_db
from app.main import app
from app.models.enums import (
    PaymentMethod,
    PaymentStatus,
    RecoveryActionStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
    OutcomeType,
)
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.recovery import RecoveryAction, RecoveryCase, RecoveryOutcome


# ── Fixtures ─────────────────────────────────────────────────────────────────

WEBHOOK_SECRET = "test_webhook_secret_123"


@pytest.fixture(scope="module")
def test_engine():
    from app.config import settings
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(test_engine):
    """Provide a DB session and override the FastAPI get_db dependency."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    # Override the get_db dependency so the webhook handler uses our test session
    def override_get_db():
        try:
            yield session
        finally:
            pass  # We manage the session lifecycle ourselves

    app.dependency_overrides[get_db] = override_get_db
    yield session
    app.dependency_overrides.clear()
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client():
    return TestClient(app)


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()


def _make_webhook_payload(plink_id: str, amount_paise: int) -> dict:
    return {
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": plink_id,
                    "amount": amount_paise,
                    "status": "paid",
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_test_xyz",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "captured",
                }
            },
        },
    }


def _create_test_chain(db: Session, amount: Decimal = Decimal("12500.00"), plink_id: str = "plink_wh_test"):
    merchant = Merchant(name="WH Test Merchant", currency="INR")
    db.add(merchant)
    db.flush()

    customer = Customer(
        merchant_id=merchant.id,
        external_customer_id=f"wh_cust_{uuid.uuid4().hex[:6]}",
        email="wh@test.com",
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        merchant_id=merchant.id,
        customer_id=customer.id,
        amount=amount,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        status=PaymentStatus.FAILED,
    )
    db.add(payment)
    db.flush()

    case = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.OPEN,
        amount_at_risk=amount,
    )
    db.add(case)
    db.flush()

    action = RecoveryAction(
        recovery_case_id=case.id,
        action_type=RecoveryActionType.PAYMENT_LINK,
        status=RecoveryActionStatus.EXECUTING,
        external_reference=plink_id,
        payment_link_url="https://rzp.io/test",
        executed_at=datetime.now(timezone.utc),
    )
    db.add(action)
    db.flush()

    return merchant, customer, payment, case, action


def _send_webhook(client, payload: dict, event_id: str = ""):
    body = json.dumps(payload).encode("utf-8")
    sig = _sign(body)
    headers = {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": sig,
    }
    if event_id:
        headers["x-razorpay-event-id"] = event_id

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.integrations.razorpay_service.settings.razorpay_webhook_secret", WEBHOOK_SECRET)
        return client.post("/api/webhooks/razorpay", content=body, headers=headers)


# ── Tests ──────────────────────────────────────────────────────────────────


# 1. Valid payment_link.paid → recovery outcome created
def test_valid_webhook_creates_recovery_outcome(db_session, client):
    plink_id = f"plink_valid_{uuid.uuid4().hex[:6]}"
    _, _, payment, case, action = _create_test_chain(db_session, plink_id=plink_id)

    payload = _make_webhook_payload(plink_id, 1250000)
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    resp = _send_webhook(client, payload, event_id)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recovered"

    db_session.expire_all()
    outcome = db_session.query(RecoveryOutcome).filter(
        RecoveryOutcome.recovery_case_id == case.id
    ).first()
    assert outcome is not None
    assert outcome.outcome_type == OutcomeType.RECOVERED
    assert outcome.recovered_amount == Decimal("12500.00")

    refreshed_case = db_session.get(RecoveryCase, case.id)
    assert refreshed_case.status == RecoveryCaseStatus.RECOVERED


# 2. Invalid signature → 400
def test_invalid_signature_rejected(client):
    payload = _make_webhook_payload("plink_badsig", 1000)
    body = json.dumps(payload).encode("utf-8")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.integrations.razorpay_service.settings.razorpay_webhook_secret", WEBHOOK_SECRET)
        resp = client.post(
            "/api/webhooks/razorpay",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": "definitely_wrong_signature",
            },
        )

    assert resp.status_code == 400
    assert "Invalid signature" in resp.json()["error"]


# 3. Missing signature → 400
def test_missing_signature_rejected(client):
    payload = _make_webhook_payload("plink_nosig", 1000)
    body = json.dumps(payload).encode("utf-8")
    resp = client.post(
        "/api/webhooks/razorpay",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Missing signature" in resp.json()["error"]


# 4. Duplicate event ID → idempotent 200
def test_duplicate_event_id_is_idempotent(db_session, client):
    plink_id = f"plink_dup_{uuid.uuid4().hex[:6]}"
    _, _, _, case, _ = _create_test_chain(db_session, plink_id=plink_id)

    payload = _make_webhook_payload(plink_id, 1250000)
    event_id = f"evt_dup_{uuid.uuid4().hex[:8]}"

    resp1 = _send_webhook(client, payload, event_id)
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "recovered"

    resp2 = _send_webhook(client, payload, event_id)
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "already_processed"

    outcomes = db_session.query(RecoveryOutcome).filter(
        RecoveryOutcome.recovery_case_id == case.id
    ).all()
    assert len(outcomes) == 1


# 5. Unknown Payment Link → 404
def test_unknown_payment_link_returns_404(db_session, client):
    payload = _make_webhook_payload("plink_unknown_xyz", 1000)
    event_id = f"evt_unknown_{uuid.uuid4().hex[:6]}"
    resp = _send_webhook(client, payload, event_id)
    assert resp.status_code == 404
    assert "Unknown Payment Link" in resp.json()["error"]


# 6. Amount mismatch (partial) → not fully recovered
def test_amount_mismatch_not_fully_recovered(db_session, client):
    plink_id = f"plink_partial_{uuid.uuid4().hex[:6]}"
    _, _, _, case, _ = _create_test_chain(
        db_session, amount=Decimal("10000.00"), plink_id=plink_id
    )

    payload = _make_webhook_payload(plink_id, 5000)  # ₹50 instead of ₹10,000
    event_id = f"evt_partial_{uuid.uuid4().hex[:6]}"
    resp = _send_webhook(client, payload, event_id)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "partial_payment"

    db_session.expire_all()
    refreshed_case = db_session.get(RecoveryCase, case.id)
    assert refreshed_case.status != RecoveryCaseStatus.RECOVERED


# 7. Webhook amount does not override DB amount
def test_webhook_amount_does_not_override_db(db_session, client):
    plink_id = f"plink_trust_{uuid.uuid4().hex[:6]}"
    db_amount = Decimal("5000.00")
    _, _, _, case, _ = _create_test_chain(db_session, amount=db_amount, plink_id=plink_id)

    # Webhook claims ₹99,999 paid
    payload = _make_webhook_payload(plink_id, 9999900)
    event_id = f"evt_trust_{uuid.uuid4().hex[:6]}"
    resp = _send_webhook(client, payload, event_id)

    assert resp.status_code == 200
    assert resp.json()["status"] == "recovered"

    db_session.expire_all()
    outcome = db_session.query(RecoveryOutcome).filter(
        RecoveryOutcome.recovery_case_id == case.id
    ).first()
    assert outcome.recovered_amount == db_amount  # ₹5000, not ₹99,999
