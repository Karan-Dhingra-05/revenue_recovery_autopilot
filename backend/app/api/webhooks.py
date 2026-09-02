"""
Razorpay webhook endpoint.

Handles incoming Razorpay webhook events with:
  1. Raw body signature verification (HMAC SHA-256).
  2. Event ID deduplication.
  3. payment_link.paid processing → RecoveryOutcome.

Security:
  - Signature is verified BEFORE any JSON parsing.
  - Event ID prevents duplicate processing.
  - Recovered amount is verified against trusted DB data.
  - No secrets are logged.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.integrations.razorpay_service import verify_webhook_signature
from app.models.enums import (
    AuditActorType,
    OutcomeType,
    RecoveryActionStatus,
    RecoveryCaseStatus,
)
from app.models.audit import AuditLog
from app.models.recovery import RecoveryAction, RecoveryCase, RecoveryOutcome

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhooks"])


@router.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, db: Session = Depends(get_db)) -> Response:
    """
    Receive and process Razorpay webhook events.

    Flow:
      1. Read raw body (bytes) — do NOT parse first.
      2. Validate X-Razorpay-Signature.
      3. Read x-razorpay-event-id for idempotency.
      4. Parse JSON only after signature validation.
      5. Dispatch to event handler.
    """
    # ── Step 1: Read raw body ──
    raw_body = await request.body()

    # ── Step 2: Verify signature ──
    signature = request.headers.get("X-Razorpay-Signature", "")
    if not signature:
        logger.warning("Webhook received without X-Razorpay-Signature header")
        return Response(
            content=json.dumps({"error": "Missing signature"}),
            status_code=400,
            media_type="application/json",
        )

    if not verify_webhook_signature(raw_body, signature):
        logger.warning("Webhook signature verification failed")
        return Response(
            content=json.dumps({"error": "Invalid signature"}),
            status_code=400,
            media_type="application/json",
        )

    # ── Step 3: Read event ID ──
    event_id = request.headers.get("x-razorpay-event-id", "")

    # ── Step 4: Parse JSON after signature validation ──
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.error("Webhook body is not valid JSON")
        return Response(
            content=json.dumps({"error": "Invalid JSON body"}),
            status_code=400,
            media_type="application/json",
        )

    event_name = payload.get("event", "")
    logger.info("Webhook received: event=%s event_id=%s", event_name, event_id)

    # ── Step 5: Dispatch ──
    if event_name == "payment_link.paid":
        return _handle_payment_link_paid(payload, event_id, db)

    # Unknown event — acknowledge but don't process
    logger.info("Ignoring unhandled webhook event: %s", event_name)
    return Response(
        content=json.dumps({"status": "ignored", "event": event_name}),
        status_code=200,
        media_type="application/json",
    )


def _handle_payment_link_paid(payload: dict, event_id: str, db: Session) -> Response:
    """
    Process a payment_link.paid webhook event.

    Steps:
      1. Extract Payment Link ID from payload.
      2. Find the corresponding RecoveryAction by external_reference.
      3. Deduplicate using event_id (DB unique constraint on razorpay_event_id).
      4. Verify paid amount against trusted DB amount.
      5. Create RecoveryOutcome and update RecoveryCase status.
    """
    try:
        # Extract Payment Link entity
        payment_link_entity = (
            payload.get("payload", {})
            .get("payment_link", {})
            .get("entity", {})
        )
        plink_id = payment_link_entity.get("id", "")

        # Extract payment entity for amount verification
        payment_entity = (
            payload.get("payload", {})
            .get("payment", {})
            .get("entity", {})
        )
        paid_amount_paise = payment_entity.get("amount", 0)

        if not plink_id:
            logger.warning("payment_link.paid webhook missing Payment Link ID")
            return Response(
                content=json.dumps({"error": "Missing payment_link.entity.id"}),
                status_code=400,
                media_type="application/json",
            )

        # ── Find RecoveryAction ──
        action = (
            db.query(RecoveryAction)
            .filter(RecoveryAction.external_reference == plink_id)
            .first()
        )
        if not action:
            logger.warning("No RecoveryAction found for Payment Link %s", plink_id)
            return Response(
                content=json.dumps({"error": "Unknown Payment Link", "plink_id": plink_id}),
                status_code=404,
                media_type="application/json",
            )

        case = action.case

        # ── Deduplicate using event_id ──
        if event_id:
            existing_outcome = (
                db.query(RecoveryOutcome)
                .filter(RecoveryOutcome.razorpay_event_id == event_id)
                .first()
            )
            if existing_outcome:
                logger.info("Duplicate webhook event %s — already processed", event_id)
                return Response(
                    content=json.dumps({"status": "already_processed", "event_id": event_id}),
                    status_code=200,
                    media_type="application/json",
                )

        # ── Check if case already recovered ──
        if case.status == RecoveryCaseStatus.RECOVERED:
            logger.info("Case %s already RECOVERED — ignoring duplicate", case.id)
            return Response(
                content=json.dumps({"status": "already_recovered", "case_id": str(case.id)}),
                status_code=200,
                media_type="application/json",
            )

        # ── Verify amount ──
        # Trusted amount: from our DB (Decimal rupees → paise via Decimal math)
        trusted_amount_paise = int(
            (case.amount_at_risk * Decimal("100")).to_integral_value(rounding=ROUND_HALF_UP)
        )

        amount_matches = paid_amount_paise >= trusted_amount_paise
        if not amount_matches:
            logger.warning(
                "Paid amount %d paise < expected %d paise for case %s. "
                "Partial/unexpected payment — not marking as fully recovered.",
                paid_amount_paise, trusted_amount_paise, case.id,
            )
            # Still record the outcome but as FAILED (partial)
            recovered_rupees = Decimal(paid_amount_paise) / Decimal("100")

            outcome = RecoveryOutcome(
                recovery_case_id=case.id,
                action_id=action.id,
                outcome_type=OutcomeType.FAILED,
                recovered_amount=recovered_rupees,
                razorpay_event_id=event_id if event_id else None,
            )
            db.add(outcome)

            _write_webhook_audit(
                db, case.id, event_id, plink_id,
                f"Partial payment: {paid_amount_paise} paise < expected {trusted_amount_paise} paise",
            )
            db.commit()

            return Response(
                content=json.dumps({
                    "status": "partial_payment",
                    "paid_paise": paid_amount_paise,
                    "expected_paise": trusted_amount_paise,
                }),
                status_code=200,
                media_type="application/json",
            )

        # ── Full payment: Mark RECOVERED ──
        recovered_rupees = case.amount_at_risk  # Trust DB amount, not webhook

        outcome = RecoveryOutcome(
            recovery_case_id=case.id,
            action_id=action.id,
            outcome_type=OutcomeType.RECOVERED,
            recovered_amount=recovered_rupees,
            razorpay_event_id=event_id if event_id else None,
        )
        db.add(outcome)

        action.status = RecoveryActionStatus.COMPLETED
        action.completed_at = datetime.now(timezone.utc)
        case.status = RecoveryCaseStatus.RECOVERED
        case.closed_at = datetime.now(timezone.utc)

        _write_webhook_audit(
            db, case.id, event_id, plink_id,
            f"Payment recovered: ₹{recovered_rupees}. Full amount verified.",
        )

        db.commit()

        logger.info(
            "Recovery successful: case=%s amount=₹%s plink=%s",
            case.id, recovered_rupees, plink_id,
        )

        return Response(
            content=json.dumps({
                "status": "recovered",
                "case_id": str(case.id),
                "recovered_amount": str(recovered_rupees),
            }),
            status_code=200,
            media_type="application/json",
        )

    except Exception as e:
        db.rollback()
        logger.error("Webhook processing error: %s", e, exc_info=True)
        return Response(
            content=json.dumps({"error": "Internal processing error"}),
            status_code=500,
            media_type="application/json",
        )


def _write_webhook_audit(
    db: Session,
    case_id,
    event_id: str,
    plink_id: str,
    message: str,
) -> None:
    """Write an audit log entry for webhook processing."""
    audit = AuditLog(
        recovery_case_id=case_id,
        actor_type=AuditActorType.SYSTEM,
        actor_name="razorpay_webhook",
        event_type="WEBHOOK_PROCESSED",
        decision_summary=message,
        metadata_json={
            "razorpay_event_id": event_id,
            "payment_link_id": plink_id,
        },
    )
    db.add(audit)
