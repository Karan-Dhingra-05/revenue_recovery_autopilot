"""
Recovery Action Executor.

Accepts ONLY a validated PolicyDecision (never raw LLM output).
Dispatches the final_action to the appropriate handler:
  - PAYMENT_LINK  → real Razorpay Test Mode API call
  - REMINDER      → internal simulated record
  - DELAYED_RETRY → internal simulated record
  - ESCALATE      → internal simulated record

Safety invariants:
  - Rejects BLOCK decisions.
  - Rejects already-successful payments.
  - Idempotent: duplicate calls return the existing execution result.
  - Payment amount comes from the trusted DB record, never from the LLM.
  - No Celery — synchronous execution in Phase 6.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.orm import Session

from app.agent.schemas import AllowedAction
from app.integrations.razorpay_service import create_payment_link
from app.models.enums import (
    AuditActorType,
    PaymentStatus,
    RecoveryActionStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.models.audit import AuditLog
from app.models.payment import Payment
from app.models.recovery import RecoveryAction, RecoveryCase
from app.policy.schemas import DecisionType, PolicyDecision

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Simple container for the result of an execution attempt."""

    def __init__(
        self,
        success: bool,
        action_id: uuid.UUID | None = None,
        external_reference: str | None = None,
        payment_link_url: str | None = None,
        message: str = "",
        already_existed: bool = False,
    ):
        self.success = success
        self.action_id = action_id
        self.external_reference = external_reference
        self.payment_link_url = payment_link_url
        self.message = message
        self.already_existed = already_existed

    def __repr__(self) -> str:
        return (
            f"<ExecutionResult success={self.success} "
            f"action_id={self.action_id} "
            f"ref={self.external_reference}>"
        )


# ── Pre-flight checks ───────────────────────────────────────────────────────

def _validate_execution(
    decision: PolicyDecision,
    case: RecoveryCase,
    payment: Payment,
) -> str | None:
    """
    Run pre-flight safety checks before executing.

    Returns None if safe, or an error message string if execution must be refused.
    """
    # 1. Decision must be ALLOW
    if decision.decision != DecisionType.ALLOW:
        return f"Decision is {decision.decision.value}, not ALLOW. Execution refused."

    # 2. Must have a final_action
    if decision.final_action is None:
        return "No final_action in the policy decision."

    # 3. Payment must not already be successful
    if payment.status == PaymentStatus.SUCCESS:
        return "Payment is already successful. No recovery action needed."

    # 4. Recovery case must be OPEN
    if case.status != RecoveryCaseStatus.OPEN:
        return f"Recovery case is {case.status.value}, not OPEN."

    return None


# ── Idempotency check ───────────────────────────────────────────────────────

def _find_existing_execution(
    case: RecoveryCase,
    action_type: RecoveryActionType,
    db: Session,
) -> RecoveryAction | None:
    """
    Check if an action of this type has already been executed for this case.

    Returns the existing RecoveryAction if it has been executed or is executing,
    otherwise None.
    """
    return (
        db.query(RecoveryAction)
        .filter(
            RecoveryAction.recovery_case_id == case.id,
            RecoveryAction.action_type == action_type,
            RecoveryAction.status.in_([
                RecoveryActionStatus.EXECUTING,
                RecoveryActionStatus.COMPLETED,
                RecoveryActionStatus.APPROVED,
            ]),
        )
        .first()
    )


# ── Action handlers ─────────────────────────────────────────────────────────

def _execute_payment_link(
    case: RecoveryCase,
    payment: Payment,
    decision: PolicyDecision,
    db: Session,
) -> ExecutionResult:
    """
    Create a real Razorpay TEST MODE Payment Link.

    Uses the TRUSTED payment amount from the DB (Decimal), never from LLM.
    """
    # Idempotency: check for existing execution
    existing = _find_existing_execution(case, RecoveryActionType.PAYMENT_LINK, db)
    if existing and existing.external_reference:
        logger.info(
            "Payment Link already exists for case %s: %s",
            case.id, existing.external_reference,
        )
        return ExecutionResult(
            success=True,
            action_id=existing.id,
            external_reference=existing.external_reference,
            payment_link_url=existing.payment_link_url,
            message="Payment Link already created (idempotent).",
            already_existed=True,
        )

    # Generate unique reference (Razorpay limit: 40 chars)
    # rra_ (4) + case_hex[:26] (26) + _ (1) + uuid[:8] (8) = 39 chars
    reference_id = f"rra_{case.id.hex[:26]}_{uuid.uuid4().hex[:8]}"

    # Trusted amount from DB (Decimal, not float)
    amount_rupees: Decimal = payment.amount
    currency: str = payment.currency

    # Customer info from relationship
    customer = payment.customer
    customer_name = None
    customer_email = None
    if customer:
        customer_email = customer.email
        customer_name = customer.external_customer_id

    from app.integrations.razorpay_service import EXECUTION_ENABLED
    if not EXECUTION_ENABLED:
        logger.warning(
            "Razorpay execution is disabled (RAZORPAY_EXECUTION_ENABLED=false). "
            "Action blocked."
        )
        return ExecutionResult(
            success=False,
            message="Razorpay execution is disabled. Cannot create Payment Link.",
        )

    # Create the RecoveryAction record first
    action = RecoveryAction(
        recovery_case_id=case.id,
        action_type=RecoveryActionType.PAYMENT_LINK,
        status=RecoveryActionStatus.EXECUTING,
        probability=None,
        expected_recovery=decision.expected_recovery.get(AllowedAction.PAYMENT_LINK),
        expected_net_recovery=decision.expected_net_recovery.get(AllowedAction.PAYMENT_LINK),
        reason=decision.reason,
    )
    db.add(action)
    db.flush()  # Get the action ID

    try:
        rz_result = create_payment_link(
            amount_rupees=amount_rupees,
            currency=currency,
            reference_id=reference_id,
            description=f"Recovery payment for order (case {str(case.id)[:8]})",
            customer_name=customer_name,
            customer_email=customer_email,
        )

        plink_id = rz_result.get("id", "")
        short_url = rz_result.get("short_url", "")

        action.external_reference = plink_id
        action.payment_link_url = short_url
        action.executed_at = datetime.now(timezone.utc)
        db.flush()

        logger.info(
            "Payment Link created: case=%s plink=%s url=%s",
            case.id, plink_id, short_url,
        )

        return ExecutionResult(
            success=True,
            action_id=action.id,
            external_reference=plink_id,
            payment_link_url=short_url,
            message="Payment Link created successfully.",
        )

    except Exception as e:
        action.status = RecoveryActionStatus.FAILED
        action.reason = f"Razorpay API error: {e}"
        db.flush()
        logger.error("Razorpay Payment Link creation failed: %s", e)
        return ExecutionResult(
            success=False,
            action_id=action.id,
            message=f"Razorpay API error: {e}",
        )


def _execute_simulated(
    case: RecoveryCase,
    action_type: RecoveryActionType,
    decision: PolicyDecision,
    db: Session,
) -> ExecutionResult:
    """
    Create an internal simulated execution record for non-Razorpay actions.

    Used for REMINDER, DELAYED_RETRY, and ESCALATE.
    These are clearly marked as simulated/demo and never call an external API.
    """
    existing = _find_existing_execution(case, action_type, db)
    if existing:
        return ExecutionResult(
            success=True,
            action_id=existing.id,
            message=f"Simulated {action_type.value} already recorded (idempotent).",
            already_existed=True,
        )

    allowed_action = AllowedAction(action_type.value)

    action = RecoveryAction(
        recovery_case_id=case.id,
        action_type=action_type,
        status=RecoveryActionStatus.COMPLETED,
        probability=None,
        expected_recovery=decision.expected_recovery.get(allowed_action),
        expected_net_recovery=decision.expected_net_recovery.get(allowed_action),
        reason=f"[SIMULATED] {decision.reason}",
        executed_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(action)
    db.flush()

    logger.info("Simulated %s execution for case %s", action_type.value, case.id)

    return ExecutionResult(
        success=True,
        action_id=action.id,
        message=f"Simulated {action_type.value} recorded.",
    )


# ── Mapping ──────────────────────────────────────────────────────────────────

_ACTION_MAP: dict[AllowedAction, RecoveryActionType] = {
    AllowedAction.PAYMENT_LINK: RecoveryActionType.PAYMENT_LINK,
    AllowedAction.REMINDER: RecoveryActionType.REMINDER,
    AllowedAction.DELAYED_RETRY: RecoveryActionType.DELAYED_RETRY,
    AllowedAction.ESCALATE: RecoveryActionType.ESCALATE,
}


# ── Main entry point ────────────────────────────────────────────────────────

def execute_recovery_action(
    decision: PolicyDecision,
    case: RecoveryCase,
    payment: Payment,
    db: Session,
) -> ExecutionResult:
    """
    Execute the recovery action determined by the Policy Engine.

    This is the ONLY entry point for executing recovery actions.
    It accepts a validated PolicyDecision — never raw Gemini output.

    Flow:
      1. Validate pre-flight safety checks.
      2. Dispatch to the correct handler.
      3. Write an audit log entry.
      4. Commit is the caller's responsibility.
    """
    # Pre-flight
    error = _validate_execution(decision, case, payment)
    if error:
        logger.warning("Execution refused for case %s: %s", case.id, error)
        _write_audit(case.id, decision, error, db)
        return ExecutionResult(success=False, message=error)

    final_action = decision.final_action
    action_type = _ACTION_MAP.get(final_action)
    if action_type is None:
        msg = f"Unsupported action: {final_action}"
        logger.error(msg)
        _write_audit(case.id, decision, msg, db)
        return ExecutionResult(success=False, message=msg)

    # Dispatch
    if final_action == AllowedAction.PAYMENT_LINK:
        result = _execute_payment_link(case, payment, decision, db)
    else:
        result = _execute_simulated(case, action_type, decision, db)

    # Audit
    _write_audit(
        case.id,
        decision,
        result.message,
        db,
        action_id=result.action_id,
        external_ref=result.external_reference,
    )

    return result


def _write_audit(
    case_id: uuid.UUID,
    decision: PolicyDecision,
    message: str,
    db: Session,
    action_id: uuid.UUID | None = None,
    external_ref: str | None = None,
) -> None:
    """Write an audit log entry for the execution attempt."""
    audit = AuditLog(
        recovery_case_id=case_id,
        actor_type=AuditActorType.SYSTEM,
        actor_name="recovery_executor",
        event_type="RECOVERY_EXECUTION",
        decision_summary=message,
        policy_result=decision.decision.value,
        metadata_json={
            "decision_source": decision.decision_source.value,
            "proposed_action": decision.proposed_action.value,
            "final_action": decision.final_action.value if decision.final_action else None,
            "action_id": str(action_id) if action_id else None,
            "external_reference": external_ref,
        },
    )
    db.add(audit)
    db.flush()
