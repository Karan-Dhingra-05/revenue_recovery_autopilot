import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum as SAEnum, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import (
    OutcomeType,
    RecoveryActionStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
)

if TYPE_CHECKING:
    from app.models.audit import AuditLog
    from app.models.payment import Payment


class RecoveryCase(Base):
    """
    Represents one active attempt to recover revenue from a failed payment.

    There is at most one RecoveryCase per Payment (enforced by UNIQUE on
    payment_id). 'amount_at_risk' mirrors the payment amount at case creation
    time and is used as the denominator for all recovery metrics.

    'eligibility_score' is populated by the ML model (Phase 3) and indicates
    how worthwhile pursuing this case is (0.0–1.0).
    """

    __tablename__ = "recovery_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    payment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("payments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # one case per payment
    )
    status: Mapped[RecoveryCaseStatus] = mapped_column(
        SAEnum(RecoveryCaseStatus, name="recovery_case_status"),
        nullable=False,
        default=RecoveryCaseStatus.OPEN,
    )
    # Full failed payment amount — the revenue at risk
    amount_at_risk: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    # 0.0000–1.0000 score set by ML (Phase 3); null until scored
    eligibility_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    payment: Mapped["Payment"] = relationship(  # noqa: F821
        "Payment", back_populates="recovery_case"
    )
    actions: Mapped[list["RecoveryAction"]] = relationship(
        "RecoveryAction",
        back_populates="case",
        cascade="all, delete-orphan",
    )
    outcome: Mapped[Optional["RecoveryOutcome"]] = relationship(
        "RecoveryOutcome", back_populates="case", uselist=False
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(  # noqa: F821
        "AuditLog", back_populates="case"
    )

    __table_args__ = (
        Index("ix_recovery_cases_payment_id", "payment_id"),
        Index("ix_recovery_cases_status", "status"),
        Index("ix_recovery_cases_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<RecoveryCase id={self.id} "
            f"status={self.status} "
            f"amount_at_risk={self.amount_at_risk}>"
        )


class RecoveryAction(Base):
    """
    A single intervention proposed (and possibly executed) for a RecoveryCase.

    Multiple RecoveryActions can exist per case — one per attempt or
    per action type. The ML model populates 'probability', 'expected_recovery',
    and 'expected_net_recovery' (Phase 3). The LLM populates 'reason'
    (Phase 4). 'external_reference' holds the Razorpay Payment Link ID
    for PAYMENT_LINK actions (Phase 2).
    """

    __tablename__ = "recovery_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recovery_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recovery_cases.id", ondelete="CASCADE"),
        nullable=False,
    )
    action_type: Mapped[RecoveryActionType] = mapped_column(
        SAEnum(RecoveryActionType, name="recovery_action_type"), nullable=False
    )
    status: Mapped[RecoveryActionStatus] = mapped_column(
        SAEnum(RecoveryActionStatus, name="recovery_action_status"),
        nullable=False,
        default=RecoveryActionStatus.PROPOSED,
    )
    # ML-estimated probability that this action leads to recovery
    probability: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True
    )
    # probability × amount_at_risk
    expected_recovery: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2), nullable=True
    )
    # expected_recovery minus simulated action cost and risk penalty
    expected_net_recovery: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 2), nullable=True
    )
    # Lower integer = higher priority (1 is best)
    priority: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Human-readable rationale from LLM or system (Phase 4)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # e.g. Razorpay Payment Link ID for PAYMENT_LINK actions
    external_reference: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )
    # Razorpay Payment Link short URL (customer-facing)
    payment_link_url: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True
    )
    requested_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    case: Mapped["RecoveryCase"] = relationship(
        "RecoveryCase", back_populates="actions"
    )
    outcome: Mapped[Optional["RecoveryOutcome"]] = relationship(
        "RecoveryOutcome", back_populates="action", uselist=False
    )

    __table_args__ = (
        Index("ix_recovery_actions_case_id", "recovery_case_id"),
        Index("ix_recovery_actions_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<RecoveryAction id={self.id} "
            f"type={self.action_type} "
            f"status={self.status}>"
        )


class RecoveryOutcome(Base):
    """
    The confirmed final outcome of a recovery attempt.

    Written after a Razorpay webhook (Phase 2) or an explicit system event
    confirms the result. There is at most one outcome per RecoveryCase.
    'recovered_amount' is zero for non-RECOVERED outcomes.

    This is the source of truth for 'actual recovered revenue' metrics.
    Do NOT confuse this with 'expected_recovery' (which is a forecast).
    """

    __tablename__ = "recovery_outcomes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recovery_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recovery_cases.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # one outcome per case
    )
    action_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recovery_actions.id", ondelete="SET NULL"),
        nullable=True,
    )
    outcome_type: Mapped[OutcomeType] = mapped_column(
        SAEnum(OutcomeType, name="outcome_type"), nullable=False
    )
    # Actual money recovered. Zero if the case was not RECOVERED.
    recovered_amount: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False, default=Decimal("0.00")
    )
    # Razorpay event/webhook ID used to set this outcome (Phase 2)
    razorpay_event_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, unique=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )

    # Relationships
    case: Mapped["RecoveryCase"] = relationship(
        "RecoveryCase", back_populates="outcome"
    )
    action: Mapped[Optional["RecoveryAction"]] = relationship(
        "RecoveryAction", back_populates="outcome"
    )

    __table_args__ = (
        Index("ix_recovery_outcomes_case_id", "recovery_case_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<RecoveryOutcome id={self.id} "
            f"type={self.outcome_type} "
            f"recovered={self.recovered_amount}>"
        )
