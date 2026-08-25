import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum as SAEnum, ForeignKey, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import PaymentMethod, PaymentStatus

if TYPE_CHECKING:
    from app.models.customer import Customer
    from app.models.merchant import Merchant
    from app.models.recovery import RecoveryCase


class Payment(Base):
    """
    A single payment transaction belonging to a merchant and customer.

    The 'amount' column uses Numeric(14, 2) — never Float — to prevent
    floating-point rounding errors in financial calculations.

    'razorpay_payment_id' is nullable until the Razorpay integration
    is wired up in Phase 2.
    """

    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("merchants.id", ondelete="CASCADE"),
        nullable=False,
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
    )
    # External Razorpay identifier — populated in Phase 2
    razorpay_payment_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True
    )
    # Monetary amount in the merchant's currency (e.g. ₹10,000.00).
    # Numeric prevents the float drift that would corrupt financial aggregates.
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    payment_method: Mapped[PaymentMethod] = mapped_column(
        SAEnum(PaymentMethod, name="payment_method"), nullable=False
    )
    status: Mapped[PaymentStatus] = mapped_column(
        SAEnum(PaymentStatus, name="payment_status"),
        nullable=False,
        default=PaymentStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    merchant: Mapped["Merchant"] = relationship(  # noqa: F821
        "Merchant", back_populates="payments"
    )
    customer: Mapped["Customer"] = relationship(  # noqa: F821
        "Customer", back_populates="payments"
    )
    failure: Mapped[Optional["PaymentFailure"]] = relationship(
        "PaymentFailure", back_populates="payment", uselist=False
    )
    recovery_case: Mapped[Optional["RecoveryCase"]] = relationship(  # noqa: F821
        "RecoveryCase", back_populates="payment", uselist=False
    )

    __table_args__ = (
        Index("ix_payments_merchant_id", "merchant_id"),
        Index("ix_payments_customer_id", "customer_id"),
        Index("ix_payments_status", "status"),
        Index("ix_payments_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Payment id={self.id} "
            f"amount={self.amount} "
            f"status={self.status}>"
        )


class PaymentFailure(Base):
    """
    Failure details for a single failed Payment.

    One-to-one with Payment (enforced by UNIQUE on payment_id).
    'failure_code' and 'failure_source' are used as ML features (Phase 3).
    """

    __tablename__ = "payment_failures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    payment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("payments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # one failure record per payment
    )
    failure_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Origin of the failure: 'bank', 'gateway', 'upi', 'customer', etc.
    failure_source: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    # Razorpay event / webhook ID for traceability (Phase 2)
    raw_event_reference: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True
    )

    # Relationships
    payment: Mapped["Payment"] = relationship("Payment", back_populates="failure")

    __table_args__ = (
        Index("ix_payment_failures_payment_id", "payment_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<PaymentFailure payment_id={self.payment_id} "
            f"code={self.failure_code!r}>"
        )
