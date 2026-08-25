import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base

if TYPE_CHECKING:
    from app.models.merchant import Merchant
    from app.models.payment import Payment

class Customer(Base):
    """
    A customer belonging to one merchant.

    The aggregate fields (successful_payment_count, etc.) are denormalised
    and updated when payments are ingested. They serve as quick features
    for the ML model (Phase 3) without requiring a GROUP-BY at inference time.
    """

    __tablename__ = "customers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    merchant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("merchants.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Opaque identifier from the merchant's own system
    external_customer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # Demo or hashed email — never store a real unhashed email in production
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Denormalised aggregate stats for fast ML feature extraction
    successful_payment_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    failed_payment_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_paid: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False, default=Decimal("0.00")
    )
    last_success_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    merchant: Mapped["Merchant"] = relationship(  # noqa: F821
        "Merchant", back_populates="customers"
    )
    payments: Mapped[list["Payment"]] = relationship(  # noqa: F821
        "Payment", back_populates="customer"
    )

    __table_args__ = (
        # Fast lookup of all customers for a given merchant
        Index("ix_customers_merchant_id", "merchant_id"),
        # Enforce unique external_customer_id within a merchant's scope
        Index(
            "ix_customers_merchant_external",
            "merchant_id",
            "external_customer_id",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Customer id={self.id} "
            f"external_id={self.external_customer_id!r} "
            f"merchant={self.merchant_id}>"
        )
