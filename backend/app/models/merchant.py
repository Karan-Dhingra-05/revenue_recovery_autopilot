import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base

if TYPE_CHECKING:
    from app.models.customer import Customer
    from app.models.payment import Payment

# ---------------------------------------------------------------------------
# Default policy configuration applied to every new merchant.
# Keys must remain in sync with the policy engine (Phase 4).
# ---------------------------------------------------------------------------
DEFAULT_RECOVERY_POLICY: dict = {
    "max_recovery_attempts": 2,
    "max_payment_link_amount": 100_000,     # ₹1,00,000
    "cooldown_hours": 24,
    "high_value_threshold": 50_000,         # ₹50,000
    "require_human_approval_above": 50_000,
    "minimum_expected_value": 500,          # ₹500
}


class Merchant(Base):
    """
    Represents a merchant account using the Revenue Recovery Autopilot.

    One merchant can have many customers and payments. The
    recovery_policy_config JSONB column stores configurable guardrail values
    that the policy engine (Phase 4) will read.
    """

    __tablename__ = "merchants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    recovery_policy_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: dict(DEFAULT_RECOVERY_POLICY),
    )
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    customers: Mapped[list["Customer"]] = relationship(  # noqa: F821
        "Customer", back_populates="merchant", cascade="all, delete-orphan"
    )
    payments: Mapped[list["Payment"]] = relationship(  # noqa: F821
        "Payment", back_populates="merchant", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Merchant id={self.id} name={self.name!r}>"
