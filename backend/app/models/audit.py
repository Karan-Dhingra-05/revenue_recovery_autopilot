import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum as SAEnum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base
from app.models.enums import AuditActorType

if TYPE_CHECKING:
    from app.models.recovery import RecoveryCase

class AuditLog(Base):
    """
    Immutable record of every significant decision or action in the system.

    AI decisions, policy outcomes, execution events, and human interventions
    all write AuditLog entries. Rows must never be updated or deleted —
    they form the complete, tamper-evident history of the system.

    'recovery_case_id' is nullable so system-wide events (startup, batch
    processing, etc.) can also be logged without being tied to a specific case.

    'metadata_json' stores structured data specific to each event type:
    ML scores, LLM responses, policy evaluation details, Razorpay webhook
    payloads, etc.
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    recovery_case_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recovery_cases.id", ondelete="SET NULL"),
        nullable=True,
    )
    actor_type: Mapped[AuditActorType] = mapped_column(
        SAEnum(AuditActorType, name="audit_actor_type"), nullable=False
    )
    actor_name: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    input_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decision_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    policy_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Arbitrary structured metadata: ML scores, LLM response, policy details, etc.
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )

    # Relationships
    case: Mapped[Optional["RecoveryCase"]] = relationship(  # noqa: F821
        "RecoveryCase", back_populates="audit_logs"
    )

    __table_args__ = (
        Index("ix_audit_logs_case_id", "recovery_case_id"),
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_event_type", "event_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} "
            f"actor={self.actor_type} "
            f"event={self.event_type!r}>"
        )
