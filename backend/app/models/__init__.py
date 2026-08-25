"""
Import all ORM models so SQLAlchemy's mapper registry and Alembic's
autogenerate can discover them through Base.metadata.

When a new model file is added, import it here.
"""

from app.models.merchant import Merchant  # noqa: F401
from app.models.customer import Customer  # noqa: F401
from app.models.payment import Payment, PaymentFailure  # noqa: F401
from app.models.recovery import RecoveryCase, RecoveryAction, RecoveryOutcome  # noqa: F401
from app.models.audit import AuditLog  # noqa: F401

__all__ = [
    "Merchant",
    "Customer",
    "Payment",
    "PaymentFailure",
    "RecoveryCase",
    "RecoveryAction",
    "RecoveryOutcome",
    "AuditLog",
]
