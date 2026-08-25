"""
Enumerations for the Revenue Recovery Autopilot data model.

Using str-based enums so values are JSON-serialisable and readable in logs
without any extra conversion step. PostgreSQL stores them as native enum types.
"""

import enum


class PaymentStatus(str, enum.Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REFUNDED = "REFUNDED"


class PaymentMethod(str, enum.Enum):
    CARD = "CARD"
    UPI = "UPI"
    NET_BANKING = "NET_BANKING"
    WALLET = "WALLET"
    EMI = "EMI"
    OTHER = "OTHER"


class RecoveryCaseStatus(str, enum.Enum):
    """Lifecycle states of a RecoveryCase."""

    OPEN = "OPEN"            # active — eligible for recovery actions
    RECOVERED = "RECOVERED"  # payment was successfully recovered
    FAILED = "FAILED"        # all attempts exhausted / failed
    EXPIRED = "EXPIRED"      # recovery window has passed
    ESCALATED = "ESCALATED"  # requires human review
    STOPPED = "STOPPED"      # halted by policy engine


class RecoveryActionType(str, enum.Enum):
    PAYMENT_LINK = "PAYMENT_LINK"
    REMINDER = "REMINDER"
    ESCALATE = "ESCALATE"
    DELAYED_RETRY = "DELAYED_RETRY"


class RecoveryActionStatus(str, enum.Enum):
    """Lifecycle states of a single RecoveryAction attempt."""

    PROPOSED = "PROPOSED"              # AI/ML proposed; awaiting policy check
    POLICY_BLOCKED = "POLICY_BLOCKED"  # rejected by policy engine
    APPROVED = "APPROVED"              # passed policy; queued for execution
    EXECUTING = "EXECUTING"            # in-flight (Razorpay call in progress)
    COMPLETED = "COMPLETED"            # execution finished; outcome TBD
    FAILED = "FAILED"                  # execution failed (e.g. API error)
    CANCELLED = "CANCELLED"            # cancelled before execution started


class OutcomeType(str, enum.Enum):
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    ESCALATED = "ESCALATED"
    STOPPED = "STOPPED"


class AuditActorType(str, enum.Enum):
    SYSTEM = "SYSTEM"
    ML_MODEL = "ML_MODEL"
    LLM_AGENT = "LLM_AGENT"
    POLICY_ENGINE = "POLICY_ENGINE"
    HUMAN = "HUMAN"
