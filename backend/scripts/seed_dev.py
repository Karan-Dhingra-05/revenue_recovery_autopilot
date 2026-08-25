#!/usr/bin/env python3
"""
Development seed script.

Creates a minimal set of merchants, customers, payments (some failed) and
recovery cases so the schema can be manually explored without running the
full synthetic dataset generator (Phase 3).

Usage:
    cd backend
    python scripts/seed_dev.py

Safe to re-run: skips seeding if the demo merchant already exists.
"""

import sys
from decimal import Decimal
from pathlib import Path

# Allow running as `python scripts/seed_dev.py` from the backend/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal  # noqa: E402
from app.models import *  # noqa: F401, F403 — registers all models with Base
from app.models.audit import AuditLog  # noqa: E402
from app.models.customer import Customer  # noqa: E402
from app.models.enums import (  # noqa: E402
    AuditActorType,
    PaymentMethod,
    PaymentStatus,
    RecoveryCaseStatus,
)
from app.models.merchant import Merchant  # noqa: E402
from app.models.payment import Payment, PaymentFailure  # noqa: E402
from app.models.recovery import RecoveryCase  # noqa: E402


_MERCHANT_NAME = "Demo Merchant Ltd."


def seed() -> None:
    with SessionLocal() as db:
        # ── Idempotency guard ─────────────────────────────────────────────
        if db.query(Merchant).filter_by(name=_MERCHANT_NAME).first():
            print(f"Seed data for '{_MERCHANT_NAME}' already exists. Skipping.")
            return

        # ── Merchant ──────────────────────────────────────────────────────
        merchant = Merchant(name=_MERCHANT_NAME, currency="INR")
        db.add(merchant)
        db.flush()

        # ── Customers ─────────────────────────────────────────────────────
        # A reliable repeat customer (good history)
        alice = Customer(
            merchant_id=merchant.id,
            external_customer_id="cust_001",
            email="alice@demo.example.com",
            successful_payment_count=8,
            failed_payment_count=1,
            total_paid=Decimal("42500.00"),
        )
        # A moderate customer who has failed before
        bob = Customer(
            merchant_id=merchant.id,
            external_customer_id="cust_002",
            email="bob@demo.example.com",
            successful_payment_count=3,
            failed_payment_count=2,
            total_paid=Decimal("12000.00"),
        )
        # A new customer with no successful payments
        charlie = Customer(
            merchant_id=merchant.id,
            external_customer_id="cust_003",
            email="charlie@demo.example.com",
            successful_payment_count=0,
            failed_payment_count=3,
            total_paid=Decimal("0.00"),
        )
        db.add_all([alice, bob, charlie])
        db.flush()

        # ── Payments ──────────────────────────────────────────────────────
        # One successful payment (no recovery case needed)
        p_ok = Payment(
            merchant_id=merchant.id,
            customer_id=alice.id,
            amount=Decimal("7500.00"),
            currency="INR",
            payment_method=PaymentMethod.UPI,
            status=PaymentStatus.SUCCESS,
        )
        # Three failed payments → will become recovery cases
        p_fail_1 = Payment(
            merchant_id=merchant.id,
            customer_id=alice.id,
            amount=Decimal("15000.00"),
            currency="INR",
            payment_method=PaymentMethod.CARD,
            status=PaymentStatus.FAILED,
        )
        p_fail_2 = Payment(
            merchant_id=merchant.id,
            customer_id=bob.id,
            amount=Decimal("4800.00"),
            currency="INR",
            payment_method=PaymentMethod.NET_BANKING,
            status=PaymentStatus.FAILED,
        )
        p_fail_3 = Payment(
            merchant_id=merchant.id,
            customer_id=charlie.id,
            amount=Decimal("950.00"),
            currency="INR",
            payment_method=PaymentMethod.UPI,
            status=PaymentStatus.FAILED,
        )
        db.add_all([p_ok, p_fail_1, p_fail_2, p_fail_3])
        db.flush()

        # ── Payment Failures ──────────────────────────────────────────────
        failures = [
            PaymentFailure(
                payment_id=p_fail_1.id,
                failure_code="INSUFFICIENT_FUNDS",
                failure_reason="Customer bank account had insufficient funds.",
                failure_source="bank",
            ),
            PaymentFailure(
                payment_id=p_fail_2.id,
                failure_code="GATEWAY_TIMEOUT",
                failure_reason="Net banking gateway timed out after 30 seconds.",
                failure_source="gateway",
            ),
            PaymentFailure(
                payment_id=p_fail_3.id,
                failure_code="INVALID_UPI_ID",
                failure_reason="UPI VPA could not be resolved by the PSP.",
                failure_source="upi",
            ),
        ]
        db.add_all(failures)
        db.flush()

        # ── Recovery Cases ────────────────────────────────────────────────
        cases = [
            RecoveryCase(
                payment_id=p_fail_1.id,
                status=RecoveryCaseStatus.OPEN,
                amount_at_risk=p_fail_1.amount,
            ),
            RecoveryCase(
                payment_id=p_fail_2.id,
                status=RecoveryCaseStatus.OPEN,
                amount_at_risk=p_fail_2.amount,
            ),
            RecoveryCase(
                payment_id=p_fail_3.id,
                status=RecoveryCaseStatus.OPEN,
                amount_at_risk=p_fail_3.amount,
            ),
        ]
        db.add_all(cases)
        db.flush()

        # ── Audit Log — seed event ────────────────────────────────────────
        db.add(
            AuditLog(
                actor_type=AuditActorType.SYSTEM,
                actor_name="seed_dev",
                event_type="SEED_COMPLETED",
                decision_summary=(
                    f"Dev seed created: 1 merchant, 3 customers, "
                    f"4 payments (3 failed), 3 recovery cases."
                ),
                metadata_json={"merchant_id": str(merchant.id)},
            )
        )

        db.commit()

        total_at_risk = sum(c.amount_at_risk for c in cases)
        print(f"✓ Merchant:        {merchant.name}")
        print(f"  id: {merchant.id}")
        print(f"✓ Customers:       3  (alice, bob, charlie)")
        print(f"✓ Payments:        4  (1 success, 3 failed)")
        print(f"✓ Failures:        {len(failures)}")
        print(f"✓ Recovery cases:  {len(cases)}  open")
        print(f"  Revenue at risk: ₹{total_at_risk:,.2f}")
        print("✓ Audit log entry written.")


def main() -> None:
    print("Seeding development database…")
    seed()
    print("Done.")


if __name__ == "__main__":
    main()
