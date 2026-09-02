"""
Phase 6 Demo: End-to-End Recovery Execution Loop.

Runs a single mock recovery case through the entire pipeline:
  ML Scoring → Agent Proposal → Policy Engine → Executor

If RAZORPAY_LIVE_EXECUTION_ENABLED is true and credentials are set,
this will create a real Razorpay Test Mode Payment Link.
"""

import logging
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure we're running from backend root
try:
    from app.config import settings
except ModuleNotFoundError:
    print("Please run this script from the backend directory: python -m scripts.demo_execution")
    sys.exit(1)

from app.agent.schemas import AgentProposal, AllowedAction
from app.database import SessionLocal
from app.executor.service import execute_recovery_action
from app.models.customer import Customer
from app.models.enums import PaymentMethod, PaymentStatus, RecoveryCaseStatus
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.recovery import RecoveryCase
from app.policy.engine import make_policy_decision


def setup_test_case(db: Session) -> tuple[Merchant, Customer, Payment, RecoveryCase]:
    """Create a minimal mock case for demonstration."""
    merchant = Merchant(name="Demo Merchant (Phase 6)", currency="INR")
    db.add(merchant)
    db.flush()

    customer = Customer(
        merchant_id=merchant.id,
        external_customer_id=f"demo_cust_{uuid.uuid4().hex[:6]}",
        email="demo.customer@example.com",
    )
    db.add(customer)
    db.flush()

    payment = Payment(
        merchant_id=merchant.id,
        customer_id=customer.id,
        amount=Decimal("1250.00"),  # ₹1250
        currency="INR",
        payment_method=PaymentMethod.CARD,
        status=PaymentStatus.FAILED,
    )
    db.add(payment)
    db.flush()

    case = RecoveryCase(
        payment_id=payment.id,
        status=RecoveryCaseStatus.OPEN,
        amount_at_risk=payment.amount,
    )
    db.add(case)
    db.commit()

    return merchant, customer, payment, case


def run_demo():
    print("=" * 60)
    print("PHASE 6 DEMO: END-TO-END EXECUTION")
    print("=" * 60)
    print(f"Razorpay Live Execution Enabled: {settings.razorpay_live_execution_enabled}\n")

    db = SessionLocal()
    try:
        # 1. Setup
        print("1. Setting up test data...")
        merchant, customer, payment, case = setup_test_case(db)
        print(f"   Case ID: {case.id}")
        print(f"   Amount:  ₹{case.amount_at_risk}\n")

        # 2. Mock ML & Agent output (Simulating Phases 3 & 4)
        print("2. Simulating ML Scoring & Agent Proposal...")
        expected_recovery = {
            AllowedAction.PAYMENT_LINK: 950.0,
            AllowedAction.REMINDER: 400.0,
            AllowedAction.DELAYED_RETRY: 200.0,
            AllowedAction.ESCALATE: 0.0,
        }
        
        agent_result = AgentProposal(
            recommended_action=AllowedAction.PAYMENT_LINK,
            priority="HIGH",
            reason="Customer has good history and card failure was temporary. High probability of recovery via Payment Link.",
            confidence=0.85,
        )
        print(f"   Agent proposes: {agent_result.recommended_action.value}\n")

        # 3. Policy Engine
        print("3. Running Deterministic Policy Engine...")
        from app.policy.schemas import MerchantPolicy
        from app.policy.rules import RuleContext
        
        rule_context = RuleContext(
            payment_status=payment.status.value,
            amount=float(payment.amount),
            attempt_number=1,
            hours_since_last_action=None,
            days_since_failure=1.5,
            customer_actions_this_month=0,
            has_active_recovery=False,
        )
        
        ml_scores = {
            "PAYMENT_LINK": 0.76,
            "REMINDER": 0.32,
            "DELAYED_RETRY": 0.16,
            "ESCALATE": 0.0,
        }

        decision = make_policy_decision(
            txn_features={"amount": float(payment.amount)},
            ml_scores=ml_scores,
            agent_result=agent_result,
            rule_context=rule_context,
            policy=MerchantPolicy(),
        )
        print(f"   Decision: {decision.decision.value}")
        print(f"   Final Action: {decision.final_action.value if decision.final_action else 'None'}\n")

        if decision.decision != "ALLOW" or not decision.final_action:
            print("   Execution blocked by policy. Exiting.")
            return

        # 4. Execution
        print("4. Executing Final Action...")
        result = execute_recovery_action(decision, case, payment, db)
        
        if result.success:
            db.commit()
            print("\n✅ EXECUTION SUCCESSFUL")
            print("-" * 30)
            print(f"Action ID:    {result.action_id}")
            print(f"Reference ID: {result.external_reference}")
            if result.payment_link_url:
                print(f"Payment Link: {result.payment_link_url}")
            print(f"Message:      {result.message}")
        else:
            db.rollback()
            print("\n❌ EXECUTION FAILED")
            print("-" * 30)
            print(f"Message: {result.message}")

    except Exception as e:
        db.rollback()
        logger.error(f"Demo failed: {e}", exc_info=True)
    finally:
        db.close()
        print("\n" + "=" * 60)


if __name__ == "__main__":
    run_demo()
