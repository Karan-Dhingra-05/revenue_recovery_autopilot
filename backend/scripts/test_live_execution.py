import logging
import sys
import uuid
from decimal import Decimal
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from app.config import settings
from app.agent.schemas import AgentProposal, AllowedAction, Priority
from app.database import SessionLocal
from app.executor.service import execute_recovery_action, _find_existing_execution
from app.models.customer import Customer
from app.models.enums import PaymentMethod, PaymentStatus, RecoveryCaseStatus, RecoveryActionType, RecoveryActionStatus
from app.models.merchant import Merchant
from app.models.payment import Payment
from app.models.recovery import RecoveryCase, RecoveryAction
from app.policy.engine import make_policy_decision
from app.policy.schemas import MerchantPolicy
from app.policy.rules import RuleContext
from app.integrations.razorpay_service import EXECUTION_ENABLED

def setup_case(db: Session):
    merchant = Merchant(name="Razorpay Track 3 MVP", currency="INR")
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
        amount=Decimal("1999.00"),  # ₹1999
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

def run():
    db = SessionLocal()
    try:
        print("\n--- 1. SETTING UP TEST CASE ---")
        merchant, customer, payment, case = setup_case(db)
        
        agent_result = AgentProposal(
            recommended_action=AllowedAction.PAYMENT_LINK,
            priority=Priority.HIGH,
            reason="High probability of recovery via Payment Link.",
            confidence=0.85,
        )
        
        rule_context = RuleContext(
            payment_status=payment.status.value,
            amount=float(payment.amount),
            attempt_number=1,
            hours_since_last_action=None,
            days_since_failure=0.1,
            customer_actions_this_month=0,
            has_active_recovery=False,
        )
        
        decision = make_policy_decision(
            txn_features={"amount": float(payment.amount)},
            ml_scores={"PAYMENT_LINK": 0.9, "REMINDER": 0.2, "DELAYED_RETRY": 0.1, "ESCALATE": 0.0},
            agent_result=agent_result,
            rule_context=rule_context,
            policy=MerchantPolicy(),
        )

        print("\n--- 2. PRE-FLIGHT CHECKS ---")
        print(f"Recovery Case ID       : {case.id}")
        print(f"Payment ID             : {payment.id}")
        print(f"Trusted DB Amount      : ₹{payment.amount}")
        print(f"Policy Decision        : {decision.decision.value}")
        print(f"Final Action           : {decision.final_action.value if decision.final_action else None}")
        print(f"Execution Enabled Flag : {EXECUTION_ENABLED}")

        if decision.decision.value != "ALLOW" or decision.final_action.value != "PAYMENT_LINK":
            print("Decision is not ALLOW + PAYMENT_LINK. Stopping.")
            return

        print("\n--- 3. EXECUTING TO RAZORPAY TEST API ---")
        result = execute_recovery_action(decision, case, payment, db)
        
        print("\n--- 4. POST-EXECUTION REPORT ---")
        print(f"Creation Succeeded      : {result.success}")
        print(f"Razorpay Payment Link ID: {result.external_reference}")
        print(f"Payment Link URL        : {result.payment_link_url}")
        print(f"Recovery Case ID        : {case.id}")
        print(f"Recovery Action ID      : {result.action_id}")
        print(f"Amount sent to Razorpay : {int(payment.amount * 100)} paise")
        
        print("\n--- 5. DATABASE VERIFICATIONS ---")
        # 1. Verify Payment Link ID and URL stored
        action = db.query(RecoveryAction).filter(RecoveryAction.id == result.action_id).first()
        if action and action.external_reference == result.external_reference and action.payment_link_url == result.payment_link_url:
            print("✅ Payment Link ID and URL successfully stored in PostgreSQL.")
        else:
            print("❌ Failed to store Payment Link details.")
            
        # 2. Verify case is NOT marked recovered
        db.refresh(case)
        if case.status == RecoveryCaseStatus.OPEN:
            print("✅ Recovery case is still OPEN (NOT marked RECOVERED).")
        else:
            print(f"❌ Recovery case status changed unexpectedly to {case.status.value}.")
            
        # 3. Verify execution state
        print(f"✅ Recovery action status is {action.status.value}.")
        
        # 4. Verify no duplicate payment link
        duplicate_result = execute_recovery_action(decision, case, payment, db)
        if duplicate_result.already_existed:
            print("✅ Idempotency confirmed: Duplicate execution returned existing link.")
        else:
            print("❌ Duplicate execution created a new link!")
            
        db.commit()

    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    run()
