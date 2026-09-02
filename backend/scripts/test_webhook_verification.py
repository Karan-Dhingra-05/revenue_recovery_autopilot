import hmac
import hashlib
import json
import logging
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from app.main import app
from app.database import SessionLocal
from app.config import settings
from app.models.recovery import RecoveryAction, RecoveryCase, RecoveryOutcome
from app.models.audit import AuditLog
from app.models.enums import RecoveryActionStatus, RecoveryCaseStatus

client = TestClient(app)

def create_signature(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        msg=payload_bytes,
        digestmod=hashlib.sha256
    ).hexdigest()

def setup_fresh_test_data(db: Session):
    from app.models.merchant import Merchant
    from app.models.customer import Customer
    from app.models.payment import Payment
    from app.models.enums import PaymentMethod, PaymentStatus, RecoveryActionType
    import uuid
    from decimal import Decimal
    
    merchant = Merchant(name="Webhook Test", currency="INR")
    db.add(merchant)
    db.flush()
    
    customer = Customer(merchant_id=merchant.id, external_customer_id="cust_wh", email="wh@example.com")
    db.add(customer)
    db.flush()
    
    payment = Payment(merchant_id=merchant.id, customer_id=customer.id, amount=Decimal("1999.00"), currency="INR", payment_method=PaymentMethod.CARD, status=PaymentStatus.FAILED)
    db.add(payment)
    db.flush()
    
    case = RecoveryCase(payment_id=payment.id, status=RecoveryCaseStatus.OPEN, amount_at_risk=payment.amount)
    db.add(case)
    db.flush()
    
    plink_id = f"plink_test_{uuid.uuid4().hex[:8]}"
    action = RecoveryAction(
        recovery_case_id=case.id,
        action_type=RecoveryActionType.PAYMENT_LINK,
        status=RecoveryActionStatus.EXECUTING,
        expected_recovery=Decimal("1999.00"),
        external_reference=plink_id,
        payment_link_url="https://rzp.io/test"
    )
    db.add(action)
    db.commit()
    
    return case.id, plink_id

def run():
    db = SessionLocal()
    try:
        case_id, plink_id = setup_fresh_test_data(db)
        
        # 1. State before
        print("\n--- DATABASE STATE BEFORE ---")
        case = db.query(RecoveryCase).filter(RecoveryCase.id == case_id).first()
        action = db.query(RecoveryAction).filter(RecoveryAction.external_reference == plink_id).first()
        
        print(f"Case ID: {case.id}")
        print(f"Case Status: {case.status.value}")
        print(f"Action Status: {action.status.value}")
        
        # 2. Construct Payload
        payload = {
            "entity": "event",
            "account_id": "acc_00000000000000",
            "event": "payment_link.paid",
            "contains": ["payment_link", "payment"],
            "payload": {
                "payment_link": {
                    "entity": {
                        "id": plink_id,
                        "status": "paid",
                        "amount": 199900,
                        "amount_paid": 199900,
                    }
                },
                "payment": {
                    "entity": {
                        "id": "pay_xyz",
                        "amount": 199900,
                        "status": "captured",
                        "method": "card"
                    }
                }
            },
            "created_at": 1600000000
        }
        
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode("utf-8")
        secret = settings.razorpay_webhook_secret
        valid_signature = create_signature(payload_bytes, secret)
        import uuid
        event_id = f"ev_webhook_test_{uuid.uuid4().hex[:8]}"
        
        print("\n--- 1. FIRST WEBHOOK RESULT (VALID) ---")
        headers = {
            "X-Razorpay-Signature": valid_signature,
            "x-razorpay-event-id": event_id,
            "Content-Type": "application/json"
        }
        
        response = client.post("/api/webhooks/razorpay", content=payload_bytes, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Refresh and check state
        db.refresh(case)
        db.refresh(action)
        outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.recovery_case_id == case.id).first()
        audit = db.query(AuditLog).filter(
            AuditLog.recovery_case_id == case.id, 
            AuditLog.event_type == "WEBHOOK_PROCESSED"
        ).first()
        
        print(f"\nAfter First Webhook:")
        print(f"Case Status: {case.status.value} (Expected: RECOVERED)")
        print(f"Action Status: {action.status.value} (Expected: COMPLETED)")
        print(f"Outcome Created: {outcome is not None}")
        if outcome:
            print(f"Recovered Amount: ₹{outcome.recovered_amount}")
        print(f"Audit Record Created: {audit is not None}")
        
        print("\n--- 2. DUPLICATE WEBHOOK RESULT ---")
        response_dup = client.post("/api/webhooks/razorpay", content=payload_bytes, headers=headers)
        print(f"Status Code: {response_dup.status_code}")
        print(f"Response: {response_dup.json()}")
        
        outcomes_count = db.query(RecoveryOutcome).filter(RecoveryOutcome.recovery_case_id == case.id).count()
        print(f"Total Outcomes for Case: {outcomes_count} (Expected: 1)")
        
        print("\n--- 3. INVALID SIGNATURE RESULT ---")
        invalid_signature = "invalid_hash_abc123"
        headers_invalid = headers.copy()
        headers_invalid["X-Razorpay-Signature"] = invalid_signature
        headers_invalid["x-razorpay-event-id"] = "ev_webhook_test_002"
        
        response_inv = client.post("/api/webhooks/razorpay", content=payload_bytes, headers=headers_invalid)
        print(f"Status Code: {response_inv.status_code} (Expected: 400)")
        print(f"Response: {response_inv.json()}")
        
        db.refresh(case)
        print(f"Case Status after Invalid Signature: {case.status.value}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    run()
