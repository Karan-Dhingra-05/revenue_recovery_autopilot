import sys
from decimal import Decimal
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.recovery import RecoveryCase, RecoveryAction, RecoveryOutcome
from app.models.audit import AuditLog
from app.models.enums import RecoveryCaseStatus, RecoveryActionStatus, OutcomeType

def verify_db():
    db = SessionLocal()
    try:
        # Find the most recently created RecoveryOutcome
        action = db.query(RecoveryAction).filter(RecoveryAction.external_reference == "plink_TWodHxm7u3Q26O").first()
        if not action:
            print("Real RecoveryAction not found!")
            return
        
        case = db.query(RecoveryCase).filter(RecoveryCase.id == action.recovery_case_id).first()
        outcome = db.query(RecoveryOutcome).filter(RecoveryOutcome.action_id == action.id).first()
        if not outcome:
            print("No RecoveryOutcome found for the real payment link.")
            return
        outcomes = db.query(RecoveryOutcome).filter(RecoveryOutcome.recovery_case_id == case.id).all()
        audits = db.query(AuditLog).filter(
            AuditLog.recovery_case_id == case.id,
            AuditLog.event_type == "WEBHOOK_PROCESSED"
        ).all()

        print(f"--- Verification for Case {str(case.id)[:8]}... ---")
        
        print(f"1. RecoveryCase status is RECOVERED: {case.status == RecoveryCaseStatus.RECOVERED} (Current: {case.status.value})")
        print(f"2. RecoveryAction status is COMPLETED: {action.status == RecoveryActionStatus.COMPLETED} (Current: {action.status.value})")
        print(f"3. RecoveryOutcome exists: True")
        print(f"4. RecoveryOutcome status/type is RECOVERED: {outcome.outcome_type == OutcomeType.RECOVERED} (Current: {outcome.outcome_type.value})")
        
        amount_matches = (outcome.recovered_amount == Decimal("1999.00"))
        print(f"5. Recovered amount is ₹1,999.00: {amount_matches} (Current: {outcome.recovered_amount})")
        
        print(f"6. Razorpay Payment Link ID is correctly associated: {bool(action.external_reference)} (ID: {action.external_reference})")
        
        has_event_id = bool(outcome.razorpay_event_id)
        print(f"7. Razorpay event ID is stored: {has_event_id} (Event ID: {outcome.razorpay_event_id})")
        
        print(f"8. Exactly ONE RecoveryOutcome exists for this recovery case: {len(outcomes) == 1} (Count: {len(outcomes)})")
        
        print(f"9. Audit log exists for the recovery outcome: {len(audits) >= 1} (Count: {len(audits)})")
        
        # Check no duplicate recovery amount
        # Since exactly ONE outcome exists, there is no duplicate recovery amount.
        print(f"10. No duplicate recovery amount has been recorded: {len(outcomes) == 1}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    verify_db()
