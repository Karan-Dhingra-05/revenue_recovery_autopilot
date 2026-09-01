"""
Deterministic policy rules.

Each rule is a simple pure function returning a RuleResult.
No database connections are made here; all required state is passed via RuleContext.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from app.agent.schemas import AllowedAction
from app.policy.schemas import MerchantPolicy, RuleResult


class RuleContext(BaseModel):
    """The context required to evaluate all deterministic rules."""
    payment_status: str
    amount: float
    attempt_number: int
    hours_since_last_action: float | None
    days_since_failure: float
    customer_actions_this_month: int
    has_active_recovery: bool


def rule_payment_not_successful(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.payment_status == "SUCCESS":
        return RuleResult(rule_name="payment_not_successful", passed=False, message="Payment is already successful.")
    return RuleResult(rule_name="payment_not_successful", passed=True, message="Payment is not successful.")


def rule_action_is_enabled(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    config = policy.action_configs.get(action)
    if not config or not config.is_enabled:
        return RuleResult(rule_name="action_is_enabled", passed=False, message=f"Action {action.value} is disabled by merchant.")
    return RuleResult(rule_name="action_is_enabled", passed=True, message=f"Action {action.value} is enabled.")


def rule_max_attempts(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.attempt_number > policy.max_automatic_attempts:
        return RuleResult(rule_name="max_attempts", passed=False, message=f"Attempt {ctx.attempt_number} exceeds max allowed ({policy.max_automatic_attempts}).")
    return RuleResult(rule_name="max_attempts", passed=True, message="Within maximum attempt limit.")


def rule_cooldown_satisfied(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.hours_since_last_action is not None and ctx.hours_since_last_action < policy.cooldown_hours:
        return RuleResult(rule_name="cooldown_satisfied", passed=False, message=f"Cooldown of {policy.cooldown_hours}h not met (last action {ctx.hours_since_last_action:.1f}h ago).")
    return RuleResult(rule_name="cooldown_satisfied", passed=True, message="Cooldown period satisfied.")


def rule_max_amount(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.amount > policy.max_amount_eligible:
        return RuleResult(rule_name="max_amount", passed=False, message=f"Amount ₹{ctx.amount:,.2f} exceeds max eligible ₹{policy.max_amount_eligible:,.2f}.")
    return RuleResult(rule_name="max_amount", passed=True, message="Amount is within eligible limits.")


def rule_max_customer_actions(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.customer_actions_this_month >= policy.max_customer_actions_per_month:
        return RuleResult(rule_name="max_customer_actions", passed=False, message=f"Customer has reached {ctx.customer_actions_this_month} actions this month (max {policy.max_customer_actions_per_month}).")
    return RuleResult(rule_name="max_customer_actions", passed=True, message="Customer within monthly action limits.")


def rule_recovery_window(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.days_since_failure > policy.recovery_window_days:
        return RuleResult(rule_name="recovery_window", passed=False, message=f"Recovery window of {policy.recovery_window_days} days expired ({ctx.days_since_failure:.1f} days ago).")
    return RuleResult(rule_name="recovery_window", passed=True, message="Within active recovery window.")


def rule_no_duplicate_active_recovery(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    if ctx.has_active_recovery:
        return RuleResult(rule_name="no_duplicate_active_recovery", passed=False, message="A recovery action is already active for this case.")
    return RuleResult(rule_name="no_duplicate_active_recovery", passed=True, message="No duplicate active recovery.")


def rule_repeated_failure_stopping(ctx: RuleContext, action: AllowedAction, policy: MerchantPolicy) -> RuleResult:
    """
    If a case has hit the maximum attempts, only ESCALATE might be allowed as a final resort
    by some merchants, but by default we might block everything. 
    Actually, let's say if attempts == max_automatic_attempts, only ESCALATE is allowed.
    """
    if ctx.attempt_number == policy.max_automatic_attempts and action != AllowedAction.ESCALATE:
        return RuleResult(rule_name="repeated_failure_stopping", passed=False, message=f"At attempt limit ({policy.max_automatic_attempts}); only ESCALATE is permitted.")
    return RuleResult(rule_name="repeated_failure_stopping", passed=True, message="Repeated failure rule satisfied.")


# Registry of all core policy rules
ALL_RULES: list[Callable[[RuleContext, AllowedAction, MerchantPolicy], RuleResult]] = [
    rule_payment_not_successful,
    rule_no_duplicate_active_recovery,
    rule_recovery_window,
    rule_max_amount,
    rule_max_customer_actions,
    rule_cooldown_satisfied,
    rule_max_attempts,
    rule_repeated_failure_stopping,
    rule_action_is_enabled,
]
