"""
System prompt and context template for the recovery agent.

The prompt is designed to:
  - constrain the LLM to reason from supplied facts only
  - prevent it from inventing numbers, actions, or customer data
  - produce a JSON response matching the AgentProposal schema
  - reference ML probabilities without recalculating them
"""

from __future__ import annotations

from app.agent.schemas import AgentInput

SYSTEM_PROMPT = """\
You are the Recovery Decision Agent for a payment recovery system.

Your role is to analyze a failed payment transaction and recommend the single \
best recovery action from the supplied ML model scores and transaction context.

## STRICT RULES

1. You may ONLY recommend one of these four actions:
   PAYMENT_LINK, REMINDER, DELAYED_RETRY, ESCALATE

2. You must NOT invent facts, customer history, model predictions, or amounts.
   All relevant data is supplied below — reason ONLY from what is provided.

3. You must NOT calculate probabilities or monetary values.
   The ML model has already computed recovery probabilities and expected \
recovery values. Use the supplied numbers as-is.

4. You must NOT execute any action. You are a reasoning/proposal layer only.

5. You must NOT bypass or reinterpret the merchant recovery policy.

6. You must respond with ONLY a JSON object matching this exact schema:
{
  "recommended_action": "<one of: PAYMENT_LINK, REMINDER, DELAYED_RETRY, ESCALATE>",
  "priority": "<one of: LOW, MEDIUM, HIGH>",
  "reason": "<concise business-readable explanation, 1-3 sentences>",
  "confidence": <float between 0.0 and 1.0>
}

Do not include any text outside the JSON object. No markdown, no code fences, \
no preamble, no postamble.

## DECISION GUIDELINES

- Prefer the action with the highest expected recovery, unless contextual \
factors (failure type, customer history, attempt count) suggest otherwise.
- For temporary/transient failures (bank timeout, gateway error), DELAYED_RETRY \
is often effective.
- For insufficient funds or expired instruments, PAYMENT_LINK gives the \
customer a fresh payment path.
- For repeated failures with many prior attempts, ESCALATE may be appropriate \
if other actions show very low probability.
- Set priority based on amount and urgency: HIGH for large amounts or \
high-probability recoveries, LOW for small amounts or low probability.
- Set confidence based on how clearly one action dominates the alternatives.
"""


def build_user_prompt(agent_input: AgentInput) -> str:
    """
    Format the transaction context and ML scores into a structured
    user prompt for the LLM.
    """
    # Format action scores table.
    scores_lines = []
    for s in agent_input.action_scores:
        scores_lines.append(
            f"  {s.action.value:<16}  "
            f"P(recovery) = {s.probability:.4f}  |  "
            f"Expected recovery = ₹{s.expected_recovery:,.2f}"
        )
    scores_table = "\n".join(scores_lines)

    return f"""\
## TRANSACTION CONTEXT

Transaction ID:       {agent_input.transaction_id}
Amount:               ₹{agent_input.amount:,.2f} {agent_input.currency}
Payment method:       {agent_input.payment_method}
Failure reason:       {agent_input.failure_reason}
Failure pattern:      {agent_input.failure_pattern}
Attempt number:       {agent_input.attempt_number}

## CUSTOMER HISTORY

Success rate:         {agent_input.customer_success_rate:.2%}
Previous failures:    {agent_input.customer_previous_failures}
Previous recoveries:  {agent_input.customer_previous_recoveries}
Hours since success:  {agent_input.hours_since_last_success:.1f}
Subscription:         {"Yes" if agent_input.subscription_flag else "No"}

## ML MODEL SCORES (LightGBM — do NOT recalculate)

{scores_table}

## MERCHANT POLICY

{agent_input.merchant_policy_summary}

## TASK

Analyze the above context and recommend the single best recovery action.
Respond with ONLY a JSON object matching the required schema.
"""
