"""
Policy Engine module for the Revenue Recovery Autopilot.

The Policy Engine acts as the final deterministic authority for all automated
recovery decisions. It takes the ML probabilities and the LLM's proposed
action (or ML fallback), evaluates them against a strict set of business rules,
and returns a final decision (ALLOW, MODIFY, BLOCK).
"""
