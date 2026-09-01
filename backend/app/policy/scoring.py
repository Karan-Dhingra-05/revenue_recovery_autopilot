"""
Scoring functions for the Policy Engine.

Calculates the economic value of each recovery action based on the ML
probabilities and the Merchant Policy action costs. 

Expected Recovery = P(recovery) * amount
Expected Net Recovery = Expected Recovery - action_cost - risk_penalty
"""

from __future__ import annotations

from app.agent.schemas import ActionScore, AllowedAction
from app.policy.schemas import MerchantPolicy


def calculate_economics(
    amount: float,
    ml_scores: dict[str, float],
    policy: MerchantPolicy,
) -> tuple[dict[AllowedAction, float], dict[AllowedAction, float]]:
    """
    Calculate expected recovery and expected net recovery for all allowed actions.

    Parameters
    ----------
    amount : float
        The transaction amount at risk.
    ml_scores : dict[str, float]
        Raw probabilities from the ML model (e.g. {"PAYMENT_LINK": 0.61}).
    policy : MerchantPolicy
        Configuration containing action costs and risk penalties.

    Returns
    -------
    tuple[dict[AllowedAction, float], dict[AllowedAction, float]]
        (expected_recovery_dict, expected_net_recovery_dict)
    """
    expected_recovery = {}
    expected_net_recovery = {}

    for action in AllowedAction:
        prob = ml_scores.get(action.value, 0.0)
        
        # Expected Recovery
        er = round(prob * amount, 2)
        expected_recovery[action] = er
        
        # Expected Net Recovery
        config = policy.action_configs.get(action)
        if config and config.is_enabled:
            cost = config.action_cost
            risk = config.risk_penalty
        else:
            # If disabled or missing, apply a massive penalty so it ranks lowest
            cost = 999999.0
            risk = 999999.0

        enr = round(er - cost - risk, 2)
        expected_net_recovery[action] = enr

    return expected_recovery, expected_net_recovery


def get_ml_fallback_action(expected_net_recovery: dict[AllowedAction, float]) -> AllowedAction:
    """
    Determine the best economic candidate for a fallback decision.
    Simply returns the action with the highest expected net recovery.
    """
    # Sort by expected net recovery (descending)
    sorted_actions = sorted(
        expected_net_recovery.items(), 
        key=lambda item: item[1], 
        reverse=True
    )
    return sorted_actions[0][0]
