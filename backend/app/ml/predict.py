"""
Inference helper for the recovery-probability model.

Provides a clean `score_actions()` function that takes a single
transaction's features and returns P(recovery | action) for each
candidate action.

Usage
-----
    from app.ml.predict import score_actions

    scores = score_actions(transaction_features_dict)
    # → {"PAYMENT_LINK": 0.72, "REMINDER": 0.45, ...}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml.features import ALL_FEATURES
from app.ml.train import load_model

ACTIONS = ["PAYMENT_LINK", "REMINDER", "DELAYED_RETRY", "ESCALATE"]


def score_actions(
    txn_features: dict,
    model_name: str = "lightgbm",
) -> dict[str, float]:
    """
    Score all four candidate actions for a single transaction.

    Parameters
    ----------
    txn_features : dict
        Transaction features (must contain all columns in ALL_FEATURES
        except 'action_type', which is injected per action).
    model_name : str
        Name of the saved model to load ('lightgbm' or 'logistic_regression').

    Returns
    -------
    dict[str, float]
        Mapping from action_type to predicted P(recovery).
    """
    pipe = load_model(model_name)
    scores = {}

    for action in ACTIONS:
        row = dict(txn_features)
        row["action_type"] = action
        # Build a single-row DataFrame with the expected feature columns.
        df = pd.DataFrame([row])[ALL_FEATURES]
        proba = pipe.predict_proba(df)[:, 1][0]
        scores[action] = round(float(proba), 4)

    return scores
