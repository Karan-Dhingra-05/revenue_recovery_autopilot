"""
Offline action-selection policy simulation.

Compares three policies on the held-out test set:

  1. **Baseline 1 — Always PAYMENT_LINK**
     Pick PAYMENT_LINK for every failed transaction.

  2. **Baseline 2 — Always best historical action**
     Pick whichever action had the highest recovery rate in training data.

  3. **ML policy**
     For each failed transaction, score all four actions with the model,
     compute expected_recovery = P(recovery) × amount, and select the
     action with the highest expected recovery.

For each policy, we report:
  - Recovered revenue (actual, from held-out labels — NOT predicted)
  - Recovery rate
  - Average recovered amount per failed payment
  - Action distribution
  - Revenue uplift vs. Baseline 1

Usage
-----
    cd backend
    python -m app.ml.policy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features import ALL_FEATURES, LABEL, get_X_y
from app.ml.train import _load_split, load_model

ACTIONS = ["PAYMENT_LINK", "REMINDER", "DELAYED_RETRY", "ESCALATE"]


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


# ── Policy implementations ──────────────────────────────────────────────────


def policy_always_action(test_df: pd.DataFrame, action: str) -> pd.DataFrame:
    """
    Fixed-action policy: always choose the given action.

    Returns the rows from test_df where action_type == action.
    Each row's 'recovered' column is the actual held-out outcome.
    """
    selected = test_df[test_df["action_type"] == action].copy()
    selected["selected_action"] = action
    return selected


def policy_best_historical(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pick the action with the highest empirical recovery rate in training data.
    Apply that same action to every test transaction.
    """
    rr_by_action = train_df.groupby("action_type")["recovered"].mean()
    best_action = rr_by_action.idxmax()
    print(f"  Best historical action: {best_action} (train RR={rr_by_action[best_action]:.4f})")
    return policy_always_action(test_df, best_action)


def policy_ml(
    test_df: pd.DataFrame,
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    """
    ML-based policy: score all actions, pick the one with highest
    expected_recovery = P(recovery) × amount.

    Uses the actual held-out outcome (not the predicted probability)
    as the realized result.
    """
    pipe = load_model(model_name)
    X_test, _ = get_X_y(test_df)

    # Score every (transaction, action) pair.
    y_proba = pipe.predict_proba(X_test)[:, 1]
    scored = test_df.copy()
    scored["predicted_proba"] = y_proba
    scored["expected_recovery"] = scored["predicted_proba"] * scored["amount"]

    # For each unique transaction, pick the action with max expected recovery.
    best_idx = scored.groupby("transaction_id")["expected_recovery"].idxmax()
    selected = scored.loc[best_idx].copy()
    selected["selected_action"] = selected["action_type"]

    return selected


# ── Reporting ────────────────────────────────────────────────────────────────


def report_policy(name: str, selected: pd.DataFrame, total_at_risk: float) -> dict:
    """Print a summary for a policy and return key metrics."""
    n_txns = len(selected)
    n_recovered = selected["recovered"].sum()
    recovered_revenue = selected.loc[selected["recovered"] == 1, "amount"].sum()
    recovery_rate = n_recovered / n_txns if n_txns > 0 else 0.0
    avg_per_txn = recovered_revenue / n_txns if n_txns > 0 else 0.0

    print(f"  Transactions:           {n_txns:>8,}")
    print(f"  Recovered count:        {n_recovered:>8,}")
    print(f"  Recovery rate:          {recovery_rate:>8.2%}")
    print(f"  Revenue at risk:      ₹{total_at_risk:>12,.2f}")
    print(f"  Recovered revenue:    ₹{recovered_revenue:>12,.2f}")
    print(f"  Avg recovered / txn:  ₹{avg_per_txn:>12,.2f}")

    # Action distribution.
    if "selected_action" in selected.columns:
        print(f"\n  Action distribution:")
        action_counts = selected["selected_action"].value_counts()
        for action in ACTIONS:
            count = action_counts.get(action, 0)
            pct = count / n_txns * 100 if n_txns > 0 else 0
            print(f"    {action:<16} {count:>5}  ({pct:.1f}%)")

    return {
        "policy": name,
        "n_txns": n_txns,
        "n_recovered": n_recovered,
        "recovery_rate": recovery_rate,
        "recovered_revenue": recovered_revenue,
        "avg_per_txn": avg_per_txn,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline policy simulation.")
    parser.add_argument(
        "--model",
        default="lightgbm",
        choices=["lightgbm", "logistic_regression"],
        help="Model for ML policy (default: lightgbm)",
    )
    args = parser.parse_args()

    train_df = _load_split("train")
    test_df = _load_split("test")

    # Total revenue at risk = sum of unique failed-transaction amounts.
    unique_txns = test_df.drop_duplicates("transaction_id")
    total_at_risk = unique_txns["amount"].sum()
    n_unique_txns = len(unique_txns)

    print(f"Test set: {len(test_df)} rows, {n_unique_txns} unique transactions")
    print(f"Revenue at risk: ₹{total_at_risk:,.2f}")

    results = []

    # Baseline 1: Always PAYMENT_LINK
    _section("Baseline 1 — Always PAYMENT_LINK")
    b1 = policy_always_action(test_df, "PAYMENT_LINK")
    r1 = report_policy("Always PAYMENT_LINK", b1, total_at_risk)
    results.append(r1)

    # Baseline 2: Best historical action
    _section("Baseline 2 — Best Historical Action")
    b2 = policy_best_historical(train_df, test_df)
    r2 = report_policy("Best Historical", b2, total_at_risk)
    results.append(r2)

    # ML Policy
    _section(f"ML Policy — {args.model}")
    ml = policy_ml(test_df, model_name=args.model)
    r_ml = report_policy(f"ML ({args.model})", ml, total_at_risk)
    results.append(r_ml)

    # ── Comparison ───────────────────────────────────────────────────────
    _section("Comparison")
    baseline_rev = r1["recovered_revenue"]
    print(f"  {'Policy':<28} {'Recovered':>12} {'Rate':>8} {'Uplift vs B1':>14}")
    for r in results:
        uplift = r["recovered_revenue"] - baseline_rev
        uplift_pct = (uplift / baseline_rev * 100) if baseline_rev > 0 else 0
        sign = "+" if uplift >= 0 else ""
        print(
            f"  {r['policy']:<28} "
            f"₹{r['recovered_revenue']:>11,.2f} "
            f"{r['recovery_rate']:>7.2%} "
            f"{sign}₹{uplift:>10,.2f} ({sign}{uplift_pct:.1f}%)"
        )

    print()


if __name__ == "__main__":
    main()
