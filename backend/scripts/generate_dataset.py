#!/usr/bin/env python3
"""
Synthetic dataset generator for the Revenue Recovery Autopilot.

Produces a realistic, reproducible payment-failure dataset with action-
conditional recovery outcomes.  The generator simulates ~200 customers
making payments over a 90-day window, resulting in ≥ 5 000 transactions.

Usage
-----
    cd backend
    python scripts/generate_dataset.py              # 5 000 txns, seed=42
    python scripts/generate_dataset.py --records 1000 --seed 7

Output (written to data/generated/)
-----------------------------------
    transactions.csv          – all payments (both successful and failed)
    recovery_action_outcomes.csv – per-action recovery labels for failed txns
    train.csv                 – feature + label rows (first 70 % by time)
    validation.csv            – feature + label rows (next 15 % by time)
    test.csv                  – feature + label rows (last 15 % by time)
    dataset_metadata.json     – run metadata (seed, timestamp, row counts)

Design principles
-----------------
* Customer behaviour is **correlated over time**: each customer has a
  reliability archetype (RELIABLE / MODERATE / UNRELIABLE / NEW) that
  governs base failure rates.  Past failures and successes accumulate
  and shift future probabilities.
* Five labelled failure patterns (A–E from the spec) are assigned to
  each failed transaction and drive action-conditional recovery probs.
* Recovery probabilities are **action-specific** — the dataset encodes
  P(recovery | context, action) so the ML model can learn to recommend
  the best action for each case.
* No feature leaks the target: `recovered` is only in the label table,
  never in the transaction features.
* Train / val / test split is **time-aware**: oldest 70 %, next 15 %,
  newest 15 % by `created_at`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYMENT_METHODS = ["CARD", "UPI", "NET_BANKING", "WALLET", "EMI"]
PAYMENT_METHOD_WEIGHTS = [0.35, 0.30, 0.15, 0.12, 0.08]

FAILURE_REASONS = [
    "INSUFFICIENT_FUNDS",
    "BANK_TIMEOUT",
    "AUTHENTICATION_FAILED",
    "CARD_EXPIRED",
    "INVALID_UPI_ID",
    "GATEWAY_ERROR",
    "LIMIT_EXCEEDED",
    "ACCOUNT_BLOCKED",
]

FAILURE_SOURCES = {
    "INSUFFICIENT_FUNDS": "bank",
    "BANK_TIMEOUT": "bank",
    "AUTHENTICATION_FAILED": "customer",
    "CARD_EXPIRED": "customer",
    "INVALID_UPI_ID": "upi",
    "GATEWAY_ERROR": "gateway",
    "LIMIT_EXCEEDED": "bank",
    "ACCOUNT_BLOCKED": "bank",
}

# Recovery actions the ML model will later rank.
RECOVERY_ACTIONS = ["PAYMENT_LINK", "REMINDER", "DELAYED_RETRY", "ESCALATE"]

# Customer reliability archetypes — mix of archetypes across the population.
ARCHETYPES = ["RELIABLE", "MODERATE", "UNRELIABLE", "NEW"]
ARCHETYPE_WEIGHTS = [0.30, 0.35, 0.20, 0.15]

# Base success probability per archetype (before contextual adjustments).
ARCHETYPE_BASE_SUCCESS = {
    "RELIABLE": 0.90,
    "MODERATE": 0.72,
    "UNRELIABLE": 0.45,
    "NEW": 0.65,
}

# Failure-pattern labels (spec patterns A–E)
# Each failed transaction is assigned one pattern.
FAILURE_PATTERNS = [
    "A_TEMP_BANK",       # temporary bank issue — retry likely works
    "B_INSUF_FUNDS",     # insufficient funds — reminder / payment link
    "C_EXPIRED_INSTR",   # expired / invalid instrument — needs new method
    "D_REPEATED_FAIL",   # repeated failure — diminishing returns
    "E_LOW_VALUE",       # low-value case — cost exceeds expected gain
]

# Mapping from failure_reason to most-likely pattern.
REASON_TO_PATTERN: dict[str, str] = {
    "INSUFFICIENT_FUNDS": "B_INSUF_FUNDS",
    "BANK_TIMEOUT": "A_TEMP_BANK",
    "AUTHENTICATION_FAILED": "C_EXPIRED_INSTR",
    "CARD_EXPIRED": "C_EXPIRED_INSTR",
    "INVALID_UPI_ID": "C_EXPIRED_INSTR",
    "GATEWAY_ERROR": "A_TEMP_BANK",
    "LIMIT_EXCEEDED": "B_INSUF_FUNDS",
    "ACCOUNT_BLOCKED": "D_REPEATED_FAIL",
}

# Action-specific BASE recovery probability per pattern.
# These are modulated by customer context later.
# Rows: pattern, Columns: action
# fmt: off
PATTERN_ACTION_PROBS: dict[str, dict[str, float]] = {
    "A_TEMP_BANK": {
        "DELAYED_RETRY": 0.72,
        "PAYMENT_LINK":  0.60,
        "REMINDER":      0.45,
        "ESCALATE":      0.20,
    },
    "B_INSUF_FUNDS": {
        "PAYMENT_LINK":  0.55,
        "REMINDER":      0.50,
        "DELAYED_RETRY": 0.30,
        "ESCALATE":      0.15,
    },
    "C_EXPIRED_INSTR": {
        "PAYMENT_LINK":  0.62,
        "REMINDER":      0.20,
        "DELAYED_RETRY": 0.10,
        "ESCALATE":      0.25,
    },
    "D_REPEATED_FAIL": {
        "PAYMENT_LINK":  0.18,
        "REMINDER":      0.12,
        "DELAYED_RETRY": 0.08,
        "ESCALATE":      0.35,
    },
    "E_LOW_VALUE": {
        "PAYMENT_LINK":  0.40,
        "REMINDER":      0.30,
        "DELAYED_RETRY": 0.25,
        "ESCALATE":      0.05,
    },
}
# fmt: on


# ---------------------------------------------------------------------------
# Customer simulation
# ---------------------------------------------------------------------------


def _generate_customers(rng: np.random.Generator, n_customers: int) -> pd.DataFrame:
    """Create a population of customers with stable attributes."""
    archetypes = rng.choice(ARCHETYPES, size=n_customers, p=ARCHETYPE_WEIGHTS)

    # Subscription flag: reliable customers are more likely to be subscribers.
    sub_probs = {
        "RELIABLE": 0.55,
        "MODERATE": 0.35,
        "UNRELIABLE": 0.15,
        "NEW": 0.25,
    }
    subscription_flags = np.array(
        [rng.random() < sub_probs[a] for a in archetypes]
    )

    # Preferred payment method (correlated with archetype).
    methods = rng.choice(
        PAYMENT_METHODS, size=n_customers, p=PAYMENT_METHOD_WEIGHTS
    )

    return pd.DataFrame(
        {
            "customer_id": [f"cust_{i:04d}" for i in range(n_customers)],
            "archetype": archetypes,
            "subscription_flag": subscription_flags.astype(int),
            "preferred_method": methods,
            # Running counters — updated as transactions are generated.
            "success_count": np.zeros(n_customers, dtype=int),
            "fail_count": np.zeros(n_customers, dtype=int),
            "total_paid": np.zeros(n_customers, dtype=float),
            "previous_recoveries": np.zeros(n_customers, dtype=int),
            "last_success_ts": pd.Series([pd.NaT] * n_customers, dtype="datetime64[ns, UTC]"),
        }
    )


# ---------------------------------------------------------------------------
# Transaction generation (with behavioural correlation)
# ---------------------------------------------------------------------------


def _choose_amount(rng: np.random.Generator, archetype: str, is_subscription: bool) -> float:
    """Pick a realistic transaction amount (INR)."""
    if is_subscription:
        # Subscription amounts cluster around fixed tiers.
        tier = rng.choice([299, 499, 999, 1999, 4999], p=[0.15, 0.25, 0.30, 0.20, 0.10])
        return float(tier)

    # Non-subscription: log-normal-ish distribution, shifted by archetype.
    base_mu = {"RELIABLE": 8.2, "MODERATE": 7.5, "UNRELIABLE": 6.8, "NEW": 7.0}
    amount = float(rng.lognormal(mean=base_mu[archetype], sigma=0.9))
    return round(min(max(amount, 50), 500_000), 2)  # ₹50–₹5,00,000


def _decide_outcome(
    rng: np.random.Generator,
    archetype: str,
    success_count: int,
    fail_count: int,
    attempt_number: int,
) -> bool:
    """
    Decide whether this payment succeeds.

    The probability is driven by archetype base rate plus contextual
    adjustments for history and attempt number.
    """
    p = ARCHETYPE_BASE_SUCCESS[archetype]

    # Good history boosts success; bad history lowers it.
    total = success_count + fail_count
    if total > 0:
        success_rate = success_count / total
        p += (success_rate - 0.5) * 0.10  # ±5 pp max shift from history

    # Later attempts on same logical payment are slightly less likely.
    if attempt_number > 1:
        p -= 0.05 * min(attempt_number - 1, 4)

    p = max(0.05, min(p, 0.98))  # clamp
    return bool(rng.random() < p)


def _assign_failure_reason(
    rng: np.random.Generator,
    archetype: str,
    attempt_number: int,
) -> str:
    """Pick a failure reason influenced by archetype and attempt number."""
    # Build per-archetype weights for failure reasons.
    weights = np.ones(len(FAILURE_REASONS))

    if archetype == "RELIABLE":
        # Reliable customers mainly fail on transient issues.
        weights[FAILURE_REASONS.index("BANK_TIMEOUT")] = 4.0
        weights[FAILURE_REASONS.index("GATEWAY_ERROR")] = 3.0
    elif archetype == "UNRELIABLE":
        weights[FAILURE_REASONS.index("INSUFFICIENT_FUNDS")] = 5.0
        weights[FAILURE_REASONS.index("ACCOUNT_BLOCKED")] = 2.0
    elif archetype == "NEW":
        weights[FAILURE_REASONS.index("AUTHENTICATION_FAILED")] = 3.0
        weights[FAILURE_REASONS.index("INVALID_UPI_ID")] = 2.0

    if attempt_number >= 3:
        weights[FAILURE_REASONS.index("ACCOUNT_BLOCKED")] += 2.0

    weights /= weights.sum()
    return str(rng.choice(FAILURE_REASONS, p=weights))


def _assign_failure_pattern(
    failure_reason: str,
    amount: float,
    attempt_number: int,
) -> str:
    """Deterministic mapping from failure context to the spec pattern label."""
    if attempt_number >= 3:
        return "D_REPEATED_FAIL"
    if amount < 300:
        return "E_LOW_VALUE"
    return REASON_TO_PATTERN.get(failure_reason, "A_TEMP_BANK")


def generate_transactions(
    rng: np.random.Generator,
    n_records: int,
    start_date: datetime,
    end_date: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a stream of payment transactions over [start_date, end_date].

    Returns
    -------
    transactions : DataFrame
        Every payment (SUCCESS + FAILED) with contextual features.
    customers_final : DataFrame
        Final customer-level aggregates (for reference, not ML input).
    """
    n_customers = max(50, n_records // 25)  # ~25 txns per customer on average
    customers = _generate_customers(rng, n_customers)

    rows: list[dict] = []
    span_seconds = int((end_date - start_date).total_seconds())

    # For each transaction, pick a customer and simulate temporal behaviour.
    txn_idx = 0
    # Track per-customer attempt counters for logical payment groups.
    customer_attempt: dict[str, int] = {}  # customer_id → current attempt #

    while txn_idx < n_records:
        ci = int(rng.integers(0, len(customers)))
        cust = customers.iloc[ci]
        cid = cust["customer_id"]

        # Random timestamp within the window.
        offset = int(rng.integers(0, span_seconds))
        created_at = start_date + timedelta(seconds=offset)

        is_sub = bool(cust["subscription_flag"])
        amount = _choose_amount(rng, cust["archetype"], is_sub)
        method = (
            cust["preferred_method"]
            if rng.random() < 0.7
            else rng.choice(PAYMENT_METHODS)
        )

        attempt_number = customer_attempt.get(cid, 0) + 1

        success = _decide_outcome(
            rng,
            cust["archetype"],
            int(cust["success_count"]),
            int(cust["fail_count"]),
            attempt_number,
        )

        # Compute features at the moment of the transaction (pre-outcome).
        total = int(cust["success_count"]) + int(cust["fail_count"])
        customer_success_rate = (
            int(cust["success_count"]) / total if total > 0 else 0.0
        )
        last_success = cust["last_success_ts"]
        hours_since_last_success = (
            (created_at - last_success).total_seconds() / 3600
            if pd.notna(last_success)
            else -1.0  # sentinel: no prior success
        )

        row = {
            "transaction_id": f"txn_{txn_idx:06d}",
            "customer_id": cid,
            "amount": round(amount, 2),
            "currency": "INR",
            "payment_method": method,
            "status": "SUCCESS" if success else "FAILED",
            "attempt_number": attempt_number,
            "customer_success_rate": round(customer_success_rate, 4),
            "customer_previous_failures": int(cust["fail_count"]),
            "customer_previous_recoveries": int(cust["previous_recoveries"]),
            "hours_since_last_success": round(hours_since_last_success, 2),
            "subscription_flag": int(is_sub),
            "created_at": created_at.isoformat(),
            "created_hour": created_at.hour,
            "created_day": created_at.strftime("%A"),
        }

        if not success:
            reason = _assign_failure_reason(rng, cust["archetype"], attempt_number)
            pattern = _assign_failure_pattern(reason, amount, attempt_number)
            row["failure_reason"] = reason
            row["failure_source"] = FAILURE_SOURCES[reason]
            row["failure_pattern"] = pattern
        else:
            row["failure_reason"] = None
            row["failure_source"] = None
            row["failure_pattern"] = None

        rows.append(row)

        # Update running customer state.
        if success:
            customers.at[ci, "success_count"] += 1
            customers.at[ci, "total_paid"] += amount
            customers.at[ci, "last_success_ts"] = created_at
            customer_attempt[cid] = 0  # reset attempt counter
        else:
            customers.at[ci, "fail_count"] += 1
            customer_attempt[cid] = attempt_number

        txn_idx += 1

    transactions = pd.DataFrame(rows)
    # Sort chronologically for time-aware splitting.
    transactions = transactions.sort_values("created_at").reset_index(drop=True)

    return transactions, customers


# ---------------------------------------------------------------------------
# Action-conditional outcome generation
# ---------------------------------------------------------------------------


def generate_action_outcomes(
    rng: np.random.Generator,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every FAILED transaction, generate a recovery outcome for each
    candidate action.  This is the ML label table.

    The recovery probability is derived from:
      1. The failure pattern (base probability per action)
      2. Customer success rate (history bonus / penalty)
      3. Attempt number (diminishing returns)
      4. Amount (high-value gets slight penalty for ESCALATE scenarios)
      5. Random noise

    No feature from the *outcome* leaks into the *transaction* features.
    """
    failed = transactions[transactions["status"] == "FAILED"].copy()
    outcome_rows: list[dict] = []

    for _, txn in failed.iterrows():
        pattern = txn["failure_pattern"]
        base_probs = PATTERN_ACTION_PROBS[pattern]

        for action in RECOVERY_ACTIONS:
            p = base_probs[action]

            # Customer-history modifier (±10 pp).
            csr = txn["customer_success_rate"]
            p += (csr - 0.5) * 0.20

            # Attempt-number decay (repeated failures → harder to recover).
            attempt = txn["attempt_number"]
            if attempt > 1:
                p -= 0.06 * min(attempt - 1, 4)

            # High-value penalty for ESCALATE (escalation is expensive).
            if action == "ESCALATE" and txn["amount"] > 20_000:
                p += 0.08  # but higher amounts justify escalation

            # Low-value boost for STOP/do-nothing proxy.
            if txn["amount"] < 300:
                p -= 0.15  # recovery cost exceeds value

            # Random noise ±5 pp.
            p += rng.normal(0, 0.05)

            p = max(0.01, min(p, 0.99))  # clamp

            # Bernoulli draw — did this action actually recover the payment?
            recovered = bool(rng.random() < p)

            outcome_rows.append(
                {
                    "transaction_id": txn["transaction_id"],
                    "action_type": action,
                    "recovery_probability": round(p, 4),
                    "recovered": int(recovered),
                    "amount": txn["amount"],
                    "failure_pattern": pattern,
                }
            )

    return pd.DataFrame(outcome_rows)


# ---------------------------------------------------------------------------
# Train / validation / test split (time-aware)
# ---------------------------------------------------------------------------


def split_for_ml(
    transactions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge transaction features with action outcomes and split by time.

    The merged rows contain one row per (transaction, action) pair — only
    for failed transactions.  The split is:
        70 % train  |  15 % validation  |  15 % test
    ordered by `created_at` so we simulate predicting future from past.

    The features available are everything in the transaction row EXCEPT
    'status' (which is always FAILED in this subset) and 'failure_pattern'
    (which encodes the target somewhat — we keep it as a sanity-check
    column but it should be dropped before model training).
    """
    # Left-join outcomes onto the failed-transaction features.
    failed_features = transactions[transactions["status"] == "FAILED"].drop(
        columns=["status"]
    )
    merged = outcomes.merge(
        failed_features,
        on="transaction_id",
        how="left",
        suffixes=("", "_txn"),
    )
    # Drop the duplicate amount column from outcomes
    if "amount_txn" in merged.columns:
        merged = merged.drop(columns=["amount_txn"])

    # Sort by time for time-aware split.
    merged = merged.sort_values("created_at").reset_index(drop=True)

    n = len(merged)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = merged.iloc[:train_end].copy()
    val = merged.iloc[train_end:val_end].copy()
    test = merged.iloc[val_end:].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "generated"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic payment dataset for Revenue Recovery Autopilot."
    )
    parser.add_argument(
        "--records",
        type=int,
        default=5_000,
        help="Number of transaction records to generate (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 3, 31, tzinfo=timezone.utc)

    print(f"Generating {args.records} transactions (seed={args.seed})…")

    transactions, customers = generate_transactions(
        rng, args.records, start_date, end_date
    )
    print(f"  Transactions generated: {len(transactions)}")
    print(
        f"    Success: {(transactions['status'] == 'SUCCESS').sum()}  "
        f"Failed: {(transactions['status'] == 'FAILED').sum()}"
    )

    outcomes = generate_action_outcomes(rng, transactions)
    print(f"  Action-outcome rows: {len(outcomes)}")

    train, val, test = split_for_ml(transactions, outcomes)
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    # Write output files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False)
    outcomes.to_csv(OUTPUT_DIR / "recovery_action_outcomes.csv", index=False)
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    # Metadata for reproducibility.
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_records": args.records,
        "n_customers": transactions["customer_id"].nunique(),
        "n_failed": int((transactions["status"] == "FAILED").sum()),
        "n_success": int((transactions["status"] == "SUCCESS").sum()),
        "n_outcome_rows": len(outcomes),
        "train_size": len(train),
        "validation_size": len(val),
        "test_size": len(test),
        "output_dir": str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Files written to {OUTPUT_DIR}/")
    print("  transactions.csv")
    print("  recovery_action_outcomes.csv")
    print("  train.csv")
    print("  validation.csv")
    print("  test.csv")
    print("  dataset_metadata.json")


if __name__ == "__main__":
    main()
