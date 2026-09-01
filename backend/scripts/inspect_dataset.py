#!/usr/bin/env python3
"""
Data-quality / inspection script for the synthetic dataset.

Loads the generated CSVs and prints a comprehensive statistical report
covering distributions, recovery rates, leakage checks, and split sizes.

Usage
-----
    cd backend
    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "generated"


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist. Run generate_dataset.py first.")
        sys.exit(1)

    txns = pd.read_csv(DATA_DIR / "transactions.csv")
    outcomes = pd.read_csv(DATA_DIR / "recovery_action_outcomes.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # ── 1. Total rows ────────────────────────────────────────────────────
    _section("1. Row Counts")
    print(f"  transactions.csv             {len(txns):>7,}")
    print(f"  recovery_action_outcomes.csv {len(outcomes):>7,}")
    print(f"  train.csv                    {len(train):>7,}")
    print(f"  validation.csv               {len(val):>7,}")
    print(f"  test.csv                     {len(test):>7,}")
    print(f"  train+val+test               {len(train) + len(val) + len(test):>7,}")

    # ── 2. Unique customers ──────────────────────────────────────────────
    _section("2. Unique Customers")
    print(f"  In transactions:  {txns['customer_id'].nunique()}")
    print(f"  In outcomes:      {outcomes.merge(txns[['transaction_id', 'customer_id']], on='transaction_id')['customer_id'].nunique()}")

    # ── 3. Success / failure distribution ────────────────────────────────
    _section("3. Transaction Status Distribution")
    status_counts = txns["status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(txns) * 100
        print(f"  {status:<10} {count:>6,}  ({pct:.1f}%)")

    n_failed = (txns["status"] == "FAILED").sum()
    expected_outcome_rows = n_failed * len(outcomes["action_type"].unique())
    print(f"\n  Failed × actions = {n_failed} × {outcomes['action_type'].nunique()} = {expected_outcome_rows}")
    print(f"  Actual outcome rows: {len(outcomes)}")
    assert len(outcomes) == expected_outcome_rows, "MISMATCH in outcome row count!"

    # ── 4. Failure reason distribution ───────────────────────────────────
    _section("4. Failure Reason Distribution")
    failed = txns[txns["status"] == "FAILED"]
    reason_counts = failed["failure_reason"].value_counts()
    for reason, count in reason_counts.items():
        pct = count / len(failed) * 100
        print(f"  {reason:<25} {count:>5,}  ({pct:.1f}%)")

    # ── 5. Failure pattern distribution ──────────────────────────────────
    _section("5. Failure Pattern Distribution")
    pattern_counts = failed["failure_pattern"].value_counts()
    for pattern, count in pattern_counts.items():
        pct = count / len(failed) * 100
        print(f"  {pattern:<20} {count:>5,}  ({pct:.1f}%)")

    # ── 6. Action distribution ───────────────────────────────────────────
    _section("6. Action Type Distribution (in outcomes)")
    action_counts = outcomes["action_type"].value_counts()
    for action, count in action_counts.items():
        pct = count / len(outcomes) * 100
        print(f"  {action:<16} {count:>6,}  ({pct:.1f}%)")

    # ── 7. Overall recovery rate ─────────────────────────────────────────
    _section("7. Overall Recovery Rate")
    overall_rr = outcomes["recovered"].mean()
    print(f"  Mean recovered (across all actions): {overall_rr:.4f}  ({overall_rr*100:.1f}%)")

    # ── 8. Amount distribution ───────────────────────────────────────────
    _section("8. Amount Distribution (all transactions)")
    desc = txns["amount"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    for stat, val_stat in desc.items():
        label = stat.replace("%", "th pctile")
        print(f"  {label:<15} ₹{val_stat:>12,.2f}")

    # ── 9. Recovery rate by action ───────────────────────────────────────
    _section("9. Recovery Rate by Action")
    rr_by_action = outcomes.groupby("action_type")["recovered"].mean().sort_values(ascending=False)
    for action, rr in rr_by_action.items():
        print(f"  {action:<16} {rr:.4f}  ({rr*100:.1f}%)")

    # ── 10. Recovery rate by failure reason ──────────────────────────────
    _section("10. Recovery Rate by Failure Reason")
    # Join failure_reason from transactions onto outcomes.
    out_with_reason = outcomes.merge(
        txns[["transaction_id", "failure_reason"]], on="transaction_id"
    )
    rr_by_reason = (
        out_with_reason.groupby("failure_reason")["recovered"]
        .mean()
        .sort_values(ascending=False)
    )
    for reason, rr in rr_by_reason.items():
        print(f"  {reason:<25} {rr:.4f}  ({rr*100:.1f}%)")

    # ── 11. Recovery rate by failure pattern ─────────────────────────────
    _section("11. Recovery Rate by Failure Pattern")
    rr_by_pattern = outcomes.groupby("failure_pattern")["recovered"].mean().sort_values(ascending=False)
    for pattern, rr in rr_by_pattern.items():
        print(f"  {pattern:<20} {rr:.4f}  ({rr*100:.1f}%)")

    # ── 12. Train / validation / test sizes ──────────────────────────────
    _section("12. Train / Validation / Test Split")
    total_ml = len(train) + len(val) + len(test)
    for name, df in [("train", train), ("validation", val), ("test", test)]:
        pct = len(df) / total_ml * 100 if total_ml else 0
        print(f"  {name:<12} {len(df):>6,}  ({pct:.1f}%)")

    # Verify time ordering (no future leakage into training).
    if "created_at" in train.columns:
        train_max = train["created_at"].max()
        val_min = val["created_at"].min()
        test_min = test["created_at"].min()
        print(f"\n  Train max date:  {train_max}")
        print(f"  Val   min date:  {val_min}")
        print(f"  Test  min date:  {test_min}")
        if train_max <= val_min and val_min <= test_min:
            print("  ✓ Time ordering: CORRECT (no temporal leakage)")
        else:
            print("  ⚠ WARNING: Time ordering violated! Possible temporal leakage.")

    # ── 13. Leakage check ────────────────────────────────────────────────
    _section("13. Target Leakage Check")
    # 'recovered' must NOT appear as a feature in transactions.csv.
    txn_cols = set(txns.columns)
    if "recovered" in txn_cols:
        print("  ✗ LEAKAGE: 'recovered' column found in transactions.csv!")
    else:
        print("  ✓ No 'recovered' column in transactions.csv")

    # 'status' in the ML splits should always be absent (we dropped it).
    for name, df in [("train", train), ("validation", val), ("test", test)]:
        if "status" in df.columns:
            unique_statuses = df["status"].unique()
            print(f"  ⚠ '{name}' contains 'status' column: {unique_statuses}")
        else:
            print(f"  ✓ '{name}' does not contain 'status' column (correct)")

    # ── 14. Impossible values check ──────────────────────────────────────
    _section("14. Impossible Values Check")
    issues = []

    if (txns["amount"] <= 0).any():
        issues.append("Negative or zero amounts found in transactions")
    if outcomes["recovery_probability"].min() < 0 or outcomes["recovery_probability"].max() > 1:
        issues.append("recovery_probability outside [0, 1]")
    if not outcomes["recovered"].isin([0, 1]).all():
        issues.append("'recovered' contains values other than 0/1")
    if txns["attempt_number"].min() < 1:
        issues.append("attempt_number < 1 found")

    if issues:
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("  ✓ No impossible values detected")

    # ── Summary ──────────────────────────────────────────────────────────
    _section("SUMMARY")
    print(f"  Transactions:  {len(txns):,}")
    print(f"  Customers:     {txns['customer_id'].nunique()}")
    print(f"  Failed:        {n_failed:,} ({n_failed/len(txns)*100:.1f}%)")
    print(f"  Outcomes:      {len(outcomes):,}")
    print(f"  Overall RR:    {overall_rr*100:.1f}%")
    print(f"  ML rows:       {total_ml:,}  (train {len(train):,} / val {len(val):,} / test {len(test):,})")
    print()


if __name__ == "__main__":
    main()
