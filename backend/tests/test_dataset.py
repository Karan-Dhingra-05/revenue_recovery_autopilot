"""
Tests for the synthetic dataset generator.

These tests verify:
  1. Reproducibility — same seed produces identical output.
  2. Row-count invariants — outcome rows = failed × actions.
  3. No target leakage — 'recovered' never appears in transaction features.
  4. No impossible values — amounts > 0, probabilities in [0,1], etc.
  5. Time-aware split — train dates ≤ val dates ≤ test dates.
  6. Feature schema — required columns are present in every output file.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# Import the generator functions directly so we don't have to
# shell out or touch the filesystem.
from scripts.generate_dataset import (
    RECOVERY_ACTIONS,
    generate_action_outcomes,
    generate_transactions,
    split_for_ml,
)


# Use a small dataset for speed.  200 records exercises all code paths
# (multiple customers, multiple patterns, all actions).
SEED = 42
N_RECORDS = 200
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 3, 31, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def generated_data():
    """Generate a small dataset once for all tests in this module."""
    rng = np.random.default_rng(SEED)
    txns, customers = generate_transactions(rng, N_RECORDS, START, END)
    rng2 = np.random.default_rng(SEED + 1)  # separate stream for outcomes
    outcomes = generate_action_outcomes(rng2, txns)
    train, val, test = split_for_ml(txns, outcomes)
    return txns, outcomes, train, val, test


# ----- Reproducibility -----

def test_reproducibility():
    """Same seed → identical transactions."""
    rng1 = np.random.default_rng(SEED)
    txns1, _ = generate_transactions(rng1, N_RECORDS, START, END)

    rng2 = np.random.default_rng(SEED)
    txns2, _ = generate_transactions(rng2, N_RECORDS, START, END)

    pd.testing.assert_frame_equal(txns1, txns2)


# ----- Row counts -----

def test_transaction_count(generated_data):
    txns, outcomes, train, val, test = generated_data
    assert len(txns) == N_RECORDS


def test_outcome_rows_match_failed_times_actions(generated_data):
    txns, outcomes, *_ = generated_data
    n_failed = (txns["status"] == "FAILED").sum()
    n_actions = len(RECOVERY_ACTIONS)
    assert len(outcomes) == n_failed * n_actions


def test_split_rows_sum_to_outcomes(generated_data):
    _, outcomes, train, val, test = generated_data
    assert len(train) + len(val) + len(test) == len(outcomes)


# ----- No target leakage -----

def test_no_recovered_column_in_transactions(generated_data):
    txns, *_ = generated_data
    assert "recovered" not in txns.columns


def test_no_status_column_in_splits(generated_data):
    _, _, train, val, test = generated_data
    for name, df in [("train", train), ("val", val), ("test", test)]:
        assert "status" not in df.columns, f"'status' leaked into {name}"


# ----- No impossible values -----

def test_amounts_are_positive(generated_data):
    txns, *_ = generated_data
    assert (txns["amount"] > 0).all()


def test_recovery_probability_in_range(generated_data):
    _, outcomes, *_ = generated_data
    assert (outcomes["recovery_probability"] >= 0).all()
    assert (outcomes["recovery_probability"] <= 1).all()


def test_recovered_is_binary(generated_data):
    _, outcomes, *_ = generated_data
    assert outcomes["recovered"].isin([0, 1]).all()


def test_attempt_number_positive(generated_data):
    txns, *_ = generated_data
    assert (txns["attempt_number"] >= 1).all()


# ----- Time-aware split -----

def test_time_ordering_of_splits(generated_data):
    _, _, train, val, test = generated_data
    if "created_at" in train.columns:
        assert train["created_at"].max() <= val["created_at"].min()
        assert val["created_at"].max() <= test["created_at"].min()


# ----- Schema completeness -----

REQUIRED_TXN_COLS = {
    "transaction_id",
    "customer_id",
    "amount",
    "currency",
    "payment_method",
    "status",
    "attempt_number",
    "customer_success_rate",
    "customer_previous_failures",
    "customer_previous_recoveries",
    "hours_since_last_success",
    "subscription_flag",
    "created_at",
    "created_hour",
    "created_day",
    "failure_reason",
    "failure_source",
    "failure_pattern",
}

REQUIRED_OUTCOME_COLS = {
    "transaction_id",
    "action_type",
    "recovery_probability",
    "recovered",
    "amount",
    "failure_pattern",
}


def test_transaction_schema(generated_data):
    txns, *_ = generated_data
    missing = REQUIRED_TXN_COLS - set(txns.columns)
    assert not missing, f"Missing transaction columns: {missing}"


def test_outcome_schema(generated_data):
    _, outcomes, *_ = generated_data
    missing = REQUIRED_OUTCOME_COLS - set(outcomes.columns)
    assert not missing, f"Missing outcome columns: {missing}"
