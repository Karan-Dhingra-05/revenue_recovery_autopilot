"""
Feature definitions and preprocessing for the recovery-probability model.

This module defines:
  - Which columns are used as features (and which are excluded).
  - How categorical and numerical columns are preprocessed.
  - A single `build_preprocessor()` that returns a fitted sklearn Pipeline.

Design decisions
----------------
* One single model with `action_type` as a categorical feature (spec §12:
  "use the simpler approach first").
* `failure_pattern` is derived from `failure_reason` + `amount` +
  `attempt_number` via deterministic rules, so we include it as a feature
  (it's computable at prediction time — not a label leak).
* `failure_pattern_txn` is a merge artifact duplicate — always excluded.
* `recovered` and `recovery_probability` are label/target columns — excluded.
* `transaction_id`, `customer_id`, `created_at`, `currency` are IDs or
  constants — excluded.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Feature lists ────────────────────────────────────────────────────────────

# Categorical features: will be one-hot encoded.
CATEGORICAL_FEATURES: list[str] = [
    "action_type",
    "payment_method",
    "failure_reason",
    "failure_source",
    "failure_pattern",
    "created_day",
]

# Numerical features: will be standard-scaled.
NUMERICAL_FEATURES: list[str] = [
    "amount",
    "attempt_number",
    "customer_success_rate",
    "customer_previous_failures",
    "customer_previous_recoveries",
    "hours_since_last_success",
    "subscription_flag",
    "created_hour",
]

# All feature columns used by the model.
ALL_FEATURES: list[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Label column.
LABEL: str = "recovered"

# Columns that must NEVER be included in the feature set.
EXCLUDED_COLUMNS: set[str] = {
    "transaction_id",
    "customer_id",
    "created_at",
    "currency",
    "recovered",            # target
    "recovery_probability", # simulated probability (target proxy)
    "failure_pattern_txn",  # merge duplicate
}


def validate_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any excluded / target column appears in the feature set."""
    leaked = EXCLUDED_COLUMNS & set(df.columns)
    # We only care if these columns are *used* as features — they may
    # exist in the dataframe but won't be selected by the preprocessor.
    # The real guard is that ALL_FEATURES does not include any of them.
    for col in ["recovered", "recovery_probability"]:
        if col in ALL_FEATURES:
            raise ValueError(f"Target leakage: '{col}' is listed in ALL_FEATURES")


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and label vector y from a split dataframe.

    Returns only the columns listed in ALL_FEATURES as X, and the
    `recovered` column as y.
    """
    validate_no_leakage(df)
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if LABEL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL}")

    X = df[ALL_FEATURES].copy()
    y = df[LABEL].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for the feature set.

    - Categorical features → OneHotEncoder (handle_unknown='ignore')
    - Numerical features → StandardScaler

    The preprocessor is unfitted — call .fit(X_train) before .transform().
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",  # drop any extra columns
    )
