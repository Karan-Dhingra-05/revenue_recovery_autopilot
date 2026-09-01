"""
Training logic for the recovery-probability models.

Two models are trained:
  1. Logistic Regression (baseline)
  2. LightGBM (primary)

Both use the same preprocessor + feature set.  The trained artefacts
(preprocessor + model) are saved as joblib files under `models/`.

Usage
-----
    cd backend
    python -m app.ml.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from app.ml.features import ALL_FEATURES, LABEL, build_preprocessor, get_X_y

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "generated"
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/generate_dataset.py` first."
        )
    return pd.read_csv(path)


# ── Logistic Regression ─────────────────────────────────────────────────────


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """
    Train a Logistic Regression pipeline (preprocessor + model).

    Uses L2 regularisation with C=1.0 and max_iter=1000.
    """
    preprocessor = build_preprocessor()
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    # Report quick validation metrics.
    y_val_proba = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_proba)
    ll = log_loss(y_val, y_val_proba)
    print(f"  [LR] Val ROC-AUC: {auc:.4f}  Log-loss: {ll:.4f}")

    return pipe


# ── LightGBM ────────────────────────────────────────────────────────────────


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, lgb.LGBMClassifier]:
    """
    Train a LightGBM pipeline (preprocessor + model).

    LightGBM uses its own categorical handling internally, but we run
    through the same preprocessor to keep the pipeline consistent.

    Hyperparameters are set conservatively for a 6k-row dataset:
    - 300 boosting rounds, early stopping on val log-loss
    - max_depth=6, learning_rate=0.05
    - colsample_bytree=0.8 to reduce overfitting
    """
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        min_child_samples=20,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_val_proba = model.predict_proba(X_val_t)[:, 1]
    auc = roc_auc_score(y_val, y_val_proba)
    ll = log_loss(y_val, y_val_proba)
    print(f"  [LGBM] Val ROC-AUC: {auc:.4f}  Log-loss: {ll:.4f}")
    print(f"  [LGBM] Best iteration: {model.best_iteration_}")

    # Wrap into a pipeline-like object for consistent predict API.
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    return pipe, model


# ── Save / Load ──────────────────────────────────────────────────────────────


def save_model(pipe: Pipeline, name: str) -> Path:
    """Save a trained pipeline to models/<name>.joblib."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(pipe, path)
    print(f"  Saved: {path}")
    return path


def load_model(name: str) -> Pipeline:
    """Load a trained pipeline from models/<name>.joblib."""
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading data…")
    train_df = _load_split("train")
    val_df = _load_split("validation")

    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)

    print(f"  Train: {len(X_train)} rows  Val: {len(X_val)} rows")
    print(f"  Positive rate (train): {y_train.mean():.4f}")
    print(f"  Positive rate (val):   {y_val.mean():.4f}")
    print()

    # 1. Logistic Regression baseline
    print("Training Logistic Regression…")
    lr_pipe = train_logistic_regression(X_train, y_train, X_val, y_val)
    save_model(lr_pipe, "logistic_regression")
    print()

    # 2. LightGBM primary model
    print("Training LightGBM…")
    lgbm_pipe, lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    save_model(lgbm_pipe, "lightgbm")
    print()

    # Save training metadata.
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "features": ALL_FEATURES,
        "label": LABEL,
        "lgbm_best_iteration": lgbm_model.best_iteration_,
    }
    meta_path = MODEL_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
