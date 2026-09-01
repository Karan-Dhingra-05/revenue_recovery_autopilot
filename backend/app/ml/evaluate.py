"""
Evaluation module for the recovery-probability models.

Reports on a held-out set (test or validation):
  - ROC-AUC
  - PR-AUC
  - Log loss
  - Brier score
  - Confusion matrix at a sensible threshold
  - Calibration analysis (mean predicted vs. actual in bins)

Usage
-----
    cd backend
    python -m app.ml.evaluate            # evaluates on test set
    python -m app.ml.evaluate --split validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from app.ml.features import get_X_y
from app.ml.train import MODEL_DIR, _load_split, load_model

# ── Helpers ──────────────────────────────────────────────────────────────────

ACTIONS = ["PAYMENT_LINK", "REMINDER", "DELAYED_RETRY", "ESCALATE"]


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def evaluate_model(
    name: str,
    pipe,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict:
    """Compute all evaluation metrics and print a report."""
    _section(f"{name} — {split_name} set")

    y_proba = pipe.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    ll = log_loss(y, y_proba)
    brier = brier_score_loss(y, y_proba)

    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    print(f"  Log loss:    {ll:.4f}")
    print(f"  Brier score: {brier:.4f}")

    # Confusion matrix at threshold = 0.5 (standard) and at the
    # threshold that maximises F1 (more practical).
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion matrix (threshold={threshold}):")
    print(f"    TN={tn:>5}  FP={fp:>5}")
    print(f"    FN={fn:>5}  TP={tp:>5}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"    Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    # Calibration analysis — 10 bins.
    print(f"\n  Calibration (10 bins):")
    print(f"  {'Bin':>12} {'Mean pred':>10} {'Actual rate':>12} {'Count':>6}")
    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            y, y_proba, n_bins=10, strategy="uniform"
        )
        bin_edges = np.linspace(0, 1, 11)
        for i in range(len(fraction_of_positives)):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_proba >= lo) & (y_proba < hi)
            count = mask.sum()
            print(
                f"  [{lo:.1f}–{hi:.1f}] "
                f"{mean_predicted[i]:>10.4f} "
                f"{fraction_of_positives[i]:>12.4f} "
                f"{count:>6}"
            )
    except ValueError:
        print("  (Insufficient data for calibration bins)")

    return {
        "model": name,
        "split": split_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": ll,
        "brier_score": brier,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recovery models.")
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Which split to evaluate on (default: test)",
    )
    args = parser.parse_args()

    print(f"Evaluating on {args.split} set…")
    df = _load_split(args.split)
    X, y = get_X_y(df)
    print(f"  Rows: {len(X)}  Positive rate: {y.mean():.4f}")

    results = []

    for name in ["logistic_regression", "lightgbm"]:
        try:
            pipe = load_model(name)
        except FileNotFoundError:
            print(f"\n  ⚠ Model '{name}' not found. Train first.")
            continue
        res = evaluate_model(name, pipe, X, y, args.split)
        results.append(res)

    if results:
        _section("Summary")
        print(f"  {'Model':<22} {'ROC-AUC':>8} {'PR-AUC':>8} {'Log-loss':>9} {'Brier':>7}")
        for r in results:
            print(
                f"  {r['model']:<22} "
                f"{r['roc_auc']:>8.4f} "
                f"{r['pr_auc']:>8.4f} "
                f"{r['log_loss']:>9.4f} "
                f"{r['brier_score']:>7.4f}"
            )


if __name__ == "__main__":
    main()
