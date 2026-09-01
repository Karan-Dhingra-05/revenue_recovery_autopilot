#!/usr/bin/env python3
"""
Diagnostic analysis of the ML policy's action selection behaviour.

Investigates why the LightGBM policy selects ESCALATE for ~41% of test
transactions, and whether the +24.2% revenue uplift is robust or an
artefact of synthetic-generator rules.

Usage
-----
    cd backend
    python scripts/diagnose_policy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.features import ALL_FEATURES, LABEL, get_X_y
from app.ml.train import load_model

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "generated"
ACTIONS = ["PAYMENT_LINK", "REMINDER", "DELAYED_RETRY", "ESCALATE"]


def _section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def _subsection(title: str) -> None:
    print(f"\n  ── {title} ──")


def load_test_with_scores(model_name: str = "lightgbm") -> pd.DataFrame:
    """Load the test set and append model predictions."""
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    pipe = load_model(model_name)
    X, _ = get_X_y(test_df)
    test_df["predicted_proba"] = pipe.predict_proba(X)[:, 1]
    test_df["expected_recovery"] = test_df["predicted_proba"] * test_df["amount"]
    return test_df


def assign_ml_selections(scored: pd.DataFrame) -> pd.DataFrame:
    """For each transaction, pick the action with max expected recovery."""
    best_idx = scored.groupby("transaction_id")["expected_recovery"].idxmax()
    selected = scored.loc[best_idx].copy()
    selected["selected_action"] = selected["action_type"]
    return selected


def main() -> None:
    print("Loading test set and scoring…")
    scored = load_test_with_scores()
    selected = assign_ml_selections(scored)
    n_txns = len(selected)

    # Also load the full train set for reference distributions
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    # Also load transactions.csv for archetype info (not in split files)
    txns_full = pd.read_csv(DATA_DIR / "transactions.csv")

    # =====================================================================
    # 1. Action selection distribution by various breakdowns
    # =====================================================================
    _section("1. ACTION SELECTION DISTRIBUTION")

    # 1a. Overall
    _subsection("1a. Overall")
    action_dist = selected["selected_action"].value_counts()
    for a in ACTIONS:
        c = action_dist.get(a, 0)
        print(f"    {a:<16} {c:>5}  ({c/n_txns*100:.1f}%)")

    # 1b. By failure pattern
    _subsection("1b. By failure pattern")
    ct = pd.crosstab(selected["failure_pattern"], selected["selected_action"])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(f"    {'Pattern':<20}", end="")
    for a in ACTIONS:
        print(f" {a:>14}", end="")
    print(f" {'Total':>8}")
    for pattern in sorted(ct.index):
        print(f"    {pattern:<20}", end="")
        for a in ACTIONS:
            count = ct.at[pattern, a] if a in ct.columns else 0
            pct = ct_pct.at[pattern, a] if a in ct_pct.columns else 0
            print(f" {count:>5}({pct:>5.1f}%)", end="")
        print(f" {ct.loc[pattern].sum():>8}")

    # 1c. By amount bucket
    _subsection("1c. By amount bucket")
    bins = [0, 300, 1000, 5000, 20000, float("inf")]
    labels_b = ["<300", "300-1k", "1k-5k", "5k-20k", ">20k"]
    selected["amount_bucket"] = pd.cut(selected["amount"], bins=bins, labels=labels_b)
    ct_amt = pd.crosstab(selected["amount_bucket"], selected["selected_action"])
    ct_amt_pct = ct_amt.div(ct_amt.sum(axis=1), axis=0) * 100
    print(f"    {'Bucket':<12}", end="")
    for a in ACTIONS:
        print(f" {a:>14}", end="")
    print(f" {'Total':>8}")
    for bucket in labels_b:
        if bucket not in ct_amt.index:
            continue
        print(f"    {bucket:<12}", end="")
        for a in ACTIONS:
            count = ct_amt.at[bucket, a] if a in ct_amt.columns else 0
            pct = ct_amt_pct.at[bucket, a] if a in ct_amt_pct.columns else 0
            print(f" {count:>5}({pct:>5.1f}%)", end="")
        print(f" {ct_amt.loc[bucket].sum():>8}")

    # 1d. By customer success rate (proxy for archetype/reliability)
    _subsection("1d. By customer reliability (success rate bucket)")
    bins_sr = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels_sr = ["0-30%", "30-50%", "50-70%", "70-90%", "90-100%"]
    selected["csr_bucket"] = pd.cut(
        selected["customer_success_rate"], bins=bins_sr, labels=labels_sr, right=False
    )
    ct_csr = pd.crosstab(selected["csr_bucket"], selected["selected_action"])
    ct_csr_pct = ct_csr.div(ct_csr.sum(axis=1), axis=0) * 100
    print(f"    {'CSR Bucket':<12}", end="")
    for a in ACTIONS:
        print(f" {a:>14}", end="")
    print(f" {'Total':>8}")
    for bucket in labels_sr:
        if bucket not in ct_csr.index:
            continue
        print(f"    {bucket:<12}", end="")
        for a in ACTIONS:
            count = ct_csr.at[bucket, a] if a in ct_csr.columns else 0
            pct = ct_csr_pct.at[bucket, a] if a in ct_csr_pct.columns else 0
            print(f" {count:>5}({pct:>5.1f}%)", end="")
        print(f" {ct_csr.loc[bucket].sum():>8}")

    # 1e. By attempt number
    _subsection("1e. By attempt number")
    ct_att = pd.crosstab(selected["attempt_number"], selected["selected_action"])
    ct_att_pct = ct_att.div(ct_att.sum(axis=1), axis=0) * 100
    print(f"    {'Attempt':<10}", end="")
    for a in ACTIONS:
        print(f" {a:>14}", end="")
    print(f" {'Total':>8}")
    for attempt in sorted(ct_att.index):
        print(f"    {attempt:<10}", end="")
        for a in ACTIONS:
            count = ct_att.at[attempt, a] if a in ct_att.columns else 0
            pct = ct_att_pct.at[attempt, a] if a in ct_att_pct.columns else 0
            print(f" {count:>5}({pct:>5.1f}%)", end="")
        print(f" {ct_att.loc[attempt].sum():>8}")

    # =====================================================================
    # 2. Per-action statistics for selected transactions
    # =====================================================================
    _section("2. PER-ACTION STATISTICS (ML-SELECTED TRANSACTIONS)")
    print(f"    {'Action':<16} {'Avg Amount':>12} {'Avg P(rec)':>11} "
          f"{'Avg E[rec]':>11} {'Actual RR':>10} {'Count':>6}")
    for a in ACTIONS:
        subset = selected[selected["selected_action"] == a]
        if len(subset) == 0:
            print(f"    {a:<16} {'(no selections)':>50}")
            continue
        avg_amt = subset["amount"].mean()
        avg_p = subset["predicted_proba"].mean()
        avg_er = subset["expected_recovery"].mean()
        actual_rr = subset["recovered"].mean()
        print(f"    {a:<16} ₹{avg_amt:>10,.2f}  {avg_p:>10.4f}  "
              f"₹{avg_er:>9,.2f}  {actual_rr:>9.2%}  {len(subset):>5}")

    # =====================================================================
    # 3. Predicted probability vs realized recovery rate by action
    # =====================================================================
    _section("3. PREDICTED vs REALIZED (ALL TEST ROWS, NOT JUST SELECTED)")
    print(f"    {'Action':<16} {'Avg P(pred)':>12} {'Actual RR':>10} "
          f"{'Delta':>8} {'Count':>6}")
    for a in ACTIONS:
        subset = scored[scored["action_type"] == a]
        avg_p = subset["predicted_proba"].mean()
        actual_rr = subset["recovered"].mean()
        delta = avg_p - actual_rr
        print(f"    {a:<16} {avg_p:>12.4f} {actual_rr:>10.4f} "
              f"{delta:>+8.4f} {len(subset):>6}")

    # =====================================================================
    # 4. Why is ESCALATE selected? Causal analysis
    # =====================================================================
    _section("4. ESCALATE SELECTION ANALYSIS")

    esc_selected = selected[selected["selected_action"] == "ESCALATE"]
    non_esc = selected[selected["selected_action"] != "ESCALATE"]

    _subsection("4a. Amount comparison: ESCALATE vs others")
    print(f"    ESCALATE selected:     avg amount = ₹{esc_selected['amount'].mean():,.2f}  "
          f"median = ₹{esc_selected['amount'].median():,.2f}")
    print(f"    Other actions:         avg amount = ₹{non_esc['amount'].mean():,.2f}  "
          f"median = ₹{non_esc['amount'].median():,.2f}")
    print(f"    Ratio (ESCALATE/other): {esc_selected['amount'].mean() / non_esc['amount'].mean():.2f}x")

    _subsection("4b. Expected recovery comparison across actions for ESCALATE-selected txns")
    # For transactions where ESCALATE was selected, show what all four
    # action scores looked like
    esc_txn_ids = set(esc_selected["transaction_id"])
    esc_all_scores = scored[scored["transaction_id"].isin(esc_txn_ids)]
    print(f"    {'Action':<16} {'Avg P(rec)':>11} {'Avg E[rec]':>12}")
    for a in ACTIONS:
        sub = esc_all_scores[esc_all_scores["action_type"] == a]
        print(f"    {a:<16} {sub['predicted_proba'].mean():>11.4f} "
              f"₹{sub['expected_recovery'].mean():>10,.2f}")

    _subsection("4c. Failure pattern distribution of ESCALATE selections")
    fp_dist = esc_selected["failure_pattern"].value_counts()
    for fp, cnt in fp_dist.items():
        print(f"    {fp:<20} {cnt:>5}  ({cnt/len(esc_selected)*100:.1f}%)")

    _subsection("4d. Generator rule check: ESCALATE high-value bonus")
    # The generator adds +0.08 to ESCALATE probability when amount > 20,000
    high_val = selected[selected["amount"] > 20_000]
    if len(high_val) > 0:
        hv_esc = (high_val["selected_action"] == "ESCALATE").sum()
        print(f"    Txns with amount > ₹20,000: {len(high_val)}")
        print(f"    Of those, ESCALATE selected: {hv_esc} ({hv_esc/len(high_val)*100:.1f}%)")
    else:
        print(f"    No transactions with amount > ₹20,000 in test set")

    _subsection("4e. D_REPEATED_FAIL dominance check")
    d_repeat = selected[selected["failure_pattern"] == "D_REPEATED_FAIL"]
    d_esc = (d_repeat["selected_action"] == "ESCALATE").sum()
    print(f"    D_REPEATED_FAIL txns: {len(d_repeat)}")
    print(f"    Of those, ESCALATE selected: {d_esc} ({d_esc/len(d_repeat)*100:.1f}%)" if len(d_repeat) > 0 else "    None")
    print(f"    D_REPEATED_FAIL share of all ESCALATE selections: "
          f"{d_esc}/{len(esc_selected)} ({d_esc/len(esc_selected)*100:.1f}%)" if len(esc_selected) > 0 else "")

    # =====================================================================
    # 5 & 6. Feature importance
    # =====================================================================
    _section("5 & 6. FEATURE IMPORTANCE (LightGBM)")

    pipe = load_model("lightgbm")
    preprocessor = pipe.named_steps["preprocessor"]
    lgbm_model = pipe.named_steps["classifier"]

    # Get feature names from the preprocessor
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out())
    num_names = list(preprocessor.named_transformers_["num"].get_feature_names_out())
    all_names = cat_names + num_names

    _subsection("5a. Built-in (split) feature importance — top 20")
    importances = lgbm_model.feature_importances_
    imp_df = pd.DataFrame({"feature": all_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    print(f"    {'Rank':<6} {'Feature':<50} {'Importance':>10}")
    for i, row in imp_df.head(20).iterrows():
        print(f"    {i+1:<6} {row['feature']:<50} {row['importance']:>10}")

    _subsection("5b. Feature importance by ORIGINAL feature group")
    # Aggregate OHE columns back to their original feature
    group_imp = {}
    for feat, imp in zip(all_names, importances):
        # OHE features look like "payment_method_UPI" etc.
        original = feat
        for cat_feat in ["action_type", "payment_method", "failure_reason",
                         "failure_source", "failure_pattern", "created_day"]:
            if feat.startswith(cat_feat + "_"):
                original = cat_feat
                break
        group_imp[original] = group_imp.get(original, 0) + imp

    group_df = pd.DataFrame(
        [{"feature_group": k, "total_importance": v} for k, v in group_imp.items()]
    ).sort_values("total_importance", ascending=False).reset_index(drop=True)

    total_imp = group_df["total_importance"].sum()
    print(f"    {'Rank':<6} {'Feature Group':<35} {'Importance':>10} {'Share':>8}")
    for i, row in group_df.iterrows():
        share = row["total_importance"] / total_imp * 100
        print(f"    {i+1:<6} {row['feature_group']:<35} {row['total_importance']:>10} {share:>7.1f}%")

    _subsection("6. Permutation importance (test set, 10 repeats)")
    from sklearn.inspection import permutation_importance
    X_test, y_test = get_X_y(pd.read_csv(DATA_DIR / "test.csv"))
    X_test_t = preprocessor.transform(X_test)
    perm_result = permutation_importance(
        lgbm_model, X_test_t, y_test, n_repeats=10,
        scoring="roc_auc", random_state=42
    )
    perm_df = pd.DataFrame({
        "feature": all_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    print(f"    {'Rank':<6} {'Feature':<50} {'Mean ΔrAUC':>11} {'±Std':>8}")
    for i, row in perm_df.head(15).iterrows():
        print(f"    {i+1:<6} {row['feature']:<50} {row['importance_mean']:>+11.4f} "
              f"±{row['importance_std']:>.4f}")

    # Aggregate permutation importance by group
    _subsection("6b. Permutation importance by feature group")
    perm_group = {}
    for feat, imp in zip(all_names, perm_result.importances_mean):
        original = feat
        for cat_feat in ["action_type", "payment_method", "failure_reason",
                         "failure_source", "failure_pattern", "created_day"]:
            if feat.startswith(cat_feat + "_"):
                original = cat_feat
                break
        perm_group[original] = perm_group.get(original, 0) + imp

    perm_group_df = pd.DataFrame(
        [{"feature_group": k, "perm_importance": v} for k, v in perm_group.items()]
    ).sort_values("perm_importance", ascending=False).reset_index(drop=True)

    print(f"    {'Rank':<6} {'Feature Group':<35} {'Perm Importance':>16}")
    for i, row in perm_group_df.iterrows():
        print(f"    {i+1:<6} {row['feature_group']:<35} {row['perm_importance']:>+16.4f}")

    # =====================================================================
    # 7. Dominance check
    # =====================================================================
    _section("7. DOMINANCE CHECK")
    top3_split = group_df.head(3)
    top3_perm = perm_group_df.head(3)
    print(f"    Top 3 by split importance:       {', '.join(top3_split['feature_group'].tolist())}")
    print(f"    Top 3 by permutation importance: {', '.join(top3_perm['feature_group'].tolist())}")

    total_split = group_df["total_importance"].sum()
    top1_share = group_df.iloc[0]["total_importance"] / total_split * 100
    print(f"\n    Top feature group ({group_df.iloc[0]['feature_group']}) "
          f"accounts for {top1_share:.1f}% of total split importance.")
    if top1_share > 40:
        print(f"    ⚠ WARNING: Single feature group dominates (>{40}%).")
    else:
        print(f"    ✓ No single feature group dominates (threshold: 40%).")

    # =====================================================================
    # 8. Target leakage verification
    # =====================================================================
    _section("8. TARGET LEAKAGE CHECK")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    txn_raw = pd.read_csv(DATA_DIR / "transactions.csv")

    # Check 1: 'recovered' not in feature list
    if "recovered" in ALL_FEATURES:
        print("    ✗ LEAK: 'recovered' in feature list!")
    else:
        print("    ✓ 'recovered' NOT in feature list")

    # Check 2: 'recovery_probability' not in feature list
    if "recovery_probability" in ALL_FEATURES:
        print("    ✗ LEAK: 'recovery_probability' in feature list!")
    else:
        print("    ✓ 'recovery_probability' NOT in feature list")

    # Check 3: 'status' not in feature list
    if "status" in ALL_FEATURES:
        print("    ✗ LEAK: 'status' in feature list!")
    else:
        print("    ✓ 'status' NOT in feature list")

    # Check 4: Correlation between features and target
    print("\n    Feature-target correlations (top 5 numerical features):")
    for feat in ["amount", "attempt_number", "customer_success_rate",
                 "customer_previous_failures", "hours_since_last_success"]:
        corr = test_raw[feat].corr(test_raw["recovered"])
        flag = " ⚠" if abs(corr) > 0.5 else ""
        print(f"      {feat:<35} r={corr:>+.4f}{flag}")

    # Check 5: failure_pattern encodes target information via generator rules
    print("\n    failure_pattern ↔ recovered correlation (point-biserial by pattern):")
    for fp in sorted(test_raw["failure_pattern"].unique()):
        sub = test_raw[test_raw["failure_pattern"] == fp]
        rr = sub["recovered"].mean()
        print(f"      {fp:<20} actual RR = {rr:.4f}")

    # =====================================================================
    # 9. Uplift robustness assessment
    # =====================================================================
    _section("9. UPLIFT ROBUSTNESS ASSESSMENT")

    # Compare ML policy action-conditioned recovery rates vs marginal rates
    print("    ML policy selects actions that match the generator's structure:")
    print()
    print(f"    {'Pattern':<20} {'ML Action':>14} {'Pred P':>8} {'Actual RR':>10} {'N':>5}")

    for fp in sorted(selected["failure_pattern"].unique()):
        sub = selected[selected["failure_pattern"] == fp]
        for a in sub["selected_action"].unique():
            asub = sub[sub["selected_action"] == a]
            if len(asub) < 2:
                continue
            pred = asub["predicted_proba"].mean()
            actual = asub["recovered"].mean()
            print(f"    {fp:<20} {a:>14} {pred:>8.4f} {actual:>10.2%} {len(asub):>5}")

    # Summary: is the uplift real or generator-rule-shaped?
    print("\n    Summary of uplift drivers:")
    b1_rev = selected.copy()
    # Re-compute Baseline 1 revenue for comparison
    b1 = scored[scored["action_type"] == "PAYMENT_LINK"].copy()
    b1_rev_total = b1.loc[b1["recovered"] == 1, "amount"].sum()
    ml_rev_total = selected.loc[selected["recovered"] == 1, "amount"].sum()
    uplift = ml_rev_total - b1_rev_total
    uplift_pct = uplift / b1_rev_total * 100 if b1_rev_total > 0 else 0

    # How much of the ML uplift comes from ESCALATE selections?
    esc_recovered = esc_selected.loc[esc_selected["recovered"] == 1, "amount"].sum()
    # What would those same transactions have recovered under PAYMENT_LINK?
    esc_txn_ids_list = list(esc_txn_ids)
    esc_as_plink = scored[
        (scored["transaction_id"].isin(esc_txn_ids_list)) &
        (scored["action_type"] == "PAYMENT_LINK")
    ]
    esc_plink_recovered = esc_as_plink.loc[esc_as_plink["recovered"] == 1, "amount"].sum()

    print(f"    ML total recovered revenue:          ₹{ml_rev_total:>12,.2f}")
    print(f"    Baseline (PAYMENT_LINK) revenue:     ₹{b1_rev_total:>12,.2f}")
    print(f"    ML uplift:                           ₹{uplift:>12,.2f} ({uplift_pct:+.1f}%)")
    print()
    print(f"    ESCALATE selections recovered:       ₹{esc_recovered:>12,.2f}")
    print(f"    Same txns under PAYMENT_LINK:        ₹{esc_plink_recovered:>12,.2f}")
    print(f"    Revenue gain from ESCALATE switch:   ₹{esc_recovered - esc_plink_recovered:>12,.2f}")
    print(f"    ESCALATE's share of total uplift:    "
          f"{(esc_recovered - esc_plink_recovered) / uplift * 100:.1f}%" if uplift > 0 else "N/A")

    # Same for DELAYED_RETRY
    dr_selected = selected[selected["selected_action"] == "DELAYED_RETRY"]
    if len(dr_selected) > 0:
        dr_txn_ids = set(dr_selected["transaction_id"])
        dr_recovered = dr_selected.loc[dr_selected["recovered"] == 1, "amount"].sum()
        dr_as_plink = scored[
            (scored["transaction_id"].isin(dr_txn_ids)) &
            (scored["action_type"] == "PAYMENT_LINK")
        ]
        dr_plink_recovered = dr_as_plink.loc[dr_as_plink["recovered"] == 1, "amount"].sum()
        print(f"\n    DELAYED_RETRY selections recovered:  ₹{dr_recovered:>12,.2f}")
        print(f"    Same txns under PAYMENT_LINK:        ₹{dr_plink_recovered:>12,.2f}")
        print(f"    Revenue gain from DR switch:         ₹{dr_recovered - dr_plink_recovered:>12,.2f}")

    print()
    print("  ── Final Assessment ──")
    print()
    print("  The analysis above decomposes the uplift by action and shows how")
    print("  much of the gain is driven by each action switch relative to the")
    print("  PAYMENT_LINK baseline. Review the ESCALATE share of uplift and")
    print("  whether it reflects genuine generator heterogeneity or a synthetic")
    print("  rule artefact (e.g. the +0.08 high-value ESCALATE bonus).")


if __name__ == "__main__":
    main()
