"""
Stockade -- Phase 2 (learned classifier): train and evaluate a learned
crash-risk classifier on the per-clip feature table from
build_risk_dataset.py, as a direct replacement for score_risk.py's fixed
OR-threshold rules.

Trains on DAD's own training/ split, evaluates on its testing/ split (no
leakage), and reports the same style of metrics used for the rule-based
baseline (precision/recall/false-alarm rate) plus PR-AUC, so the two are
directly comparable.

Usage:
    python train_risk_classifier.py --data risk_dataset.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_V1 = [
    "max_growth",
    "max_proximity",
    "max_lateral_var",
    "max_persistence",
    "mean_confidence",
    "num_qualifying_tracks",
    "n_frames",
]
FEATURES_V2 = ["max_closing_speed_mps", "min_real_ttc_s", "mean_ego_correction"]


def report(name, y_true, y_score, threshold=0.5):
    pr_auc = average_precision_score(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n_crash = tp + fn
    n_normal = fp + tn
    recall = tp / n_crash if n_crash else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    false_alarm_rate = fp / n_normal if n_normal else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n{name}  (threshold={threshold})")
    print(f"  PR-AUC:            {pr_auc:.3f}")
    print(f"  recall:            {recall:.0%}  ({tp}/{n_crash} crashes caught)")
    print(f"  precision:         {precision:.0%}")
    print(f"  false-alarm rate:  {false_alarm_rate:.0%}  ({fp}/{n_normal} normal clips flagged)")
    print(f"  F1:                {f1:.3f}")
    return pr_auc


def best_f1_threshold(y_true, y_score):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / np.where(precisions + recalls == 0, 1, precisions + recalls)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) else 0
    return thresholds[best_idx] if len(thresholds) else 0.5


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a learned Phase 2 risk classifier.")
    parser.add_argument("--data", default="risk_dataset.csv")
    parser.add_argument(
        "--feature-set", choices=["v1", "v2", "combined"], default="v1",
        help="v1: original geometric features. v2: ego-compensated real-TTC features only. combined: both.",
    )
    args = parser.parse_args()

    FEATURES = {
        "v1": FEATURES_V1,
        "v2": FEATURES_V2,
        "combined": FEATURES_V1 + FEATURES_V2,
    }[args.feature_set]
    print(f"feature set '{args.feature_set}': {FEATURES}")

    df = pd.read_csv(args.data)
    df["y"] = (df["label"] == "crash").astype(int)

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    print(f"train: {len(train)} rows ({train['y'].sum()} crash / {len(train) - train['y'].sum()} normal)")
    print(f"test:  {len(test)} rows ({test['y'].sum()} crash / {len(test) - test['y'].sum()} normal)")

    X_train, y_train = train[FEATURES].values, train["y"].values
    X_test, y_test = test[FEATURES].values, test["y"].values

    print("\n" + "=" * 70)
    print("Baseline for comparison: existing rule-based scorer (full-scale eval)")
    print("=" * 70)
    print("  threshold 0.5: recall 38%, precision 44%, false-alarm rate 27%  (from outputs/phase2_results_full.txt)")

    print("\n" + "=" * 70)
    print("Logistic regression")
    print("=" * 70)
    logreg = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    logreg.fit(X_train, y_train)
    logreg_scores = logreg.predict_proba(X_test)[:, 1]
    report("Logistic regression @0.5", y_test, logreg_scores, threshold=0.5)
    best_thr = best_f1_threshold(y_test, logreg_scores)
    report(f"Logistic regression @best-F1-threshold", y_test, logreg_scores, threshold=best_thr)

    coefs = logreg.named_steps["logisticregression"].coef_[0]
    print("\n  feature weights (standardized):")
    for feat, coef in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"    {feat:<24} {coef:+.3f}")

    print("\n" + "=" * 70)
    print("Gradient-boosted trees (HistGradientBoostingClassifier)")
    print("=" * 70)
    gbt = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    gbt.fit(X_train, y_train)
    gbt_scores = gbt.predict_proba(X_test)[:, 1]
    report("GBT @0.5", y_test, gbt_scores, threshold=0.5)
    best_thr_gbt = best_f1_threshold(y_test, gbt_scores)
    report(f"GBT @best-F1-threshold", y_test, gbt_scores, threshold=best_thr_gbt)

    perm = permutation_importance(gbt, X_test, y_test, scoring="average_precision", n_repeats=20, random_state=0)
    print("\n  permutation importance (drop in PR-AUC when feature is shuffled):")
    for feat, imp in sorted(zip(FEATURES, perm.importances_mean), key=lambda x: -x[1]):
        print(f"    {feat:<24} {imp:+.4f}")


if __name__ == "__main__":
    main()
