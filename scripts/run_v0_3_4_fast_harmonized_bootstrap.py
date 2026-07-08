#!/usr/bin/env python3
"""
Fast bootstrap CI for harmonized counterfactual predictions.

This script is intended to replace the slow bootstrap loop inside
run_v0_3_3_harmonized_counterfactual_experiment.py. It reads:

  results/counterfactual_v5_6_harmonized/counterfactual_predictions_harmonized.csv

and writes:

  results/counterfactual_v5_6_harmonized/counterfactual_bootstrap_ci_harmonized.csv

It does not regenerate the counterfactual dataset or retrain models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


LABELS = ["ALLOW", "HUMAN_REVIEW", "REJECT"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}


def normalize_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if s in {"ALLOW", "ALLOWED", "APPROVE", "APPROVED"}:
        return "ALLOW"
    if s in {"HUMAN_REVIEW", "REVIEW", "MANUAL_REVIEW", "ESCALATE", "ESCALATED"}:
        return "HUMAN_REVIEW"
    if s in {"REJECT", "REJECTED", "DENY", "DENIED", "BLOCK", "BLOCKED"}:
        return "REJECT"
    return None


def encode_labels(values):
    norm = [normalize_label(v) for v in values]
    return np.array([LABEL_TO_INT.get(v, -1) for v in norm], dtype=np.int16)


def macro_f1_from_int(y_true, y_pred):
    f1s = []
    for label_id in range(len(LABELS)):
        tp = np.sum((y_true == label_id) & (y_pred == label_id))
        fp = np.sum((y_true != label_id) & (y_pred == label_id))
        fn = np.sum((y_true == label_id) & (y_pred != label_id))
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else (2 * tp / denom))
    return float(np.mean(f1s))


def weighted_f1_from_int(y_true, y_pred):
    total = len(y_true)
    if total == 0:
        return 0.0
    out = 0.0
    for label_id in range(len(LABELS)):
        support = np.sum(y_true == label_id)
        if support == 0:
            continue
        tp = np.sum((y_true == label_id) & (y_pred == label_id))
        fp = np.sum((y_true != label_id) & (y_pred == label_id))
        fn = np.sum((y_true == label_id) & (y_pred != label_id))
        denom = (2 * tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2 * tp / denom)
        out += (support / total) * f1
    return float(out)


def loss_vector(y_true, y_pred):
    losses = np.ones(len(y_true), dtype=float)
    losses[y_true == y_pred] = 0.0

    allow = LABEL_TO_INT["ALLOW"]
    review = LABEL_TO_INT["HUMAN_REVIEW"]
    reject = LABEL_TO_INT["REJECT"]

    losses[(y_pred == allow) & (y_true == reject)] = 5.0
    losses[(y_pred == allow) & (y_true == review)] = 2.0
    losses[(y_pred == reject) & (y_true == allow)] = 1.0
    losses[(y_pred == reject) & (y_true == review)] = 1.0
    losses[(y_pred == review) & (y_true == allow)] = 0.5
    losses[(y_pred == review) & (y_true == reject)] = 1.5
    return losses


def metric_values(y_true, y_pred, critical_gold):
    allow = LABEL_TO_INT["ALLOW"]
    review = LABEL_TO_INT["HUMAN_REVIEW"]
    reject = LABEL_TO_INT["REJECT"]

    unsafe_allow = (y_pred == allow) & (y_true != allow)
    critical_unsafe_allow = (y_pred == allow) & critical_gold
    over_reject = (y_pred == reject) & (y_true == allow)
    human_review = y_pred == review

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1_from_int(y_true, y_pred),
        "weighted_f1": weighted_f1_from_int(y_true, y_pred),
        "unsafe_allow_rate": float(np.mean(unsafe_allow)),
        "critical_unsafe_allow_rate": float(np.mean(critical_unsafe_allow)),
        "over_reject_rate": float(np.mean(over_reject)),
        "human_review_load": float(np.mean(human_review)),
        "risk_weighted_safety_loss_proxy": float(np.mean(loss_vector(y_true, y_pred))),
    }


def bootstrap_group(group, n_bootstrap, seed):
    y_true = encode_labels(group["actual"])
    y_pred = encode_labels(group["predicted"])
    critical_gold = group["critical_gold"].astype(str).str.lower().isin(["true", "1"]).to_numpy()

    mask = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    critical_gold = critical_gold[mask]

    dataset = group["dataset"].iloc[0]
    model = group["model"].iloc[0]
    n = len(y_true)

    if n == 0:
        return []

    point = metric_values(y_true, y_pred, critical_gold)
    metrics = list(point.keys())

    rng = np.random.default_rng(seed)
    boot_values = {m: np.empty(n_bootstrap, dtype=float) for m in metrics}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        vals = metric_values(y_true[idx], y_pred[idx], critical_gold[idx])
        for m in metrics:
            boot_values[m][b] = vals[m]

    rows = []
    for m in metrics:
        rows.append({
            "dataset": dataset,
            "model": model,
            "metric": m,
            "point_estimate": point[m],
            "ci_lower_95": float(np.quantile(boot_values[m], 0.025)),
            "ci_upper_95": float(np.quantile(boot_values[m], 0.975)),
            "n_bootstrap": n_bootstrap,
            "n": n,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    pred_path = repo / "results" / "counterfactual_v5_6_harmonized" / "counterfactual_predictions_harmonized.csv"
    out_path = repo / "results" / "counterfactual_v5_6_harmonized" / "counterfactual_bootstrap_ci_harmonized.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    df = pd.read_csv(pred_path)
    required = {"dataset", "model", "actual", "predicted", "critical_gold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(df.columns)}")

    rows = []
    groups = list(df.groupby(["dataset", "model"], sort=False))
    for i, ((dataset, model), group) in enumerate(groups, start=1):
        print(f"[{i}/{len(groups)}] Bootstrapping {dataset} / {model} ...", flush=True)
        rows.extend(bootstrap_group(group, args.n_bootstrap, args.seed + i))

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Done. Fast bootstrap CI written to: {out_path}")


if __name__ == "__main__":
    main()
