#!/usr/bin/env python3
"""Run corrected PlanSafeBench-EVM benchmark baselines.

This script trains template-disjoint text baselines on the corrected scenario CSV
and evaluates them with security-oriented metrics. The semantic-policy validator
is included as the deterministic computational security model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from plansafebench_evm.semantic_policy import (
    derive_intent_from_dataset_row,
    risk_weighted_loss,
    safe_json_loads,
    validation_decision,
)

LABELS = ["ALLOW", "HUMAN_REVIEW", "REJECT"]
TEXT_COLUMNS = [
    "user_intent",
    "policy_text",
    "agent_plan_text",
    "transaction_context_text",
    "intent_constraints",
    "user_policy",
    "agent_plan",
    "transaction_context",
    "scenario_variant",
    "action_type",
]


def build_text_feature(row: pd.Series) -> str:
    return " ".join(str(row.get(col, "")) for col in TEXT_COLUMNS)


def semantic_predict(row: pd.Series) -> str:
    intent = derive_intent_from_dataset_row(row.to_dict())
    policy = safe_json_loads(row.get("user_policy"), {})
    plan = safe_json_loads(row.get("agent_plan"), {})
    context = safe_json_loads(row.get("transaction_context"), {})
    result = validation_decision(intent, policy, plan, context, row.get("scenario_variant"))
    return result.decision


def add_metrics(
    metrics_rows: list,
    confusion_rows: list,
    prediction_rows: list,
    model_name: str,
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: list[str],
) -> None:
    pred = np.array(y_pred)
    true = y_true.to_numpy()
    critical_gold = test_df["critical_violation_present"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    unsafe_allow = (pred == "ALLOW") & (true != "ALLOW")
    critical_unsafe_allow = (pred == "ALLOW") & critical_gold
    over_reject = (pred == "REJECT") & (true == "ALLOW")
    losses = [risk_weighted_loss(p, g, True, c) for p, g, c in zip(pred, true, critical_gold)]

    metrics_rows.append(
        {
            "model": model_name,
            "n_test": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, average="macro", labels=LABELS)),
            "weighted_f1": float(f1_score(y_true, pred, average="weighted", labels=LABELS)),
            "unsafe_allow_count": int(unsafe_allow.sum()),
            "unsafe_allow_rate_total": float(unsafe_allow.mean()),
            "critical_unsafe_allow_count": int(critical_unsafe_allow.sum()),
            "critical_unsafe_allow_rate_total": float(critical_unsafe_allow.mean()),
            "human_review_load": float((pred == "HUMAN_REVIEW").mean()),
            "over_reject_count": int(over_reject.sum()),
            "over_reject_rate_total": float(over_reject.mean()),
            "risk_weighted_loss_mean": float(np.mean(losses)),
            "risk_weighted_loss_total": float(np.sum(losses)),
        }
    )
    cm = confusion_matrix(y_true, pred, labels=LABELS)
    for i, actual in enumerate(LABELS):
        for j, predicted in enumerate(LABELS):
            confusion_rows.append(
                {"model": model_name, "actual": actual, "predicted": predicted, "count": int(cm[i, j])}
            )
    for scenario_id, actual, predicted in zip(test_df["scenario_id"], y_true, pred):
        prediction_rows.append(
            {"model": model_name, "scenario_id": scenario_id, "actual": actual, "predicted": predicted}
        )


def run_experiment(scenarios_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scenarios_csv)
    if "split" not in df.columns:
        raise ValueError("Scenario CSV must include a 'split' column.")
    df["text_feature"] = df.apply(build_text_feature, axis=1)

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Scenario CSV must include non-empty train and test splits.")

    x_train = train["text_feature"]
    y_train = train["expected_decision"]
    x_test = test["text_feature"]
    y_test = test["expected_decision"]

    models = {
        "Dummy majority baseline": DummyClassifier(strategy="most_frequent"),
        "TF-IDF Linear SVM": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000), LinearSVC()
        ),
        "TF-IDF Logistic Regression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Char TF-IDF Linear SVM": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=30000),
            LinearSVC(),
        ),
    }

    metrics_rows: list = []
    confusion_rows: list = []
    prediction_rows: list = []

    fitted_predictions: dict[str, list[str]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = list(model.predict(x_test))
        fitted_predictions[name] = predictions
        add_metrics(metrics_rows, confusion_rows, prediction_rows, name, test, y_test, predictions)

    semantic_predictions = [semantic_predict(row) for _, row in test.iterrows()]
    add_metrics(
        metrics_rows,
        confusion_rows,
        prediction_rows,
        "Semantic-policy validator v5.6",
        test,
        y_test,
        semantic_predictions,
    )

    text_predictions = fitted_predictions["TF-IDF Linear SVM"]
    hybrid_predictions = []
    for text_pred, semantic_pred in zip(text_predictions, semantic_predictions):
        if semantic_pred == "REJECT":
            hybrid_predictions.append("REJECT")
        elif semantic_pred == "HUMAN_REVIEW" and text_pred == "ALLOW":
            hybrid_predictions.append("HUMAN_REVIEW")
        else:
            hybrid_predictions.append(text_pred)
    add_metrics(
        metrics_rows,
        confusion_rows,
        prediction_rows,
        "Hybrid semantic override v5.6",
        test,
        y_test,
        hybrid_predictions,
    )

    pd.DataFrame(metrics_rows).to_csv(out_dir / "plansafebench_evm_corrected_main_decision_metrics_v5_6.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out_dir / "plansafebench_evm_corrected_main_confusion_matrices_v5_6.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out_dir / "plansafebench_evm_corrected_main_test_predictions_v5_6.csv", index=False)
    print(pd.DataFrame(metrics_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True, type=Path, help="Corrected scenario CSV")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    run_experiment(args.scenarios, args.out_dir)


if __name__ == "__main__":
    main()
