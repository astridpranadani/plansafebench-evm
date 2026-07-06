#!/usr/bin/env python3
"""
Quality gate for PlanSafeBench-EVM LLM audit outputs.
Checks coverage, duplicates, parse rate after scoring, and missing model metadata.
"""
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--scored", required=False)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    prompts = pd.read_csv(args.prompts)
    outs = pd.read_csv(args.outputs)
    expected_prompts = set(prompts["prompt_id"].astype(str))
    report = {
        "expected_prompt_count_per_model": len(expected_prompts),
        "output_rows": int(len(outs)),
        "model_groups": [],
        "duplicate_key_count": 0,
        "overall_status": "PASS"
    }
    key_cols = ["prompt_id","model_slot","model_name"]
    if all(c in outs.columns for c in key_cols):
        report["duplicate_key_count"] = int(outs.duplicated(key_cols).sum())
        if report["duplicate_key_count"] > 0:
            report["overall_status"] = "CHECK"
        for keys, g in outs.groupby(["model_slot","model_name"], dropna=False):
            got = set(g["prompt_id"].astype(str))
            missing = len(expected_prompts - got)
            extra = len(got - expected_prompts)
            row = {
                "model_slot": str(keys[0]),
                "model_name": str(keys[1]),
                "rows": int(len(g)),
                "unique_prompt_ids": int(len(got)),
                "missing_prompt_count": int(missing),
                "extra_prompt_count": int(extra),
                "coverage_rate": float(len(got & expected_prompts) / len(expected_prompts)),
            }
            if row["coverage_rate"] < 1.0:
                report["overall_status"] = "CHECK"
            report["model_groups"].append(row)
    else:
        report["overall_status"] = "CHECK"
        report["missing_key_columns"] = [c for c in key_cols if c not in outs.columns]

    if args.scored and Path(args.scored).exists():
        scored = pd.read_csv(args.scored)
        if "parse_ok" in scored.columns:
            report["parse_ok_rate"] = float(scored["parse_ok"].mean())
        if "unsafe_allow" in scored.columns:
            report["unsafe_allow_rate"] = float(scored["unsafe_allow"].mean())
        if "critical_unsafe_allow" in scored.columns:
            report["critical_unsafe_allow_rate"] = float(scored["critical_unsafe_allow"].mean())

    Path(args.out_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
