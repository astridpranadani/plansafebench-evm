#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter

TX_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-total", type=int, default=150)
    ap.add_argument("--min-categories", type=int, default=9)
    ap.add_argument("--require-receipt-ratio", type=float, default=0.90)
    args = ap.parse_args()

    rows = []
    for line_no, line in enumerate(Path(args.templates).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                r = json.loads(line)
                r["_line_no"] = line_no
                rows.append(r)
            except Exception as e:
                rows.append({"_line_no": line_no, "_parse_error": str(e)})

    errors = []
    hashes = []
    categories = Counter()
    receipt_ok = 0
    input_ok = 0
    selector_ok = 0
    ok_rows = 0

    for r in rows:
        if "_parse_error" in r:
            errors.append({"line": r["_line_no"], "error": r["_parse_error"]})
            continue
        txh = str(r.get("transaction_hash","")).lower()
        if not TX_RE.match(txh):
            errors.append({"line": r["_line_no"], "error": "invalid_transaction_hash"})
        else:
            hashes.append(txh)
        if r.get("extraction_status") == "ok":
            ok_rows += 1
        if r.get("receipt_available") or r.get("receipt_status") is not None:
            receipt_ok += 1
        if isinstance(r.get("input"), str) and r.get("input", "") != "":
            input_ok += 1
        if isinstance(r.get("method_selector"), str) and r.get("method_selector", "").startswith("0x") and len(r.get("method_selector")) == 10:
            selector_ok += 1
        cat = r.get("action_type") or r.get("candidate_action_type") or r.get("source_contract_group") or "unknown"
        categories[str(cat)] += 1

    duplicate_count = len(hashes) - len(set(hashes))
    n = len(rows)
    receipt_ratio = receipt_ok / n if n else 0
    input_ratio = input_ok / n if n else 0
    selector_ratio = selector_ok / n if n else 0

    gate_pass = (
        n >= args.min_total and
        len(categories) >= args.min_categories and
        duplicate_count == 0 and
        len(errors) == 0 and
        receipt_ratio >= args.require_receipt_ratio and
        input_ratio >= 0.90 and
        selector_ratio >= 0.80
    )

    summary = {
        "row_count": n,
        "unique_transaction_hashes": len(set(hashes)),
        "duplicate_transaction_hash_count": duplicate_count,
        "category_counts": dict(categories),
        "represented_category_count": len(categories),
        "ok_rows": ok_rows,
        "receipt_ratio": receipt_ratio,
        "input_ratio": input_ratio,
        "method_selector_ratio": selector_ratio,
        "error_count": len(errors),
        "errors_sample": errors[:25],
        "q1_level_1_2_gate_pass": gate_pass,
        "claim_if_pass": "real-world transaction-template anchored benchmark",
        "claim_if_fail": "not yet final Q1 real-world transaction-template dataset"
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
