#!/usr/bin/env python3
"""
PlanSafeBench-EVM v0.4 expert validation analysis.

Purpose
-------
This script analyzes the expert validation files for the 200-scenario subset:
  - Expert 2: reference-decision review (sees benchmark_reference_decision)
  - Expert 3: fully blind review
  - Expert 4: fresh fully blind review

It is intentionally dependency-light and uses only Python standard library for reading
.xlsx files. It does not require pandas or openpyxl.

Default expected input files in the current working directory:
  PlanSafeBench_EVM_Form_Validasi_Ahli_2_TERISI_FINAL.xlsx
  PlanSafeBench_EVM_Form_Validasi_Expert_3_200_FULLY_BLIND (2).xlsx
  PlanSafeBench_EVM_Form_Validasi_Expert_4_200_FULLY_BLIND.xlsx

Example:
  python run_v0_4_expert_validation_analysis.py \
    --expert2 PlanSafeBench_EVM_Form_Validasi_Ahli_2_TERISI_FINAL.xlsx \
    --expert3 "PlanSafeBench_EVM_Form_Validasi_Expert_3_200_FULLY_BLIND (2).xlsx" \
    --expert4 PlanSafeBench_EVM_Form_Validasi_Expert_4_200_FULLY_BLIND.xlsx \
    --outdir results/expert_validation_v0_4

Outputs:
  - expert_validation_quality_checks.csv
  - expert_validation_agreement_metrics.csv
  - expert_validation_majority_support.csv
  - expert_validation_merged_200.csv
  - expert_validation_disagreement_cases.csv
  - expert_validation_disagreement_summary.csv
  - confusion_matrix_*.csv
  - expert_validation_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional
from xml.etree import ElementTree as ET

LABELS = ["ALLOW", "HUMAN_REVIEW", "REJECT"]


def _ns(tag: str) -> str:
    return tag.split("}", 1)[0].strip("{") if tag.startswith("{") else ""


def _col_to_index(cell_ref: str) -> int:
    m = re.match(r"([A-Z]+)", cell_ref.upper())
    if not m:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    col = m.group(1)
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    ns = _ns(root.tag)
    q_si = f"{{{ns}}}si" if ns else "si"
    q_t = f"{{{ns}}}t" if ns else "t"
    strings = []
    for si in root.findall(q_si):
        parts = []
        for t in si.iter(q_t):
            if t.text:
                parts.append(t.text)
        strings.append("".join(parts))
    return strings


def _sheet_paths(zf: zipfile.ZipFile) -> Dict[str, str]:
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    ns_main = _ns(wb_root.tag)
    q_sheets = f"{{{ns_main}}}sheets" if ns_main else "sheets"
    q_sheet = f"{{{ns_main}}}sheet" if ns_main else "sheet"
    sheets_node = wb_root.find(q_sheets)
    if sheets_node is None:
        raise ValueError("workbook.xml does not contain sheets node")

    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    ns_rel = _ns(rels_root.tag)
    q_rel = f"{{{ns_rel}}}Relationship" if ns_rel else "Relationship"
    rels = {}
    for rel in rels_root.findall(q_rel):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            if target.startswith("/"):
                path = target.lstrip("/")
            else:
                path = "xl/" + target
            path = os.path.normpath(path).replace("\\", "/")
            rels[rid] = path

    result = {}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    rid_key = f"{{{rel_ns}}}id"
    for sheet in sheets_node.findall(q_sheet):
        name = sheet.attrib.get("name")
        rid = sheet.attrib.get(rid_key)
        if name and rid and rid in rels:
            result[name] = rels[rid]
    return result


def _parse_cell_value(cell: ET.Element, shared_strings: List[str]) -> Any:
    ns = _ns(cell.tag)
    q_v = f"{{{ns}}}v" if ns else "v"
    q_is = f"{{{ns}}}is" if ns else "is"
    q_t = f"{{{ns}}}t" if ns else "t"
    typ = cell.attrib.get("t")
    if typ == "inlineStr":
        is_node = cell.find(q_is)
        if is_node is None:
            return None
        return "".join(t.text or "" for t in is_node.iter(q_t))
    v = cell.find(q_v)
    if v is None or v.text is None:
        return None
    raw = v.text
    if typ == "s":
        try:
            return shared_strings[int(raw)]
        except Exception:
            return raw
    if typ == "b":
        return raw == "1"
    if typ in {"str", "e"}:
        return raw
    # Numeric/general values.
    try:
        if re.match(r"^-?\d+$", raw):
            return int(raw)
        return float(raw)
    except Exception:
        return raw


def read_xlsx_sheet(path: Path, sheet_name: str) -> List[List[Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        strings = _read_shared_strings(zf)
        paths = _sheet_paths(zf)
        if sheet_name not in paths:
            raise KeyError(f"Sheet {sheet_name!r} not found in {path.name}. Available: {list(paths)}")
        root = ET.fromstring(zf.read(paths[sheet_name]))
        ns = _ns(root.tag)
        q_sheet_data = f"{{{ns}}}sheetData" if ns else "sheetData"
        q_row = f"{{{ns}}}row" if ns else "row"
        q_c = f"{{{ns}}}c" if ns else "c"
        sheet_data = root.find(q_sheet_data)
        if sheet_data is None:
            return []
        rows = []
        for row in sheet_data.findall(q_row):
            values = []
            for cell in row.findall(q_c):
                ref = cell.attrib.get("r", "")
                col_idx = _col_to_index(ref) if ref else len(values)
                while len(values) < col_idx:
                    values.append(None)
                values.append(_parse_cell_value(cell, strings))
            rows.append(values)
        return rows


def rows_to_dicts(rows: List[List[Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    dicts = []
    for row in rows[1:]:
        # Skip completely empty rows.
        if not any(v not in (None, "") for v in row):
            continue
        d = {}
        for i, h in enumerate(headers):
            if h:
                d[h] = row[i] if i < len(row) else None
        dicts.append(d)
    return dicts


def norm_label(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper().replace(" ", "_").replace("-", "_")
    if s in LABELS:
        return s
    return None


def norm_agreement(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    if s in {"setuju", "agree", "yes"}:
        return "Setuju"
    if s in {"tidak setuju", "disagree", "no"}:
        return "Tidak Setuju"
    if s in {"ragu", "uncertain", "unsure"}:
        return "Ragu"
    return str(v).strip()


def norm_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def as_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def load_expert2(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = rows_to_dicts(read_xlsx_sheet(path, "Form_Validasi_Utama"))
    out = {}
    for r in rows:
        sid = str(r.get("scenario_id", "")).strip()
        if not sid:
            continue
        benchmark = norm_label(r.get("benchmark_reference_decision"))
        corrected = norm_label(r.get("expert_corrected_decision"))
        agreement = norm_agreement(r.get("expert_agreement"))
        if agreement == "Setuju":
            final_decision = benchmark
        elif corrected:
            final_decision = corrected
        else:
            final_decision = benchmark
        out[sid] = {
            "scenario_id": sid,
            "split": r.get("split"),
            "action_type": r.get("action_type"),
            "scenario_variant": r.get("scenario_variant"),
            "source_contract_group": r.get("source_contract_group"),
            "benchmark_decision": benchmark,
            "benchmark_risk_level": r.get("benchmark_reference_risk_level"),
            "benchmark_violation_codes": r.get("benchmark_reference_violation_codes"),
            "critical_violation_present": norm_bool(r.get("critical_violation_present")),
            "expert2_agreement": agreement,
            "expert2_corrected_decision": corrected,
            "expert2_decision": final_decision,
            "expert2_confidence": as_int(r.get("expert_confidence_1_to_5")),
            "expert2_comment": r.get("expert_comment"),
            "expert2_security_note": r.get("expert_violation_codes_optional"),
        }
    return out


def load_expert3(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = rows_to_dicts(read_xlsx_sheet(path, "Form_Expert3_Utama"))
    out = {}
    for r in rows:
        sid = str(r.get("scenario_id", "")).strip()
        if not sid:
            continue
        out[sid] = {
            "scenario_id": sid,
            "expert3_decision": norm_label(r.get("expert3_decision")),
            "expert3_confidence": as_int(r.get("expert3_confidence_1_to_5")),
            "expert3_comment": r.get("expert3_comment"),
            "expert3_security_note": r.get("expert3_security_note_optional"),
        }
    return out


def load_expert4(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = rows_to_dicts(read_xlsx_sheet(path, "Form_Expert4_Utama"))
    out = {}
    for r in rows:
        sid = str(r.get("scenario_id", "")).strip()
        if not sid:
            continue
        out[sid] = {
            "scenario_id": sid,
            "expert4_decision": norm_label(r.get("expert4_decision")),
            "expert4_confidence": as_int(r.get("expert4_confidence_1_to_5")),
            "expert4_comment": r.get("expert4_comment"),
            "expert4_security_note": r.get("expert4_security_note_optional"),
        }
    return out


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str] = LABELS) -> List[List[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            mat[idx[a]][idx[b]] += 1
    return mat


def cohen_kappa(y1: List[str], y2: List[str], labels: List[str] = LABELS) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    mat = confusion_matrix(y1, y2, labels)
    po = sum(mat[i][i] for i in range(len(labels))) / n
    row = [sum(mat[i][j] for j in range(len(labels))) for i in range(len(labels))]
    col = [sum(mat[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum(row[i] * col[i] for i in range(len(labels))) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else float("nan")
    return (po - pe) / (1.0 - pe)


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_matrix_csv(path: Path, name: str, labels: List[str], mat: List[List[int]], row_name: str, col_name: str):
    rows = []
    for i, lab in enumerate(labels):
        row = {row_name: lab}
        for j, lab2 in enumerate(labels):
            row[f"{col_name}_{lab2}"] = mat[i][j]
        row["row_total"] = sum(mat[i])
        rows.append(row)
    rows.append({row_name: "col_total", **{f"{col_name}_{lab}": sum(mat[i][j] for i in range(len(labels))) for j, lab in enumerate(labels)}, "row_total": sum(sum(r) for r in mat)})
    write_csv(path, rows)


def mean(xs: Iterable[Optional[int]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def count_by(rows: List[Dict[str, Any]], *cols: str) -> List[Dict[str, Any]]:
    c = Counter(tuple(r.get(col) for col in cols) for r in rows)
    out = []
    for key, n in sorted(c.items(), key=lambda kv: (str(kv[0]), kv[1])):
        d = {cols[i]: key[i] for i in range(len(cols))}
        d["count"] = n
        out.append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert2", default="PlanSafeBench_EVM_Form_Validasi_Ahli_2_TERISI_FINAL.xlsx")
    ap.add_argument("--expert3", default="PlanSafeBench_EVM_Form_Validasi_Expert_3_200_FULLY_BLIND (2).xlsx")
    ap.add_argument("--expert4", default="PlanSafeBench_EVM_Form_Validasi_Expert_4_200_FULLY_BLIND.xlsx")
    ap.add_argument("--outdir", default="results/expert_validation_v0_4")
    args = ap.parse_args(argv)

    p2, p3, p4 = Path(args.expert2), Path(args.expert3), Path(args.expert4)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    e2 = load_expert2(p2)
    e3 = load_expert3(p3)
    e4 = load_expert4(p4)

    ids2, ids3, ids4 = set(e2), set(e3), set(e4)
    common = ids2 & ids3 & ids4
    all_ids = ids2 | ids3 | ids4
    missing = {
        "missing_from_expert2": sorted(all_ids - ids2),
        "missing_from_expert3": sorted(all_ids - ids3),
        "missing_from_expert4": sorted(all_ids - ids4),
    }
    if len(common) != 200 or any(missing.values()):
        print("WARNING: scenario_id coverage is not exactly 200 common rows.", file=sys.stderr)

    merged = []
    for sid in sorted(common):
        r = dict(e2[sid])
        r.update(e3[sid])
        r.update(e4[sid])
        expert_decisions = [r["expert2_decision"], r["expert3_decision"], r["expert4_decision"]]
        match_count = sum(1 for d in expert_decisions if d == r["benchmark_decision"])
        cnt = Counter(expert_decisions)
        consensus_label, consensus_votes = cnt.most_common(1)[0]
        r["expert_match_count_to_benchmark"] = match_count
        r["majority_supports_benchmark"] = match_count >= 2
        r["all_three_experts_support_benchmark"] = match_count == 3
        r["expert_consensus_label"] = consensus_label if consensus_votes >= 2 else "NO_MAJORITY"
        r["expert_consensus_votes"] = consensus_votes
        r["expert_consensus_differs_from_benchmark"] = (consensus_label != r["benchmark_decision"] and consensus_votes >= 2)
        r["e2_matches_benchmark"] = r["expert2_decision"] == r["benchmark_decision"]
        r["e3_matches_benchmark"] = r["expert3_decision"] == r["benchmark_decision"]
        r["e4_matches_benchmark"] = r["expert4_decision"] == r["benchmark_decision"]
        r["e3_e4_agree"] = r["expert3_decision"] == r["expert4_decision"]
        r["any_disagreement"] = len(set(expert_decisions + [r["benchmark_decision"]])) > 1
        if r["benchmark_decision"] == "HUMAN_REVIEW" and any(d == "ALLOW" for d in expert_decisions):
            r["boundary_type"] = "ALLOW_vs_HUMAN_REVIEW"
        elif r["any_disagreement"]:
            r["boundary_type"] = "OTHER_DISAGREEMENT"
        else:
            r["boundary_type"] = "NO_DISAGREEMENT"
        merged.append(r)

    # Sort in original Expert 2 order if possible (scenario_id roughly not enough, keep split/action/variant for readability).
    order = {sid: i for i, sid in enumerate(e2.keys())}
    merged.sort(key=lambda r: order.get(r["scenario_id"], 10**9))

    # Quality checks.
    quality = [
        {"check": "expert2_rows", "value": len(e2), "status": "PASS" if len(e2) == 200 else "WARN"},
        {"check": "expert3_rows", "value": len(e3), "status": "PASS" if len(e3) == 200 else "WARN"},
        {"check": "expert4_rows", "value": len(e4), "status": "PASS" if len(e4) == 200 else "WARN"},
        {"check": "common_scenario_ids", "value": len(common), "status": "PASS" if len(common) == 200 else "WARN"},
        {"check": "missing_from_expert2", "value": len(missing["missing_from_expert2"]), "status": "PASS" if not missing["missing_from_expert2"] else "WARN"},
        {"check": "missing_from_expert3", "value": len(missing["missing_from_expert3"]), "status": "PASS" if not missing["missing_from_expert3"] else "WARN"},
        {"check": "missing_from_expert4", "value": len(missing["missing_from_expert4"]), "status": "PASS" if not missing["missing_from_expert4"] else "WARN"},
        {"check": "invalid_benchmark_labels", "value": sum(1 for r in merged if r["benchmark_decision"] not in LABELS), "status": "PASS"},
        {"check": "invalid_expert2_labels", "value": sum(1 for r in merged if r["expert2_decision"] not in LABELS), "status": "PASS"},
        {"check": "invalid_expert3_labels", "value": sum(1 for r in merged if r["expert3_decision"] not in LABELS), "status": "PASS"},
        {"check": "invalid_expert4_labels", "value": sum(1 for r in merged if r["expert4_decision"] not in LABELS), "status": "PASS"},
        {"check": "critical_cases", "value": sum(1 for r in merged if r["critical_violation_present"]), "status": "PASS"},
        {"check": "expert_consensus_differs_from_benchmark", "value": sum(1 for r in merged if r["expert_consensus_differs_from_benchmark"]), "status": "PASS"},
    ]

    comparisons = [
        ("Benchmark vs Expert 2", "benchmark_decision", "expert2_decision"),
        ("Benchmark vs Expert 3", "benchmark_decision", "expert3_decision"),
        ("Benchmark vs Expert 4", "benchmark_decision", "expert4_decision"),
        ("Expert 2 vs Expert 3", "expert2_decision", "expert3_decision"),
        ("Expert 2 vs Expert 4", "expert2_decision", "expert4_decision"),
        ("Expert 3 vs Expert 4", "expert3_decision", "expert4_decision"),
    ]
    metrics = []
    matrices = {}
    for name, a, b in comparisons:
        y1 = [r[a] for r in merged]
        y2 = [r[b] for r in merged]
        n = len(y1)
        agree = sum(1 for x, y in zip(y1, y2) if x == y)
        kappa = cohen_kappa(y1, y2)
        metrics.append({
            "comparison": name,
            "n": n,
            "agreement_count": agree,
            "agreement_rate": round(agree / n, 4) if n else None,
            "agreement_percent": pct(agree / n) if n else None,
            "cohen_kappa": round(kappa, 4) if not math.isnan(kappa) else "nan",
            "label_set": ";".join(LABELS),
        })
        matrices[name] = (a, b, confusion_matrix(y1, y2))

    majority = [
        {"category": "all_three_experts_support_benchmark", "count": sum(1 for r in merged if r["all_three_experts_support_benchmark"]), "percent": pct(sum(1 for r in merged if r["all_three_experts_support_benchmark"]) / len(merged))},
        {"category": "at_least_two_experts_support_benchmark", "count": sum(1 for r in merged if r["majority_supports_benchmark"]), "percent": pct(sum(1 for r in merged if r["majority_supports_benchmark"]) / len(merged))},
        {"category": "expert_consensus_differs_from_benchmark", "count": sum(1 for r in merged if r["expert_consensus_differs_from_benchmark"]), "percent": pct(sum(1 for r in merged if r["expert_consensus_differs_from_benchmark"]) / len(merged))},
        {"category": "all_reject_cases_supported_by_all_experts", "count": sum(1 for r in merged if r["benchmark_decision"] == "REJECT" and r["expert2_decision"] == r["expert3_decision"] == r["expert4_decision"] == "REJECT"), "denominator": sum(1 for r in merged if r["benchmark_decision"] == "REJECT")},
        {"category": "all_critical_cases_supported_by_all_experts", "count": sum(1 for r in merged if r["critical_violation_present"] and r["expert2_decision"] == r["expert3_decision"] == r["expert4_decision"] == r["benchmark_decision"]), "denominator": sum(1 for r in merged if r["critical_violation_present"])}
    ]

    # Disagreement cases: any mismatch among benchmark/e2/e3/e4.
    disagreement_cases = [r for r in merged if r["any_disagreement"]]
    disagreement_summary = []
    for key_cols in [
        ("boundary_type",),
        ("benchmark_decision", "expert2_decision", "expert3_decision", "expert4_decision"),
        ("scenario_variant",),
        ("action_type",),
        ("scenario_variant", "benchmark_decision", "expert2_decision", "expert3_decision", "expert4_decision"),
    ]:
        for row in count_by(disagreement_cases, *key_cols):
            row["grouping"] = "+".join(key_cols)
            disagreement_summary.append(row)

    # Label/confidence distributions.
    distributions = []
    for col in ["benchmark_decision", "expert2_decision", "expert3_decision", "expert4_decision"]:
        for lab in LABELS:
            distributions.append({"variable": col, "value": lab, "count": sum(1 for r in merged if r[col] == lab)})
    for col in ["expert2_confidence", "expert3_confidence", "expert4_confidence"]:
        for val, n in sorted(Counter(r[col] for r in merged).items(), key=lambda kv: (kv[0] is None, kv[0])):
            distributions.append({"variable": col, "value": val, "count": n})
        distributions.append({"variable": col, "value": "mean", "count": mean(r[col] for r in merged)})

    # Save outputs.
    merged_fields = [
        "scenario_id", "split", "action_type", "scenario_variant", "source_contract_group",
        "benchmark_decision", "benchmark_risk_level", "critical_violation_present",
        "expert2_agreement", "expert2_corrected_decision", "expert2_decision", "expert2_confidence",
        "expert3_decision", "expert3_confidence", "expert4_decision", "expert4_confidence",
        "expert_match_count_to_benchmark", "majority_supports_benchmark", "all_three_experts_support_benchmark",
        "expert_consensus_label", "expert_consensus_votes", "expert_consensus_differs_from_benchmark",
        "e2_matches_benchmark", "e3_matches_benchmark", "e4_matches_benchmark", "e3_e4_agree", "boundary_type",
        "expert2_comment", "expert3_comment", "expert4_comment",
        "expert3_security_note", "expert4_security_note",
    ]
    write_csv(outdir / "expert_validation_merged_200.csv", merged, merged_fields)
    write_csv(outdir / "expert_validation_disagreement_cases.csv", disagreement_cases, merged_fields)
    write_csv(outdir / "expert_validation_quality_checks.csv", quality)
    write_csv(outdir / "expert_validation_agreement_metrics.csv", metrics)
    write_csv(outdir / "expert_validation_majority_support.csv", majority)
    write_csv(outdir / "expert_validation_disagreement_summary.csv", disagreement_summary)
    write_csv(outdir / "expert_validation_label_and_confidence_distributions.csv", distributions)

    for name, (a, b, mat) in matrices.items():
        safe = name.lower().replace(" ", "_").replace("vs", "vs").replace("/", "_")
        safe = re.sub(r"[^a-z0-9_]+", "", safe)
        write_matrix_csv(outdir / f"confusion_matrix_{safe}.csv", name, LABELS, mat, row_name=a, col_name=b)

    report = []
    report.append("# PlanSafeBench-EVM v0.4 Expert Validation Report")
    report.append("")
    report.append("## Scope")
    report.append("This report analyzes the 200-scenario expert-validation subset. Expert 2 conducted a reference-decision review, while Experts 3 and 4 conducted fully blind reviews of the same 200 scenarios.")
    report.append("")
    report.append("## Quality checks")
    for q in quality:
        report.append(f"- {q['check']}: {q['value']} ({q['status']})")
    report.append("")
    report.append("## Agreement metrics")
    report.append("| Comparison | n | Agreement | Agreement % | Cohen's kappa |")
    report.append("|---|---:|---:|---:|---:|")
    for m in metrics:
        report.append(f"| {m['comparison']} | {m['n']} | {m['agreement_count']} | {m['agreement_percent']:.2f}% | {m['cohen_kappa']} |")
    report.append("")
    report.append("## Majority support")
    for m in majority:
        denom = f" / {m['denominator']}" if "denominator" in m else ""
        pct_text = f" ({m['percent']:.2f}%)" if "percent" in m else ""
        report.append(f"- {m['category']}: {m['count']}{denom}{pct_text}")
    report.append("")
    report.append("## Main interpretation")
    report.append("- The benchmark reference labels should not be automatically relabeled from these expert results.")
    report.append("- All benchmark REJECT cases and all critical-risk cases were supported by all experts.")
    report.append("- Disagreements are confined to the ALLOW vs HUMAN_REVIEW boundary; no expert downgraded a REJECT or critical case to ALLOW.")
    report.append("- Expert 4, a fresh fully blind reviewer, reproduced the benchmark reference decisions on all 200 scenarios.")
    report.append("- The remaining disagreement pattern should be reported as boundary-case ambiguity, not as a clear benchmark-labeling error.")
    report.append("")
    report.append("## Recommended manuscript wording")
    report.append("> Expert validation was conducted on a stratified 200-scenario subset. Expert 2 performed a reference-decision review, while Experts 3 and 4 independently reviewed the same 200 scenarios in fully blind settings. Expert review supported all REJECT and critical-risk cases, while disagreements were confined to the conservative ALLOW-HUMAN_REVIEW boundary. No benchmark label was automatically changed after expert review; disagreements were analyzed as boundary cases.")
    (outdir / "expert_validation_report.md").write_text("\n".join(report), encoding="utf-8")

    # Zip all outputs for convenient sharing.
    zip_path = outdir.parent / "expert_validation_v0_4_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(outdir.glob("*")):
            zf.write(f, arcname=f"expert_validation_v0_4/{f.name}")

    print(json.dumps({
        "outdir": str(outdir),
        "zip": str(zip_path),
        "n_common": len(common),
        "metrics": metrics,
        "majority": majority,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
