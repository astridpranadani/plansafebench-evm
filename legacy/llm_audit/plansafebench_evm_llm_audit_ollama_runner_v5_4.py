#!/usr/bin/env python3
"""
Ollama local runner for PlanSafeBench-EVM LLM audit.

Requires Ollama running locally.

Environment variables:
  OLLAMA_MODEL      required, e.g. llama3.1:8b or qwen2.5:7b
  OLLAMA_HOST       default http://localhost:11434
  LLM_MODEL_SLOT    optional, default open_weight_or_local

Example:
  set OLLAMA_MODEL=qwen2.5:7b
  python plansafebench_evm_llm_audit_ollama_runner_v5_4.py --prompts batch.csv --out outputs_ollama.csv
"""
import argparse, json, os, time, re
from pathlib import Path
import pandas as pd
import requests

def extract_json(text):
    text = str(text or "").strip()
    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.dumps(json.loads(m.group(0)), ensure_ascii=False)
        except Exception:
            return ""
    return ""

def call_ollama(system_prompt, user_prompt, temperature, top_p):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        raise RuntimeError("Missing OLLAMA_MODEL")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"temperature": float(temperature), "top_p": float(top_p)}
    }
    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", ""), ""
    except Exception as e:
        return "", str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.prompts)
    if args.limit is not None:
        df = df.iloc[args.start:args.start + args.limit].copy()
    else:
        df = df.iloc[args.start:].copy()

    out_path = Path(args.out)
    done = set()
    rows = []
    if out_path.exists():
        old = pd.read_csv(out_path)
        rows = old.to_dict("records")
        done = set(old["prompt_id"].astype(str))

    for _, r in df.iterrows():
        prompt_id = str(r["prompt_id"])
        if prompt_id in done:
            continue
        raw, err = call_ollama(str(r["system_prompt"]), str(r["user_prompt"]), float(r.get("temperature",0.2)), float(r.get("top_p",1.0)))
        rows.append({
            "prompt_id": prompt_id,
            "source_template_id": r.get("source_template_id", ""),
            "source_transaction_hash": r.get("source_transaction_hash", ""),
            "action_type": r.get("action_type", ""),
            "prompt_variant": r.get("prompt_variant", ""),
            "challenge_type": r.get("challenge_type", ""),
            "model_slot": os.environ.get("LLM_MODEL_SLOT", "open_weight_or_local"),
            "model_name": os.environ.get("OLLAMA_MODEL", ""),
            "model_version_or_snapshot": os.environ.get("OLLAMA_MODEL", ""),
            "provider": "ollama",
            "raw_model_output": raw,
            "parsed_output_json": extract_json(raw),
            "generation_error": err,
        })
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"{len(rows)} total rows saved | prompt_id={prompt_id} | error={bool(err)}")
        if args.sleep:
            time.sleep(args.sleep)
    print(f"Saved outputs to {out_path}")

if __name__ == "__main__":
    main()
