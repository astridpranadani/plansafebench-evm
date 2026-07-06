#!/usr/bin/env python3
"""
Generic OpenAI-compatible runner for PlanSafeBench-EVM LLM audit.

This script calls a chat-completions compatible endpoint. It can be used with
providers that expose an OpenAI-compatible /chat/completions API.

Environment variables:
  LLM_API_KEY       required
  LLM_API_BASE      default: https://api.openai.com/v1
  LLM_MODEL         required
  LLM_PROVIDER      optional label, e.g. openai, openrouter, local_proxy
  LLM_MODEL_SLOT    optional label, e.g. frontier_proprietary
  LLM_TIMEOUT       default: 120

Example PowerShell:
  $env:LLM_API_KEY="..."
  $env:LLM_API_BASE="https://api.openai.com/v1"
  $env:LLM_MODEL="FILL_MODEL_NAME"
  $env:LLM_PROVIDER="FILL_PROVIDER"
  $env:LLM_MODEL_SLOT="frontier_proprietary"
  python plansafebench_evm_llm_audit_openai_compatible_runner_v5_4.py --prompts batch.csv --out outputs_batch.csv
"""
import argparse, json, os, time, re
from pathlib import Path
import pandas as pd
import requests

def extract_json(text):
    if text is None:
        return ""
    text = str(text).strip()
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return ""
    return ""

def call_chat(system_prompt, user_prompt, temperature, top_p, max_tokens, retries=4):
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY")
    base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError("Missing LLM_MODEL")

    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens)
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=int(os.environ.get("LLM_TIMEOUT", "120")))
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(min(60, 2 ** attempt * 5))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"], ""
        except Exception as e:
            last_err = str(e)
            time.sleep(min(60, 2 ** attempt * 5))
    return "", last_err or "unknown error"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="0-based start index within prompt CSV")
    ap.add_argument("--limit", type=int, default=None, help="number of rows to process")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep between calls")
    args = ap.parse_args()

    df = pd.read_csv(args.prompts)
    if args.limit is not None:
        df = df.iloc[args.start:args.start + args.limit].copy()
    else:
        df = df.iloc[args.start:].copy()

    out_path = Path(args.out)
    done = set()
    existing_rows = []
    if out_path.exists():
        old = pd.read_csv(out_path)
        if "prompt_id" in old.columns:
            done = set(old["prompt_id"].astype(str))
            existing_rows = old.to_dict("records")

    rows = existing_rows
    for _, r in df.iterrows():
        prompt_id = str(r["prompt_id"])
        if prompt_id in done:
            continue
        raw, err = call_chat(
            str(r["system_prompt"]),
            str(r["user_prompt"]),
            float(r.get("temperature", 0.2)),
            float(r.get("top_p", 1.0)),
            int(r.get("max_output_tokens", 700))
        )
        parsed = extract_json(raw)
        row = {
            "prompt_id": prompt_id,
            "source_template_id": r.get("source_template_id", ""),
            "source_transaction_hash": r.get("source_transaction_hash", ""),
            "action_type": r.get("action_type", ""),
            "prompt_variant": r.get("prompt_variant", ""),
            "challenge_type": r.get("challenge_type", ""),
            "model_slot": os.environ.get("LLM_MODEL_SLOT", ""),
            "model_name": os.environ.get("LLM_MODEL", ""),
            "model_version_or_snapshot": os.environ.get("LLM_MODEL_VERSION", os.environ.get("LLM_MODEL", "")),
            "provider": os.environ.get("LLM_PROVIDER", ""),
            "raw_model_output": raw,
            "parsed_output_json": parsed,
            "generation_error": err,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"{len(rows)} total rows saved | prompt_id={prompt_id} | error={bool(err)}")
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Saved outputs to {out_path}")

if __name__ == "__main__":
    main()
