#!/usr/bin/env python3
"""
PlanSafeBench-EVM Q1 API/RPC extractor v4.4

Purpose:
Upgrade Level-0 web-observed transaction hashes to Level-1/2 transaction templates.

Required:
- EVM_RPC_URL environment variable or --rpc-url.
- Optional ETHERSCAN_API_KEY for functionName/methodId cross-checks.

Outputs:
- JSONL transaction templates with transaction + receipt fields.
"""

from __future__ import annotations
import argparse, json, os, time, requests, re
from pathlib import Path
from typing import Any

TX_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")

def rpc_call(rpc_url: str, method: str, params: list[Any], request_id: int) -> Any:
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
    r = requests.post(rpc_url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"RPC error for {method}: {data['error']}")
    return data.get("result")

def hex_to_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(str(x), 16)
    except Exception:
        return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rpc-url", default=os.environ.get("EVM_RPC_URL"))
    ap.add_argument("--chain", default="ethereum")
    ap.add_argument("--chain-id", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    if not args.rpc_url:
        raise SystemExit("Missing --rpc-url or EVM_RPC_URL")
    hashes = [x.strip().lower() for x in Path(args.hashes).read_text().splitlines() if x.strip()]
    bad = [h for h in hashes if not TX_RE.match(h)]
    if bad:
        raise SystemExit(f"Invalid transaction hashes: {bad[:5]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i, txh in enumerate(hashes, start=1):
            try:
                tx = rpc_call(args.rpc_url, "eth_getTransactionByHash", [txh], i * 2)
                receipt = rpc_call(args.rpc_url, "eth_getTransactionReceipt", [txh], i * 2 + 1)
                if tx is None:
                    template = {"transaction_hash": txh, "extraction_status": "missing_transaction"}
                else:
                    input_data = tx.get("input") or ""
                    method_selector = input_data[:10] if isinstance(input_data, str) and len(input_data) >= 10 else None
                    template = {
                        "template_id": f"rw_api_v4_4_{i:06d}",
                        "transaction_hash": txh,
                        "chain": args.chain,
                        "chain_id": args.chain_id,
                        "block_hash": tx.get("blockHash"),
                        "block_number": hex_to_int(tx.get("blockNumber")),
                        "transaction_index": hex_to_int(tx.get("transactionIndex")),
                        "from_address": tx.get("from"),
                        "to_address": tx.get("to"),
                        "value_wei_hex": tx.get("value"),
                        "value_wei": hex_to_int(tx.get("value")),
                        "gas_hex": tx.get("gas"),
                        "gas": hex_to_int(tx.get("gas")),
                        "gas_price_hex": tx.get("gasPrice"),
                        "gas_price": hex_to_int(tx.get("gasPrice")),
                        "input": input_data,
                        "input_length": len(input_data) if isinstance(input_data, str) else None,
                        "method_selector": method_selector,
                        "receipt_status_hex": receipt.get("status") if receipt else None,
                        "receipt_status": hex_to_int(receipt.get("status")) if receipt else None,
                        "gas_used": hex_to_int(receipt.get("gasUsed")) if receipt else None,
                        "cumulative_gas_used": hex_to_int(receipt.get("cumulativeGasUsed")) if receipt else None,
                        "logs_count": len(receipt.get("logs", [])) if receipt else None,
                        "receipt_available": receipt is not None,
                        "extraction_status": "ok",
                        "extraction_level": "level_2_rpc_transaction_and_receipt" if receipt else "level_1_rpc_transaction",
                    }
                f.write(json.dumps(template, ensure_ascii=False) + "\n")
            except Exception as e:
                f.write(json.dumps({"transaction_hash": txh, "extraction_status": "error", "error": str(e)}, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()
