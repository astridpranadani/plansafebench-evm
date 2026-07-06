# PlanSafeBench-EVM LLM Audit Execution Toolkit v5.4

## Purpose

This toolkit turns the v5.3 LLM-generated plan audit design into an executable pipeline.

It does **not** fabricate LLM outputs. You must run the prompts against real LLMs and store the outputs.

## Dataset

- Prompt file: `plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv`
- Prompt count per model: 1760
- Recommended models: 3
- Total generated plans if complete: 5280

## Step 1 — Install dependency

```powershell
python -m pip install pandas requests
```

## Step 2 — Split prompts into batches

```powershell
python plansafebench_evm_llm_audit_batch_splitter_v5_4.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --out-dir batches --batch-size 100
```

## Step 3 — Smoke test first, do not run full audit yet

Run only 5 prompts first.

### OpenAI-compatible endpoint

```powershell
$env:LLM_API_KEY="ISI_API_KEY"
$env:LLM_API_BASE="https://api.openai.com/v1"
$env:LLM_MODEL="ISI_MODEL_NAME"
$env:LLM_PROVIDER="ISI_PROVIDER"
$env:LLM_MODEL_SLOT="frontier_proprietary"

python plansafebench_evm_llm_audit_openai_compatible_runner_v5_4.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --out outputs_frontier_smoke.csv --limit 5
```

### Ollama local model

```powershell
$env:OLLAMA_MODEL="ISI_MODEL_OLLAMA"
$env:LLM_MODEL_SLOT="open_weight_or_local"

python plansafebench_evm_llm_audit_ollama_runner_v5_4.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --out outputs_ollama_smoke.csv --limit 5
```

## Step 4 — Score smoke test

```powershell
python plansafebench_evm_llm_audit_scorer_v5_3.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --outputs outputs_frontier_smoke.csv --out-scored scored_frontier_smoke.csv --out-summary summary_frontier_smoke.json
```

Check that:

- valid JSON rate is acceptable.
- planner_decision exists.
- parsed output JSON is not empty.
- no systemic failure occurs.

## Step 5 — Run full model audit

After smoke test passes, run full prompts per model.

```powershell
python plansafebench_evm_llm_audit_openai_compatible_runner_v5_4.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --out outputs_frontier_full.csv --sleep 0.2
```

Repeat for other models by changing environment variables and output filename.

## Step 6 — Merge outputs

```powershell
python plansafebench_evm_llm_audit_merge_outputs_v5_4.py --inputs outputs_frontier_full.csv outputs_cost_full.csv outputs_open_full.csv --out outputs_all_models_v5_4.csv
```

## Step 7 — Score all outputs

```powershell
python plansafebench_evm_llm_audit_scorer_v5_3.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --outputs outputs_all_models_v5_4.csv --out-scored scored_all_models_v5_4.csv --out-summary summary_all_models_v5_4.json
```

## Step 8 — Quality gate

```powershell
python plansafebench_evm_llm_audit_quality_gate_v5_4.py --prompts plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv --outputs outputs_all_models_v5_4.csv --scored scored_all_models_v5_4.csv --out-report quality_gate_all_models_v5_4.json
```

## Step 9 — Generate report

```powershell
python plansafebench_evm_llm_audit_report_generator_v5_4.py --scored scored_all_models_v5_4.csv --summary-json summary_all_models_v5_4.json --out-md llm_generated_plan_audit_results_v5_4.md
```

## Rule for manuscript use

Do not include LLM audit results in the paper until:

1. Coverage is complete or clearly disclosed.
2. Model names and versions are documented.
3. Prompt template and decoding settings are reported.
4. No fabricated output is used.
5. Scored outputs and summary are archived.
