# PlanSafeBench-EVM Code Package v1.2

This repository contains reproducible research code for **PlanSafeBench-EVM**, a benchmark and deterministic semantic-policy validation framework for pre-signing safety checks of AI-generated EVM transaction plans.

The main contribution is **not** a newly trained neural model. The main computational model is a deterministic security validator:

```text
x = (I, P, A, C)
```

where:

- `I` = structured user intent,
- `P` = private policy,
- `A` = AI-generated transaction plan,
- `C` = EVM transaction context.

The validator computes semantic-policy violation predicates and returns:

```text
δ(x) ∈ {ALLOW, HUMAN_REVIEW, REJECT}
```

Machine-learning models are used as baselines, and external LLMs are used as audited planners, not as trained models in this repository.

## Authors

- Astrid Pranadani
- Dhani Ariatmanto

## Licensing

This repository separates software and data licensing:

- **Code**: MIT License, see `LICENSE`.
- **Dataset, prompts, benchmark scenarios, scored outputs, and result tables**: intended for release under **CC BY 4.0**, see `DATA_LICENSE.md`.

This distinction is important because reviewers and future researchers need to know what they may reuse. Code reuse is governed by the software license, while benchmark/data reuse is governed by the dataset license.

## What changed in v1.2

The earlier v1.0 code package contained the correct research logic, but it was not ideal for reviewer handoff because one script was monolithic and used `/mnt/data` hardcoded paths. Version 1.2 provides:

- importable Python module under `src/plansafebench_evm/`,
- command-line scripts with `--input`, `--out-dir`, and explicit file arguments,
- smoke tests for validator behavior,
- demo input files,
- repository metadata files: `pyproject.toml`, `requirements.txt`, `LICENSE`, `CITATION.cff`, `.gitignore`,
- retained legacy scripts for traceability under `legacy/`.

## Repository structure

```text
plansafebench-evm/
├── src/plansafebench_evm/
│   ├── __init__.py
│   └── semantic_policy.py
├── scripts/
│   ├── correct_dataset_labels.py
│   ├── run_main_experiment.py
│   ├── rescore_llm_outputs.py
│   ├── run_openai_compatible_llm_audit.py
│   ├── extract_rpc_templates.py
│   └── quality_gate_templates.py
├── legacy/
│   ├── data_collection/
│   └── llm_audit/
├── docs/
├── examples/
├── tests/
├── data/
├── results/
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── DATA_LICENSE.md
├── AUTHORS.md
├── CITATION.cff
└── README.md
```

## Installation

Recommended:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

For quick local use without installation, set:

```bash
export PYTHONPATH=$PWD/src
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="$PWD\src"
```

## Run smoke tests

```bash
PYTHONPATH=src python tests/test_semantic_policy.py
```

Expected output:

```text
All semantic-policy smoke tests passed.
```

## Reproduce the corrected dataset label pass

```bash
PYTHONPATH=src python scripts/correct_dataset_labels.py \
  --input data/raw/benchmark_scenarios/plansafebench_evm_final_scenarios_v4_8.csv \
  --out-dir data/processed/v5_6
```

Outputs include:

- `plansafebench_evm_final_scenarios_corrected_v5_6.csv`,
- `plansafebench_evm_final_scenarios_corrected_v5_6.jsonl`,
- `plansafebench_evm_dataset_label_revision_log_v5_6.csv`,
- `plansafebench_evm_corrected_dataset_summary_v5_6.json`.

## Run ML baselines and semantic-policy validator

```bash
PYTHONPATH=src python scripts/run_main_experiment.py \
  --scenarios data/processed/v5_6/plansafebench_evm_final_scenarios_corrected_v5_6.csv \
  --out-dir results/main_v5_6
```

Models evaluated:

- Dummy majority baseline,
- TF-IDF + Linear SVM,
- TF-IDF + Logistic Regression,
- Character TF-IDF + Linear SVM,
- Semantic-policy validator v5.6,
- Hybrid semantic override v5.6.

The ML baselines are not the paper's primary contribution. They test whether controlled benchmark boundaries are learnable. The primary computational model is the deterministic semantic-policy validator.

## Run LLM audit through OpenRouter or OpenAI-compatible API

Set API environment variables first:

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://openrouter.ai/api/v1"
export LLM_MODEL="anthropic/claude-haiku-4.5"
export LLM_PROVIDER="openrouter"
export LLM_MODEL_SLOT="cheap_frontier_proprietary"
```

Windows PowerShell:

```powershell
$env:LLM_API_KEY="..."
$env:LLM_API_BASE="https://openrouter.ai/api/v1"
$env:LLM_MODEL="anthropic/claude-haiku-4.5"
$env:LLM_PROVIDER="openrouter"
$env:LLM_MODEL_SLOT="cheap_frontier_proprietary"
```

Smoke test:

```bash
python scripts/run_openai_compatible_llm_audit.py \
  --prompts data/raw/llm_prompts/plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv \
  --out data/llm_outputs/outputs_openrouter_claudehaiku45_smoke.csv \
  --limit 5
```

Full run:

```bash
python scripts/run_openai_compatible_llm_audit.py \
  --prompts data/raw/llm_prompts/plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv \
  --out data/llm_outputs/outputs_openrouter_claudehaiku45_full1760.csv
```

Never commit API keys, `.env` files, or provider account information.

## Rescore LLM outputs with corrected v5.6 validator

```bash
PYTHONPATH=src python scripts/rescore_llm_outputs.py \
  --prompts data/raw/llm_prompts/plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv \
  --outputs Gemini=data/llm_outputs/outputs_openrouter_gemini25flashlite_full1760_resume.csv \
  --outputs Qwen=data/llm_outputs/outputs_openrouter_qwen30b_full1760.csv \
  --outputs Claude=data/llm_outputs/outputs_openrouter_claudehaiku45_full1760.csv \
  --out-dir results/llm_audit_v5_6
```

Outputs include overall model comparison, per-variant metrics, per-action-type metrics, confusion matrices, unsafe ALLOW cases, and critical unsafe ALLOW cases.

## Demo commands

The `examples/` folder includes small demonstration files so reviewers can verify that the scripts run without downloading the full dataset.

```bash
PYTHONPATH=src python scripts/correct_dataset_labels.py \
  --input examples/demo_scenarios_v4_8.csv \
  --out-dir results/demo_corrected

PYTHONPATH=src python scripts/run_main_experiment.py \
  --scenarios results/demo_corrected/plansafebench_evm_final_scenarios_corrected_v5_6.csv \
  --out-dir results/demo_main

PYTHONPATH=src python scripts/rescore_llm_outputs.py \
  --prompts examples/demo_prompts_v5_3.csv \
  --outputs ClaudeDemo=examples/demo_llm_outputs_claude_smoke.csv \
  --out-dir results/demo_llm
```

The demo is only for code execution sanity checks. It is not the paper's full experimental result.

## Data required for full reproduction

The full repository should include or link to:

```text
data/raw/transaction_templates/plansafebench_evm_transaction_templates_realworld_level2_final_v4_7.csv
data/raw/benchmark_scenarios/plansafebench_evm_final_scenarios_v4_8.csv
data/raw/llm_prompts/plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv
data/llm_outputs/outputs_openrouter_gemini25flashlite_full1760_resume.csv
data/llm_outputs/outputs_openrouter_qwen30b_full1760.csv
data/llm_outputs/outputs_openrouter_claudehaiku45_full1760.csv
```

Recommended archival release: GitHub repository + Zenodo DOI or OSF release.

## Security and ethics notes

This code is for research benchmarking and pre-signing validation experiments. It does not execute blockchain transactions and must not be used as a production wallet security system without independent security review.

## Known limitations

- User intent and private policy are controlled benchmark constructs, not observed real-user data.
- The semantic-policy validator is deterministic rule enforcement, not independent human judgment.
- The benchmark is anchored in selected Ethereum mainnet transaction templates, not all EVM chains.
- Full multi-expert row-level annotation is not included in this code package.
