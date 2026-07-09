# PlanSafeBench-EVM Reproducible Research Package

This repository contains the reproducible research package for **PlanSafeBench-EVM**, a benchmark and deterministic semantic-policy validation framework for pre-signing safety checks of AI-generated EVM transaction plans.

The package contains controlled benchmark scenarios anchored in real Ethereum transaction templates, a deterministic semantic-policy validator, machine-learning baselines, LLM-generated transaction-plan audit outputs, harmonized counterfactual robustness results, statistical hardening outputs, and public expert-validation summary artifacts.

The main contribution is **not** a newly trained neural model. Machine-learning models are used as baselines, and external LLMs are used as audited planners.

## Formal input

```text
x = (I, P, A, C)
```

where `I` is structured user intent, `P` is private policy, `A` is the AI-generated transaction plan, and `C` is EVM transaction context. The validator computes semantic-policy violation predicates and returns:

```text
delta(x) in {ALLOW, HUMAN_REVIEW, REJECT}
```

The benchmark reference labels encode a specific semantic-policy stance. They should not be described as independent human ground truth.

## Authors

- Astrid Pranadani
- Dhani Ariatmanto

## Citation

Repository:

```text
https://github.com/astridpranadani/plansafebench-evm
```

Zenodo concept DOI:

```text
10.5281/zenodo.21222238
```

Most recent archived version before this metadata-hardening pass:

```text
v1.3.1 - https://doi.org/10.5281/zenodo.21265142
```

After the next final release is created, use that release-specific DOI in the manuscript Data Availability statement.

## Licensing

- Code: MIT License, see `LICENSE`.
- Benchmark data, prompts, controlled scenarios, scored outputs, result tables, and manuscript-supporting artifacts: CC BY 4.0, see `DATA_LICENSE.md`.

## Repository structure

```text
src/plansafebench_evm/                  deterministic semantic-policy validator
scripts/                                reproduction and analysis scripts
data/raw/                               transaction templates, original benchmark scenarios, LLM prompts
data/llm_outputs/                       raw LLM audit outputs
data/processed/v5_6/                    corrected v5.6 benchmark
data/processed/v5_6_counterfactual_harmonized/
results/main_v5_6/                      main benchmark results
results/llm_audit_v5_6/                 corrected LLM audit results
results/counterfactual_v5_6_harmonized/ harmonized counterfactual results
results/expert_validation_v0_4/         public expert-validation summary artifacts
results/v0_3_hardening/                 bootstrap, McNemar, simple baseline, and violation summaries
docs/                                   supporting documentation
examples/                               small sanity-check examples
tests/                                  smoke tests
legacy/                                 provenance scripts
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell without installation:

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

## Reproduce the corrected v5.6 dataset

```bash
PYTHONPATH=src python scripts/correct_dataset_labels.py \
  --input data/raw/benchmark_scenarios/plansafebench_evm_final_scenarios_v4_8.csv \
  --out-dir data/processed/v5_6
```

Main outputs:

```text
data/processed/v5_6/plansafebench_evm_final_scenarios_corrected_v5_6.csv
data/processed/v5_6/plansafebench_evm_final_scenarios_corrected_v5_6.jsonl
data/processed/v5_6/plansafebench_evm_dataset_label_revision_log_v5_6.csv
```

## Run ML baselines and semantic-policy validator

```bash
PYTHONPATH=src python scripts/run_main_experiment.py \
  --scenarios data/processed/v5_6/plansafebench_evm_final_scenarios_corrected_v5_6.csv \
  --out-dir results/main_v5_6
```

Models evaluated: dummy majority, TF-IDF Linear SVM, TF-IDF Logistic Regression, Character TF-IDF Linear SVM, Semantic-policy validator v5.6, and Hybrid semantic override v5.6.

## Re-score LLM-generated transaction plans

```bash
PYTHONPATH=src python scripts/rescore_llm_outputs.py \
  --prompts data/raw/llm_prompts/plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv \
  --outputs Gemini=data/llm_outputs/outputs_openrouter_gemini25flashlite_full1760_resume.csv \
  --outputs Qwen=data/llm_outputs/outputs_openrouter_qwen30b_full1760.csv \
  --outputs Claude=data/llm_outputs/outputs_openrouter_claudehaiku45_full1760.csv \
  --out-dir results/llm_audit_v5_6
```

## Harmonized counterfactual robustness experiment

The harmonized counterfactual experiment is a controlled robustness stress test over amount-threshold and confirmation-bypass relations. It should not be described as full taxonomy coverage.

```bash
PYTHONPATH=src python scripts/run_v0_3_3_harmonized_counterfactual_experiment.py
PYTHONPATH=src python scripts/run_v0_3_4_fast_harmonized_bootstrap.py
```

Outputs are under:

```text
data/processed/v5_6_counterfactual_harmonized/
results/counterfactual_v5_6_harmonized/
```

## Statistical and hardening outputs

Additional reproducibility and hardening outputs are under:

```text
results/v0_3_hardening/
```

These include recomputed main metrics, bootstrap confidence intervals, McNemar tests, simple-rule baseline outputs, prompt-variant summaries, and violation-code frequency summaries.

## Expert-validation summary artifacts

Public expert-validation summary artifacts are under:

```text
results/expert_validation_v0_4/
```

The public release includes aggregate metrics, confusion matrices, majority-support summaries, and a report. Raw expert response spreadsheets, reviewer-identifying information, and private row-level review files are intentionally excluded from the public archival package.

The expert validation is a subset validation study, not full manual annotation of all 1,320 benchmark scenarios.

## Security and ethics notes

This code is for research benchmarking and pre-signing validation experiments. It does not execute blockchain transactions and must not be used as a production wallet security system without independent security review.

User intent, private policy, and AI-generated plan variants are controlled benchmark constructs, not observed private user data.

## Known limitations

- The benchmark reference labels encode a deterministic semantic-policy stance.
- The semantic-policy validator is deterministic rule enforcement, not independent human judgment.
- The expert validation is subset-level validation.
- The benchmark is anchored in selected Ethereum mainnet transaction templates, not all EVM chains.
- The counterfactual robustness experiment targets selected semantic relations, not the entire violation taxonomy.
