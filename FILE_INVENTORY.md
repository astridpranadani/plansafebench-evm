# File inventory

This file summarizes the reviewer-facing repository contents for PlanSafeBench-EVM.

## Core package

```text
src/plansafebench_evm/__init__.py
src/plansafebench_evm/semantic_policy.py
tests/test_semantic_policy.py
requirements.txt
pyproject.toml
```

## Reproduction and analysis scripts

```text
scripts/correct_dataset_labels.py
scripts/run_main_experiment.py
scripts/rescore_llm_outputs.py
scripts/run_openai_compatible_llm_audit.py
scripts/extract_rpc_templates.py
scripts/quality_gate_templates.py
scripts/run_v0_3_1_main_stats_longformat.py
scripts/run_v0_3_hardening_analysis.py
scripts/run_v0_3_2_counterfactual_experiment.py
scripts/run_v0_3_2b_counterfactual_experiment.py
scripts/run_v0_3_3_harmonized_counterfactual_experiment.py
scripts/run_v0_3_4_fast_harmonized_bootstrap.py
scripts/run_v0_4_expert_validation_analysis.py
scripts/prepare_expert2_validation_form.py
scripts/prepare_expert2_validation_form_FINAL.py
```

## Public data directories

```text
data/raw/transaction_templates/
data/raw/benchmark_scenarios/
data/raw/llm_prompts/
data/llm_outputs/
data/processed/v5_6/
data/processed/v5_6_counterfactual_harmonized/
```

## Public result directories

```text
results/main_v5_6/
results/llm_audit_v5_6/
results/counterfactual_v5_6_harmonized/
results/expert_validation_v0_4/
results/v0_3_hardening/
```

## Documentation

```text
README.md
DATA_LICENSE.md
LICENSE
CITATION.cff
AUTHORS.md
FILE_INVENTORY.md
REPO_CODE_AUDIT_REPORT_v1_4_0.md
docs/
examples/
legacy/
```

## Intentionally excluded from public release

```text
*.xlsx
*.zip
metadata_audit_snapshot.txt
results/expert_validation_v0_4/expert_validation_merged_200.csv
results/expert_validation_v0_4/expert_validation_disagreement_cases.csv
```

These files may exist locally during analysis, but they should not be committed or included in the public release because they may contain raw expert responses, reviewer-identifying information, local audit snapshots, or non-canonical compressed copies.
