# PlanSafeBench-EVM LLM-Generated Transaction Plan Audit v5.3

This package prepares the LLM-generated plan audit.

## Files

- `plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.csv`
- `plansafebench_evm_llm_generated_plan_audit_prompts_v5_3.jsonl`
- `plansafebench_evm_llm_generated_outputs_template_v5_3.csv`
- `plansafebench_evm_llm_model_manifest_template_v5_3.csv`
- `plansafebench_evm_llm_audit_sample_size_justification_v5_3.json`
- `plansafebench_evm_llm_audit_scorer_v5_3.py`
- `plansafebench_evm_llm_generated_plan_audit_runbook_v5_3.md`

## Design

- 220 Level-2 real Ethereum transaction templates.
- 8 prompt variants per template.
- 1,760 prompts per LLM.
- 3 recommended LLMs.
- 5,280 generated plans total if fully executed.

The sample size is justified using 95% confidence interval precision for binomial proportions, not by arbitrary guessing.
