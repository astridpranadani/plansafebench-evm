# PlanSafeBench-EVM v0.4 Expert Validation Report

## Scope
This report analyzes the 200-scenario expert-validation subset. Expert 2 conducted a reference-decision review, while Experts 3 and 4 conducted fully blind reviews of the same 200 scenarios.

## Quality checks
- expert2_rows: 200 (PASS)
- expert3_rows: 200 (PASS)
- expert4_rows: 200 (PASS)
- common_scenario_ids: 200 (PASS)
- missing_from_expert2: 0 (PASS)
- missing_from_expert3: 0 (PASS)
- missing_from_expert4: 0 (PASS)
- invalid_benchmark_labels: 0 (PASS)
- invalid_expert2_labels: 0 (PASS)
- invalid_expert3_labels: 0 (PASS)
- invalid_expert4_labels: 0 (PASS)
- critical_cases: 70 (PASS)
- expert_consensus_differs_from_benchmark: 0 (PASS)

## Agreement metrics
| Comparison | n | Agreement | Agreement % | Cohen's kappa |
|---|---:|---:|---:|---:|
| Benchmark vs Expert 2 | 200 | 165 | 82.50% | 0.7419 |
| Benchmark vs Expert 3 | 200 | 174 | 87.00% | 0.8067 |
| Benchmark vs Expert 4 | 200 | 200 | 100.00% | 1.0 |
| Expert 2 vs Expert 3 | 200 | 139 | 69.50% | 0.53 |
| Expert 2 vs Expert 4 | 200 | 165 | 82.50% | 0.7419 |
| Expert 3 vs Expert 4 | 200 | 174 | 87.00% | 0.8067 |

## Majority support
- all_three_experts_support_benchmark: 139 (69.50%)
- at_least_two_experts_support_benchmark: 200 (100.00%)
- expert_consensus_differs_from_benchmark: 0 (0.00%)
- all_reject_cases_supported_by_all_experts: 75 / 75
- all_critical_cases_supported_by_all_experts: 70 / 70

## Main interpretation
- The benchmark reference labels should not be automatically relabeled from these expert results.
- All benchmark REJECT cases and all critical-risk cases were supported by all experts.
- Disagreements are confined to the ALLOW vs HUMAN_REVIEW boundary; no expert downgraded a REJECT or critical case to ALLOW.
- Expert 4, a fresh fully blind reviewer, reproduced the benchmark reference decisions on all 200 scenarios.
- The remaining disagreement pattern should be reported as boundary-case ambiguity, not as a clear benchmark-labeling error.

## Recommended manuscript wording
> Expert validation was conducted on a stratified 200-scenario subset. Expert 2 performed a reference-decision review, while Experts 3 and 4 independently reviewed the same 200 scenarios in fully blind settings. Expert review supported all REJECT and critical-risk cases, while disagreements were confined to the conservative ALLOW-HUMAN_REVIEW boundary. No benchmark label was automatically changed after expert review; disagreements were analyzed as boundary cases.