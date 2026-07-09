# PlanSafeBench-EVM Repository Code Audit Report v1.4.0

## Verdict

This repository is prepared as a reviewer-facing reproducible research package for PlanSafeBench-EVM. The package now contains the corrected v5.6 benchmark, LLM audit artifacts, harmonized counterfactual robustness results, statistical hardening outputs, public expert-validation summary artifacts, and updated citation/licensing metadata.

The main contribution remains a deterministic pre-signing semantic-policy validation framework for AI-generated EVM transaction plans. It is not a newly trained LLM, not a production wallet, and not a claim of independent human ground truth.

## Current reviewer-facing package contents

The package includes:

- importable validator code under `src/plansafebench_evm/`,
- command-line reproduction scripts under `scripts/`,
- smoke tests under `tests/`,
- corrected v5.6 benchmark data under `data/processed/v5_6/`,
- LLM-generated plan audit outputs under `data/llm_outputs/` and `results/llm_audit_v5_6/`,
- harmonized counterfactual scenarios and outputs,
- expert-validation v0.4 public summary artifacts,
- statistical hardening outputs,
- repository metadata and license files.

## Methodological scope statement

The benchmark reference labels encode a deterministic semantic-policy stance. The semantic-policy validator's agreement with those labels should be interpreted as implementation consistency against encoded policy-reference decisions, not as independent superiority over human judgment.

Expert validation is subset-level validation. It supports plausibility and decision-boundary analysis, but it should not be described as full manual annotation of all 1,320 scenarios.

Counterfactual robustness results are controlled stress tests over selected amount-threshold and confirmation-bypass relations, not exhaustive coverage of the full violation taxonomy.

## Privacy and exclusion checks

The public package should not include raw expert response spreadsheets, reviewer-identifying information, local audit snapshots, API keys, `.env` files, or provider account dashboards.

The following local files are intentionally excluded:

```text
*.xlsx
*.zip
metadata_audit_snapshot.txt
results/expert_validation_v0_4/expert_validation_merged_200.csv
results/expert_validation_v0_4/expert_validation_disagreement_cases.csv
```

## Final audit conclusion

The repository is suitable for one final metadata-hardened archival release after these checks pass:

1. `git status --short` shows only planned metadata and safe reproducibility artifact changes.
2. No raw expert spreadsheets, local ZIP bundles, `.env` files, or API credentials are staged.
3. `README.md`, `FILE_INVENTORY.md`, `DATA_LICENSE.md`, `CITATION.cff`, `pyproject.toml`, `data/README.md`, and `results/README.md` are synchronized.
4. The final GitHub release is created once, after all metadata and reproducibility artifacts are committed.
5. The manuscript Data Availability statement cites the final Zenodo version DOI produced by that release.
