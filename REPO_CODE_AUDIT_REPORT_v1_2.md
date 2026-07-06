# PlanSafeBench-EVM Repository Code Audit Report v1.2

## Verdict

The previous `repo_ready_code_package_v1_0` was useful as an internal research bundle, but it was **not ideal for reviewer handoff** because it contained a monolithic v5.6 script with hardcoded `/mnt/data` paths, no dedicated tests, and limited command-line reproducibility.

This v1.2 package is the corrected reviewer-facing code package. It provides an importable module, command-line scripts, smoke tests, example files, metadata, and a documented execution flow.

## Audit findings on v1.0

| Area | Finding | Risk | Fix in v1.2 |
|---|---|---|---|
| Path handling | `analysis/plansafebench_evm_corrected_validator_and_scorer_v5_6.py` used `/mnt/data` | Reviewer cannot reproduce outside original environment | Replaced with CLI scripts using `--input`, `--prompts`, `--outputs`, `--out-dir` |
| Script design | Main v5.6 logic was monolithic | Hard to inspect, test, or rerun partially | Moved reusable logic to `src/plansafebench_evm/semantic_policy.py` |
| Testing | No explicit tests | Reviewer cannot quickly verify validator behavior | Added `tests/test_semantic_policy.py` |
| Documentation | README explained concepts but not full execution | Risk of confusion | Added reviewer quickstart in README and examples |
| Repository metadata | No package metadata | Less suitable as public research code | Added `pyproject.toml`, `LICENSE`, `CITATION.cff`, `.gitignore` |
| Data separation | Data and code responsibilities unclear | Repository may become difficult to maintain | Added `data/README.md`, `results/README.md`, and examples |
| Traceability | Old scripts still needed for provenance | Removing them would reduce audit trail | Moved old scripts to `legacy/` |

## v1.2 files that implement the core computation

- `src/plansafebench_evm/semantic_policy.py`  
  Implements the deterministic semantic-policy validation model, boolean normalization, intent derivation, decision aggregation, and risk-weighted loss.

- `scripts/correct_dataset_labels.py`  
  Applies v5.6 label correction for reverted/uncertain transaction templates.

- `scripts/run_main_experiment.py`  
  Runs ML baselines and the semantic-policy validator on the corrected scenario dataset.

- `scripts/rescore_llm_outputs.py`  
  Re-scores LLM-generated transaction plans with the corrected v5.6 validator.

- `scripts/run_openai_compatible_llm_audit.py`  
  Runs LLM audit prompts through OpenRouter or any OpenAI-compatible chat-completions API.

- `scripts/extract_rpc_templates.py` and `scripts/quality_gate_templates.py`  
  Preserve real Ethereum template extraction and quality-gate steps.

## Tests performed during audit

Syntax check:

```bash
PYTHONPATH=src python -m py_compile $(find src scripts tests legacy -name '*.py')
```

Semantic-policy smoke test:

```bash
PYTHONPATH=src python tests/test_semantic_policy.py
```

Expected output:

```text
All semantic-policy smoke tests passed.
```

Demo pipeline checks were also run with files in `examples/`:

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

Full-data reproduction was also checked against the locally available experiment artifacts. The corrected v1.2 scripts reproduced the v5.6 corrected LLM comparison table and the corrected main-experiment metrics.

## Scope statement for reviewers

This code supports a computer-science/security paper. The main contribution is a deterministic pre-signing semantic-policy validation model, not a newly trained LLM or a production wallet. Machine-learning models are baselines. External LLMs are audited planners.

## Remaining researcher actions before public release

1. Replace placeholder author metadata in `CITATION.cff`.
2. Decide and document dataset license separately from software license.
3. Upload full data and outputs into the `data/` layout or link to a Zenodo/OSF release.
4. Add DOI once archival release is available.
5. Do not commit API keys, `.env` files, or provider usage dashboards.
6. Consider adding continuous integration to run `tests/test_semantic_policy.py` automatically.

## Final audit conclusion

The v1.2 package is suitable as a reviewer-facing code package after author metadata and public data links are filled in. It is substantially stronger than v1.0 because it is portable, testable, and traceable.

## v1.2 metadata update

This package updates repository metadata for reviewer handoff:

- Author metadata set to Astrid Pranadani and Dhani Ariatmanto.
- `CITATION.cff` updated with author names and version 1.2.0.
- `pyproject.toml` updated with author metadata.
- Code license retained as MIT.
- Dataset/artifact licensing clarified separately in `DATA_LICENSE.md` as CC BY 4.0 intended release.
