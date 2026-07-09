# Dataset and Artifact License

This repository uses separate licenses for software code and research data/artifacts.

## Software code

All Python source code, scripts, and software documentation are released under the MIT License. See `LICENSE`.

## Dataset, prompts, experiment outputs, and tables

Unless otherwise stated in a specific data file, the PlanSafeBench-EVM benchmark data, controlled scenarios, prompt sets, scored outputs, result tables, and manuscript-supporting artifacts are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Recommended attribution:

> Pranadani, A., & Ariatmanto, D. PlanSafeBench-EVM: Semantic-Policy Validation for AI-Generated EVM Transaction Plans. Dataset and code release, 2026. Zenodo concept DOI: 10.5281/zenodo.21222238.

For manuscript submission, cite the specific Zenodo version DOI corresponding to the archived release used in the submitted manuscript.

## Important data notes

- Public Ethereum transaction hashes and on-chain metadata remain public blockchain identifiers.
- User intent, private policy, and AI-generated plan variants are controlled benchmark constructs, not observed private user data.
- Public expert-validation artifacts are aggregate or summary-level artifacts. Raw expert response spreadsheets and reviewer-identifying information are intentionally excluded.
- API keys, provider account information, `.env` files, and private credentials must never be committed.
