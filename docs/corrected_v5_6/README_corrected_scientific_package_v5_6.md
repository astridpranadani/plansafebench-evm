# PlanSafeBench-EVM Corrected Scientific Package v5.6

This package corrects methodological issues identified during JISA readiness audit.

## Corrections applied

1. Reverted-template ALLOW labels were corrected to HUMAN_REVIEW when the policy disallows unreveiwed reverted templates.

2. LLM scoring now normalizes boolean strings such as `"false"` correctly.

3. LLM scoring now derives a structured user-intent representation from the prompt and checks intent-policy-plan-context relations.

4. Risk-weighted safety loss was added to emphasize unsafe ALLOW decisions over conservative decisions.

5. Main benchmark baselines and LLM audit summaries were regenerated under v5.6 rules.


## Corrected dataset summary

- Total scenarios: 1320

- Label corrections: 62

- Expected decision distribution after correction: {'HUMAN_REVIEW': 502, 'REJECT': 440, 'ALLOW': 378}


## Main experiment summary

| model                          |   n_test |   accuracy |   macro_f1 |   weighted_f1 |   unsafe_allow_count |   unsafe_allow_rate_total |   critical_unsafe_allow_count |   critical_unsafe_allow_rate_total |   human_review_load |   over_reject_count |   over_reject_rate_total |   risk_weighted_loss_mean |   risk_weighted_loss_total |
|:-------------------------------|---------:|-----------:|-----------:|--------------:|---------------------:|--------------------------:|------------------------------:|-----------------------------------:|--------------------:|--------------------:|-------------------------:|--------------------------:|---------------------------:|
| Dummy majority baseline        |      264 |   0.378788 |   0.18315  |      0.208125 |                    0 |                 0         |                             0 |                                  0 |            1        |                   0 |                        0 |                 0.143939  |                         38 |
| TF-IDF Linear SVM              |      264 |   1        |   1        |      1        |                    0 |                 0         |                             0 |                                  0 |            0.378788 |                   0 |                        0 |                 0         |                          0 |
| TF-IDF Logistic Regression     |      264 |   0.988636 |   0.988472 |      0.98866  |                    3 |                 0.0113636 |                             0 |                                  0 |            0.367424 |                   0 |                        0 |                 0.0340909 |                          9 |
| Char TF-IDF Linear SVM         |      264 |   0.981061 |   0.980837 |      0.981119 |                    5 |                 0.0189394 |                             0 |                                  0 |            0.359848 |                   0 |                        0 |                 0.0568182 |                         15 |
| Semantic-policy validator v5.6 |      264 |   1        |   1        |      1        |                    0 |                 0         |                             0 |                                  0 |            0.378788 |                   0 |                        0 |                 0         |                          0 |
| Hybrid semantic override v5.6  |      264 |   1        |   1        |      1        |                    0 |                 0         |                             0 |                                  0 |            0.378788 |                   0 |                        0 |                 0         |                          0 |


## Corrected LLM audit summary

| model_slot                 | model_name                       |    n |   valid_json_rate |   validator_allow_rate |   validator_human_review_rate |   validator_reject_rate |   policy_violation_rate |   critical_violation_rate |   unsafe_allow_count |   unsafe_allow_rate |   critical_unsafe_allow_count |   critical_unsafe_allow_rate |   planner_validator_agreement_rate |   human_review_load |   planner_reject_rate |   planner_allow_rate |   risk_weighted_loss_mean |   risk_weighted_loss_total |
|:---------------------------|:---------------------------------|-----:|------------------:|-----------------------:|------------------------------:|------------------------:|------------------------:|--------------------------:|---------------------:|--------------------:|------------------------------:|-----------------------------:|-----------------------------------:|--------------------:|----------------------:|---------------------:|--------------------------:|---------------------------:|
| cheap_frontier_proprietary | anthropic/claude-haiku-4.5       | 1760 |                 1 |               0.321023 |                      0.302841 |                0.376136 |                0.678977 |                  0.126136 |                   15 |          0.00852273 |                             0 |                   0          |                           0.340341 |            0.945455 |            0.0215909  |            0.0329545 |                  0.173864 |                        306 |
| open_weight_instruct       | qwen/qwen3-30b-a3b-instruct-2507 | 1760 |                 1 |               0.314773 |                      0.302841 |                0.382386 |                0.685227 |                  0.132386 |                  310 |          0.176136   |                            51 |                   0.0289773  |                           0.375568 |            0.617045 |            0.00227273 |            0.380682  |                  0.825568 |                       1453 |
| cost_efficient_proprietary | google/gemini-2.5-flash-lite     | 1760 |                 1 |               0.225    |                      0.286932 |                0.488068 |                0.775    |                  0.238636 |                  327 |          0.185795   |                             4 |                   0.00227273 |                           0.626705 |            0.294886 |            0.305114   |            0.4       |                  0.653977 |                       1151 |


## JISA positioning

The corrected contribution should be framed as a computational security validation framework for pre-signing AI-generated EVM transaction plans. ML classifiers are baselines; the semantic-policy model and LLM audit are the main security contributions.
