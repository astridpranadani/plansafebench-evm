# Formal Semantic-Policy Validation Model v5.6

Let x = (I, P, A, C), where I is structured user intent, P is the private policy, A is the AI-generated transaction plan, and C is the EVM transaction context.

The validator computes violation predicates mapped to semantic-policy violations. The decision function δ(x) returns one of {ALLOW, HUMAN_REVIEW, REJECT} using this aggregation rule:

1. If a critical violation or hard policy violation is detected, δ(x) = REJECT.
2. Else, if ambiguity, review-threshold exposure, or context uncertainty is detected, δ(x) = HUMAN_REVIEW.
3. Else, δ(x) = ALLOW.

Primary security metrics are unsafe_allow_rate, critical_unsafe_allow_rate, human_review_load, over_reject_rate, and risk_weighted_loss. Risk-weighted loss gives higher penalty to unsafe ALLOW decisions than to conservative review/rejection decisions.
