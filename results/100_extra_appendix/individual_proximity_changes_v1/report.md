# Individual proximity changes after GRACE--Wachter refinement

For each of the128 original held-out factuals, compute
`delta_i = ||GRACE_assigned_i - factual_i||_2 - ||Wachter_Base_i - factual_i||_2`.
This uses the saved minimum-cost one-to-one assignment and original standardized
feature-space L2 metric. Positive changes mean greater factual-to-CF distance;
negative changes mean improved proximity. Statistics include all128 cases, not
only positive changes. The95th percentile uses linear interpolation. These are
quantiles of paired changes, not differences between before/after quantiles.

| dataset   | classifier   |   n |   before_mean |   after_mean |   delta_median |   delta_p95 |   delta_max |
|:----------|:-------------|----:|--------------:|-------------:|---------------:|------------:|------------:|
| heloc     | mlp          | 128 |        0.9912 |       1.2417 |         0.2576 |      0.3174 |      0.3629 |
| heloc     | logistic     | 128 |        1.6249 |       2.0043 |         0.3764 |      0.5231 |      0.6509 |
| fico      | mlp          | 128 |        1.0857 |       1.5099 |         0.4149 |      0.5459 |      0.6865 |
| fico      | logistic     | 128 |        1.7662 |       2.0912 |         0.3292 |      0.3743 |      0.4233 |

The original calibrated settings are retained. FICO headline rows are relaxed
categorical outputs. Hard realizations are separate below; FICO/logistic hard
realization violates geometry and is not a feasible constrained result.

| dataset   | classifier   |   n |   before_mean |   after_mean |   delta_median |   delta_p95 |   delta_max | status                      |
|:----------|:-------------|----:|--------------:|-------------:|---------------:|------------:|------------:|:----------------------------|
| fico      | mlp          | 128 |        1.0857 |       1.5018 |         0.4092 |      0.5326 |      0.6661 | executed                    |
| fico      | logistic     | 128 |        1.7662 |       2.0901 |         0.3275 |      0.3738 |      0.4216 | infeasible_hard_realization |

`summary.csv` also contains the full before/after marginal summaries, signed
mean/minimum changes, and counts of increases/decreases. `per_factual.csv`
contains the matched source/support identities and every paired change.
Raw files are read-only; no training, optimization, or historical output was rerun.
