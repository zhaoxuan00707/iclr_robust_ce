# Individual proximity changes after GRACE--Wachter refinement

For each original held-out factual (cohort sizes shown below), compute
`delta_i = ||GRACE_assigned_i - factual_i||_2 - ||Wachter_Base_i - factual_i||_2`.
This uses the saved minimum-cost one-to-one assignment and original standardized
feature-space L2 metric. Positive changes mean greater factual-to-CF distance;
negative changes mean improved proximity. Statistics include all cases, not
only positive changes. The95th percentile uses linear interpolation. These are
quantiles of paired changes, not differences between before/after quantiles.

| dataset   | classifier   |   n |   before_mean |   after_mean |   delta_median |   delta_p95 |   delta_max |
|:----------|:-------------|----:|--------------:|-------------:|---------------:|------------:|------------:|
| diabetes  | mlp          |  50 |        1.1025 |       1.8353 |         0.7125 |      0.8967 |      1.0674 |
| diabetes  | logistic     |  47 |        0.9276 |       1.4052 |         0.4744 |      0.5525 |      0.6704 |
| german    | mlp          |  37 |        1.9980 |       2.5881 |         0.6515 |      1.0446 |      1.3370 |
| german    | logistic     |  43 |        1.7695 |       2.0994 |         0.3543 |      0.5436 |      0.5757 |

The original calibrated settings are retained. German/FICO headline rows are relaxed
categorical outputs. Hard realizations are separate below; FICO/logistic hard
realization, if included, violates geometry and is not a feasible constrained result.

| dataset   | classifier   |   n |   before_mean |   after_mean |   delta_median |   delta_p95 |   delta_max | status   |
|:----------|:-------------|----:|--------------:|-------------:|---------------:|------------:|------------:|:---------|
| german    | mlp          |  37 |        1.9980 |       2.4314 |         0.5036 |      0.6603 |      0.9974 | executed |
| german    | logistic     |  43 |        1.7695 |       1.9971 |         0.2568 |      0.3322 |      0.3461 | executed |

`summary.csv` also contains the full before/after marginal summaries, signed
mean/minimum changes, and counts of increases/decreases. `per_factual.csv`
contains the matched source/support identities and every paired change.
Raw files are read-only; no training, optimization, or historical output was rerun.
