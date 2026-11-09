# Task100 Experiment3: single-seed diagnostic reconstruction

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1`. Seed0, eight A100 GPUs, one worker per dataset/classifier.
All 726 independent checks pass. Train/cal/test split identities,
calibration selection, complete test updates, geometry, assignments and metrics
are independently verified. Classifier FV is reported over all factuals. Missing
or infeasible constructions are failures; inspect `status`, `native_failures`,
`geometry_feasible`, `hard_category_violations` and `failure_aware_FV` alongside FV.
Unsupported categorical configurations have no usable performance claim.

This is **not an exact Table1 reproduction**: original four-dataset run provenance
was untraceable. The declared reconstruction reuses the loader, MLP and logistic
training conventions, official baseline adapters and revised update-aware GRACE.
HELOC and mixed FICO are representations of the same corpus; their loaded versions
have different row/feature counts, not independent benchmark populations.

A.5 categorical uncertainty uses logit softmax with temperature1 and nominal
smoothing1e-4; numerical coordinates are additive. Supports use a weighted simplex
projection in standardized coordinates. `relaxed` scores are not legal discrete
recourse. `hard` is a separate argmax-realization diagnostic with geometry checked
again, not a post-hoc validity correction. No protected/immutable constraints or
causal actionability are established.

45 calibration and45 test scenarios use mean/scale/tail shifts, severities
.5/1/2.5 and five realizations. Full10-step SGD(.01) updates start from the same
fitted classifier. Tail sampling keeps labels paired with rows. Mean signs and
scale transformations can repeat across seeds: independent_checks.csv records
unique/checkpoint overlap counts. Do not describe these as45 independent novel
shift distributions or as complete retraining robustness.

Calibrated GRACE uses r/b_ref in{0,.25,1,3}, beta in{1.05,1.1,1.2}, up to800
projected augmented-Lagrangian steps, primal1, rho1, temperature.05; the final
iterate is returned and infeasibility is retained. A small successive-objective
change is not a stationarity/global-optimality guarantee. All selections use
calibration FV with cost and smaller-parameter tie-breaks; test scores never tune.

The retained RobustX STCE searches label-negative training examples despite its
positive-candidate docstring; this inherited behavior is disclosed rather than
silently edited. ROAR/PROPLACE mixed-domain constraints are unsupported. Gurobi
license failures/timeouts and returned invalid candidates remain visible. ElliCE
can return a valid yet non-robust point under its native default; inspect the
separately audited `native_ellipsoid_robust_fraction`. The German/logistic
calibration-feasibility-gated follow-up is saved in results/101_ellice_convergence
and does not replace this original protocol. DiCE is
official0.12 random generation, four candidates,5000 samples, minimum valid L2
selection; no handmade replacement. Dependency and source hashes are recorded.

| dataset   | classifier   | method        | realization   |   n |         FV |      Cost | status                              |   native_failures |
|:----------|:-------------|:--------------|:--------------|----:|-----------:|----------:|:------------------------------------|------------------:|
| diabetes  | logistic     | wachter       | native        |  47 | 0.851064   |  0.92762  | executed                            |                 0 |
| diabetes  | logistic     | grace_wachter | native        |  47 | 1          |  1.40519  | executed                            |                 0 |
| diabetes  | logistic     | trex          | native        |  47 | 1          |  2.65178  | executed                            |                 0 |
| diabetes  | logistic     | roar          | native        |  47 | 0.914421   |  1.9408   | executed                            |                 0 |
| diabetes  | logistic     | proplace      | native        |  47 | 1          |  2.31609  | executed                            |                 0 |
| diabetes  | logistic     | ellice        | native        |  47 | 1          |  2.03131  | executed                            |                 0 |
| diabetes  | mlp          | wachter       | native        |  50 | 0.732889   |  1.10247  | executed                            |                 0 |
| diabetes  | mlp          | grace_wachter | native        |  50 | 0.999556   |  1.83528  | executed                            |                 0 |
| diabetes  | mlp          | trex          | native        |  50 | 1          |  3.16073  | executed                            |                 0 |
| diabetes  | mlp          | roar          | native        |  50 | 0.785778   |  2.12951  | executed                            |                 0 |
| diabetes  | mlp          | proplace      | native        |  50 | 0.0235556  |  0        | infrastructure_failure              |                50 |
| diabetes  | mlp          | ellice        | native        |  50 | 1          |  2.47562  | executed                            |                 0 |
| diabetes  | mlp          | dice          | native        |  50 | 0.949778   |  2.78529  | executed                            |                 0 |
| diabetes  | mlp          | grace_dice    | native        |  50 | 0.999111   |  2.95825  | executed                            |                 0 |
| fico      | logistic     | wachter       | native        | 128 | 0.843403   |  1.76621  | executed                            |                 0 |
| fico      | logistic     | grace_wachter | relaxed       | 128 | 0.989062   |  2.09118  | executed                            |                 0 |
| fico      | logistic     | grace_wachter | hard          | 128 | 0.988715   |  2.09012  | infeasible_hard_realization         |                 0 |
| fico      | logistic     | trex          | native        | 128 | 1          |  4.88844  | executed                            |                 0 |
| fico      | logistic     | roar          | native        | 128 | 0.00729167 |  0        | unsupported_categorical_constraints |               128 |
| fico      | logistic     | proplace      | native        | 128 | 0.00729167 |  0        | unsupported_categorical_constraints |               128 |
| fico      | logistic     | ellice        | native        | 128 | 1          |  6.10967  | executed                            |                 0 |
| fico      | mlp          | wachter       | native        | 128 | 0.793056   |  1.08565  | executed                            |                 0 |
| fico      | mlp          | grace_wachter | relaxed       | 128 | 1          |  1.50987  | executed                            |                 0 |
| fico      | mlp          | grace_wachter | hard          | 128 | 0.999306   |  1.50176  | executed                            |                 0 |
| fico      | mlp          | trex          | native        | 128 | 0.963368   |  4.71291  | executed                            |                 5 |
| fico      | mlp          | roar          | native        | 128 | 0.0125     |  0        | unsupported_categorical_constraints |               128 |
| fico      | mlp          | proplace      | native        | 128 | 0.0125     |  0        | unsupported_categorical_constraints |               128 |
| fico      | mlp          | ellice        | native        | 128 | 1          |  6.21397  | executed                            |                 0 |
| fico      | mlp          | dice          | native        | 128 | 0.949306   |  5.66522  | executed                            |                 0 |
| fico      | mlp          | grace_dice    | relaxed       | 128 | 1          |  5.72991  | executed                            |                 0 |
| fico      | mlp          | grace_dice    | hard          | 128 | 1          |  5.72835  | executed                            |                 0 |
| german    | logistic     | wachter       | native        |  43 | 0.596382   |  1.76949  | executed                            |                 0 |
| german    | logistic     | grace_wachter | relaxed       |  43 | 0.993282   |  2.09937  | executed                            |                 0 |
| german    | logistic     | grace_wachter | hard          |  43 | 0.911628   |  1.99706  | executed                            |                 0 |
| german    | logistic     | trex          | native        |  43 | 1          |  8.78844  | executed                            |                 0 |
| german    | logistic     | roar          | native        |  43 | 0.0501292  |  0        | unsupported_categorical_constraints |                43 |
| german    | logistic     | proplace      | native        |  43 | 0.0501292  |  0        | unsupported_categorical_constraints |                43 |
| german    | logistic     | ellice        | native        |  43 | 1          | 10.4201   | executed                            |                 0 |
| german    | mlp          | wachter       | native        |  37 | 0.586787   |  1.99802  | executed                            |                 0 |
| german    | mlp          | grace_wachter | relaxed       |  37 | 1          |  2.58813  | executed                            |                 0 |
| german    | mlp          | grace_wachter | hard          |  37 | 0.965766   |  2.43137  | executed                            |                 0 |
| german    | mlp          | trex          | native        |  37 | 1          |  9.12563  | executed                            |                 0 |
| german    | mlp          | roar          | native        |  37 | 0.0546547  |  0        | unsupported_categorical_constraints |                37 |
| german    | mlp          | proplace      | native        |  37 | 0.0546547  |  0        | unsupported_categorical_constraints |                37 |
| german    | mlp          | ellice        | native        |  37 | 1          | 10.8253   | executed                            |                 0 |
| german    | mlp          | dice          | native        |  37 | 0.882883   |  3.85114  | executed                            |                 0 |
| german    | mlp          | grace_dice    | relaxed       |  37 | 0.999399   |  3.9567   | executed                            |                 0 |
| german    | mlp          | grace_dice    | hard          |  37 | 0.98018    |  3.92896  | executed                            |                 0 |
| heloc     | logistic     | wachter       | native        | 128 | 0.901389   |  1.62486  | executed                            |                 0 |
| heloc     | logistic     | grace_wachter | native        | 128 | 0.995313   |  2.00432  | executed                            |                 0 |
| heloc     | logistic     | trex          | native        | 128 | 1          |  4.39964  | executed                            |                 0 |
| heloc     | logistic     | roar          | native        | 128 | 0.562847   |  2.86978  | executed                            |                 0 |
| heloc     | logistic     | proplace      | native        | 128 | 0.990625   |  3.32015  | executed                            |                 0 |
| heloc     | logistic     | ellice        | native        | 128 | 1          |  3.79668  | executed                            |                 0 |
| heloc     | mlp          | wachter       | native        | 128 | 0.778646   |  0.991175 | executed                            |                 0 |
| heloc     | mlp          | grace_wachter | native        | 128 | 0.998958   |  1.2417   | executed                            |                 0 |
| heloc     | mlp          | trex          | native        | 128 | 0.929688   |  3.64642  | executed                            |                 9 |
| heloc     | mlp          | roar          | native        | 128 | 0.84184    |  2.8896   | executed                            |                 0 |
| heloc     | mlp          | proplace      | native        | 128 | 0.0256944  |  0        | executed                            |               128 |
| heloc     | mlp          | ellice        | native        | 128 | 1          |  2.45794  | executed                            |                 0 |
| heloc     | mlp          | dice          | native        | 128 | 0.922569   |  4.30624  | executed                            |                 0 |
| heloc     | mlp          | grace_dice    | native        | 128 | 1          |  4.33161  | executed                            |                 0 |

## Answers to the diagnostic questions

1. Exact Table1 reproduction cannot be established because the historical source
   configuration/run is unverified. The table above gives the new seed0 ranking;
   unsupported/infeasible rows cannot be used to claim a method outperforms another.
2. Logistic-regression results are included on the same split definitions and
   classifier-specific unfavorable cohorts. The Base-to-GRACE comparison below
   reports whether FV improves and the associated proximity change; it does not
   establish a multi-seed ranking.
3. Official DiCE initializes its own independently calibrated GRACE population.
   The comparison below shows its observed outcome without attributing Base
   generator differences to the robustness penalty alone.

| dataset   | classifier   | base    |   base_FV |   grace_FV |   base_Cost |   grace_Cost | status   |
|:----------|:-------------|:--------|----------:|-----------:|------------:|-------------:|:---------|
| diabetes  | logistic     | wachter |  0.851064 |   1        |    0.92762  |      1.40519 | executed |
| diabetes  | mlp          | wachter |  0.732889 |   0.999556 |    1.10247  |      1.83528 | executed |
| diabetes  | mlp          | dice    |  0.949778 |   0.999111 |    2.78529  |      2.95825 | executed |
| fico      | logistic     | wachter |  0.843403 |   0.989062 |    1.76621  |      2.09118 | executed |
| fico      | mlp          | wachter |  0.793056 |   1        |    1.08565  |      1.50987 | executed |
| fico      | mlp          | dice    |  0.949306 |   1        |    5.66522  |      5.72991 | executed |
| german    | logistic     | wachter |  0.596382 |   0.993282 |    1.76949  |      2.09937 | executed |
| german    | mlp          | wachter |  0.586787 |   1        |    1.99802  |      2.58813 | executed |
| german    | mlp          | dice    |  0.882883 |   0.999399 |    3.85114  |      3.9567  | executed |
| heloc     | logistic     | wachter |  0.901389 |   0.995313 |    1.62486  |      2.00432 | executed |
| heloc     | mlp          | wachter |  0.778646 |   0.998958 |    0.991175 |      1.2417  | executed |
| heloc     | mlp          | dice    |  0.922569 |   1        |    4.30624  |      4.33161 | executed |

Separate hard-category shift diagnostics are saved in
`hard_category_shift_summary.csv`:15 hard categorical interventions per mixed
classifier, excluded from calibration and generated only after outputs freeze.
These perturb categories in training examples, unlike merely hardening the CF.

Single-seed rankings are diagnostic and cannot replace the main multi-run table.
