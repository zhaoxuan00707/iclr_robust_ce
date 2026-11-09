# GRACE population-versus-individual adversary: completed

The aggregation-only ablation is implemented and verified. Both main variants
use32 supports and exactly the same saved Base/factual/reference tensors, geometry,
complete10-step update, optimizer/stopping, one-to-one assignment and45 test models
within each paired comparison. Default population mode matches the legacy
objective, support gradient, optimizer output and trace in regression tests.

**Seed interpretation:** seeds3/4/5 resample populations from the original seed0
MLP cohort; these are NOT independent classifier-training seeds. This choice
preserves the exact original model bank. German draws32 of37 available factuals,
so cohort overlap is substantial. Reported variation is sample SD across these
three population draws, not a standard error over future checkpoints.

## Main results

All18 final outputs meet the original geometry tolerance and have100% current
validity. Small positive raw geometry violations (maximum about2.7e-5) are retained
in the CSV; feasibility is tolerance-based. German rows below use relaxed
categorical supports, not legal hard-category recourse.

| Dataset   |   r/b_ref | Variant    | FV (%)          | Proximity         |   Current validity (%) | Time (s)        |   Peak memory max (GiB) | Kappa             |
|:----------|----------:|:-----------|:----------------|:------------------|-----------------------:|:----------------|------------------------:|:------------------|
| german    |         3 | individual | 100.00 +/- 0.00 | 2.4861 +/- 0.1123 |                    100 | 580.8 +/- 307.0 |                   0.342 | 0.7199 +/- 0.0058 |
| german    |         3 | population | 100.00 +/- 0.00 | 2.5521 +/- 0.0558 |                    100 | 121.7 +/- 3.4   |                   0.042 | 0.7158 +/- 0.0023 |
| heloc     |         0 | individual | 100.00 +/- 0.00 | 1.3850 +/- 0.0215 |                    100 | 1156.6 +/- 26.6 |                   1.568 | 0.6518 +/- 0.0170 |
| heloc     |         0 | population | 100.00 +/- 0.00 | 1.3850 +/- 0.0215 |                    100 | 134.1 +/- 6.2   |                   0.157 | 0.6518 +/- 0.0170 |
| heloc     |         3 | individual | 100.00 +/- 0.00 | 1.6355 +/- 0.0459 |                    100 | 890.3 +/- 315.7 |                   3.068 | 0.7114 +/- 0.0358 |
| heloc     |         3 | population | 100.00 +/- 0.00 | 1.5777 +/- 0.0327 |                    100 | 88.1 +/- 43.1   |                   0.249 | 0.6933 +/- 0.0322 |

## Interpretation

- At the predeclared positive radius r/b_ref3, population GRACE has lower
  proximity on all three HELOC draws: mean1.5777 versus1.6355 for individual
  adversaries (difference0.0577), with identical100% FV.
- On German, the direction reverses: individual adversaries have lower proximity
  on all three draws: mean2.4861 versus2.5521 (difference0.0660), again with100% FV.
  This does not support a uniform population-adversary proximity advantage.
- HELOC's originally calibrated choice is r0. Both modes produce byte-identical
  outputs there, with proximity1.3850 and100% FV. The positive-radius ablation is
  an additional predeclared diagnostic, not a replacement for that selection.
- Kappa below1 establishes cancellation of the per-CF sensitivity vectors at
  these final populations. It does not by itself prove a better FV--proximity
  trade-off. Both ratio terms use the same vectors; zero denominator is flagged
  explicitly (no such cases in these completed runs).
- Population optimization is substantially faster and uses less allocated GPU
  memory in this implementation. These are single timing observations with
  dataset/seed jobs sharing the host on separate GPUs, not isolated benchmark
  latency or statistical speedup guarantees. Individual sensitivity chunk size1
  bounds each batched VJP; larger chunks remain configurable and tested.
- Both retain the existing stopping criterion; no convergence certificate is
  claimed, and several runs exhaust800 updates. Different stopping times can
  contribute to final differences even under the identical stopping rule.

## Hard-category German diagnostic

Hard realization is separate from optimization. Both variants remain geometry
feasible and currently valid after hardening. The per-seed hard scores are saved
in `main/hard_category_diagnostics.csv`; they must not be conflated with the
relaxed headline table.

- individual: FV 95.60% +/- 1.28 pp; proximity 2.3491 +/- 0.1006.
- population: FV 96.50% +/- 0.45 pp; proximity 2.3919 +/- 0.0614.

Here the population variant has higher mean hard-category FV and higher mean
cost; neither variant uniformly dominates on both metrics.

## Verification and saved outputs

Eight new aggregation/gradient/regression tests and three existing mixed-domain
tests pass. Corrected three-step Diabetes preflight and full10-CF debug pair pass
independent checks; the preparation-only failed preflight is preserved. Main
reporter independently checks18 saved outputs,9 paired manifests/budgets, all
per-future FV scores, original-input hashes, matching, metrics and geometry.
All original inputs and checkpoints remain unchanged. No retraining or validity
repair was performed.

Raw runs:
- `/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_profile_v1` (preparation failure)
- `/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_profile_v2` (corrected preflight)
- `/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_debug_v1` (full Diabetes debug)
- `/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_main_v1` (main18 solves)

Original parent:
`/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1`.

Files: `main/per_seed_metrics.csv`, `main/aggregated.csv`,
`main/per_future_model.csv`, `main/paired_differences.csv`,
`main/independent_checks.csv`, and `verification_tests.json`.
Commands and core mapping: `implementation_plan.md`,
`experiments/README.md`, controller106 and independent reporter107.

The optional64-support extension was not run. German has only37 frozen test
factuals; extending its cohort requires a separately declared population and
new Base generation. HELOC64 is deferred given the higher individual-mode cost;
no64-support result is claimed. No full radius sweep or classifier-training-seed
replication was performed. These limitations must accompany any reviewer response.
