# Controlled population-versus-individual adversary ablation

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_main_v1`. 18 saved outputs and 9 paired comparisons independently checked. No parent output changed.

Mean +/- sample standard deviation across seeds3/4/5 (where three runs exist). Seeds resample factual/Base populations from the fixed parent seed0 classifier cohort; they are not independent classifier-training seeds. Populations may overlap substantially, especially German (32 drawn from37). All modes/seeds reuse the exact same45 parent future checkpoints for their dataset; repeated shifts are not independent statistical units.

| dataset   |   n |   r_ratio |   beta | variant    |   seeds | FV              | proximity         | current_validity   | optimization_seconds   | kappa             | feasible   |
|:----------|----:|----------:|-------:|:-----------|--------:|:----------------|:------------------|:-------------------|:-----------------------|:------------------|:-----------|
| heloc     |  32 |         0 |    1.1 | population |       3 | 100.00 +/- 0.00 | 1.3850 +/- 0.0215 | 100.00 +/- 0.00    | 134.13 +/- 6.17        | 0.6518 +/- 0.0170 | 3/3        |
| heloc     |  32 |         0 |    1.1 | individual |       3 | 100.00 +/- 0.00 | 1.3850 +/- 0.0215 | 100.00 +/- 0.00    | 1156.59 +/- 26.63      | 0.6518 +/- 0.0170 | 3/3        |
| heloc     |  32 |         3 |    1.1 | population |       3 | 100.00 +/- 0.00 | 1.5777 +/- 0.0327 | 100.00 +/- 0.00    | 88.12 +/- 43.13        | 0.6933 +/- 0.0322 | 3/3        |
| heloc     |  32 |         3 |    1.1 | individual |       3 | 100.00 +/- 0.00 | 1.6355 +/- 0.0459 | 100.00 +/- 0.00    | 890.31 +/- 315.74      | 0.7114 +/- 0.0358 | 3/3        |
| german    |  32 |         3 |    1.1 | population |       3 | 100.00 +/- 0.00 | 2.5521 +/- 0.0558 | 100.00 +/- 0.00    | 121.70 +/- 3.36        | 0.7158 +/- 0.0023 | 3/3        |
| german    |  32 |         3 |    1.1 | individual |       3 | 100.00 +/- 0.00 | 2.4861 +/- 0.1123 | 100.00 +/- 0.00    | 580.81 +/- 307.04      | 0.7199 +/- 0.0058 | 3/3        |

FV/current validity are percentages; proximity is original standardized-feature L2. Times are synchronized optimization wall seconds including final objective diagnostics, excluding input loading, evaluation and kappa diagnostics. Peak allocated GPU memory is in per_seed_metrics.csv. Pair modes run sequentially on the same GPU; other dataset/seed jobs can overlap on other GPUs/shared host.

Only sensitivity aggregation changed. Both methods share Base, reference, data-side perturbation map, complete10-step SGD(.01) update, smooth temperature.05, optimizer, stopping, exact geometry and assignment. Individual sensitivities use sqrt(N) times the flattened training-perturbation gradient norm; kappa numerator and denominator use identical vectors at the final Q. Zero denominator is flagged and kappa is null.

HELOC original calibration selected r=0, so its paired outputs must be identical. The r/b_ref3 HELOC setting was declared before evaluation as an additional positive-radius diagnostic with unchanged beta1.1. German uses the original r/b_ref3,beta1.1. These settings were not retuned for32 supports.

German optimization is the original A.5 relaxed categorical geometry; hard realization is a separate diagnostic. Infeasible final outputs remain labeled and must not be treated as successful constrained solutions. No validity repair or convergence certificate. No new morphology/actionability claims.

All required raw metrics are in per_seed_metrics.csv, per-scenario scores in per_future_model.csv, and signed individual-minus-population differences in paired_differences.csv. No64-support German experiment: the frozen parent cohort has only37 factuals; duplication would change the population.
