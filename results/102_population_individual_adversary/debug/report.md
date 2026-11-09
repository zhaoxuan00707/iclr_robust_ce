# Controlled population-versus-individual adversary ablation

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_debug_v1`. 2 saved outputs and 1 paired comparisons independently checked. No parent output changed.

Mean +/- sample standard deviation across seeds3/4/5 (where three runs exist). Seeds resample factual/Base populations from the fixed parent seed0 classifier cohort; they are not independent classifier-training seeds. Populations may overlap substantially, especially German (32 drawn from37). All modes/seeds reuse the exact same45 parent future checkpoints for their dataset; repeated shifts are not independent statistical units.

| dataset   |   n |   r_ratio |   beta | variant    |   seeds |     FV |   proximity |   current_validity |   optimization_seconds |   kappa | feasible   |
|:----------|----:|----------:|-------:|:-----------|--------:|-------:|------------:|-------------------:|-----------------------:|--------:|:-----------|
| diabetes  |  10 |         3 |    1.2 | population |       1 |  99.78 |      2.0057 |                100 |                  18.42 |  0.9246 | 1/1        |
| diabetes  |  10 |         3 |    1.2 | individual |       1 | 100    |      2.0059 |                100 |                 100.96 |  0.9265 | 1/1        |

FV/current validity are percentages; proximity is original standardized-feature L2. Times are synchronized optimization wall seconds including final objective diagnostics, excluding input loading, evaluation and kappa diagnostics. Peak allocated GPU memory is in per_seed_metrics.csv. Pair modes run sequentially on the same GPU; other dataset/seed jobs can overlap on other GPUs/shared host.

Only sensitivity aggregation changed. Both methods share Base, reference, data-side perturbation map, complete10-step SGD(.01) update, smooth temperature.05, optimizer, stopping, exact geometry and assignment. Individual sensitivities use sqrt(N) times the flattened training-perturbation gradient norm; kappa numerator and denominator use identical vectors at the final Q. Zero denominator is flagged and kappa is null.

HELOC original calibration selected r=0, so its paired outputs must be identical. The r/b_ref3 HELOC setting was declared before evaluation as an additional positive-radius diagnostic with unchanged beta1.1. German uses the original r/b_ref3,beta1.1. These settings were not retuned for32 supports.

German optimization is the original A.5 relaxed categorical geometry; hard realization is a separate diagnostic. Infeasible final outputs remain labeled and must not be treated as successful constrained solutions. No validity repair or convergence certificate. No new morphology/actionability claims.

All required raw metrics are in per_seed_metrics.csv, per-scenario scores in per_future_model.csv, and signed individual-minus-population differences in paired_differences.csv. No64-support German experiment: the frozen parent cohort has only37 factuals; duplication would change the population.
