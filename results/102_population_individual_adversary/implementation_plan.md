# Population versus individual adversary: pre-execution plan

Core map:
- `counterfactuals/mixed_update_aware.py`: A.5 training-input perturbation and
  smooth loss, empirical-RMS dual scaling sqrt(N), population sensitivity.
- `counterfactuals/update_aware.py`: exact differentiable full-batch SGD unroll,
  differentiable W2 with an exact frozen optimal plan.
- `experiments/appendix_tabular.py:grace`: shared population optimizer, two-anchor
  constraint, projection, dual update, stopping, final-iterate retention,
  and one-to-one assignment. Only objective mode/chunk arguments will be added.
- `counterfactuals/geodesic_excess.py`: exact W2 and equal-mass assignment.
- `data/tabular.py`, `models/tabular.py`: train-only processing and MLP fitting.
- `experiments/appendix_tabular.py:future_bank,score`: original 45-scenario update
  and evaluation; unchanged and reused.

Add optional population(default)/individual mode. Compute one unrolled update
per objective call, then chunked batched per-loss vector-Jacobian products with
respect to the SAME training-input perturbation. Retain higher-order graphs for
support optimization. Keep sqrt(N) scaling, loss temperature, model update,
geometry, step size, stopping, matching and evaluation unchanged.

Parent: /home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1.
Initial Diabetes n10 seed3; original selected r/b_ref3,beta1.2.
Main HELOC/German n32 seeds3,4,5. Prefer fixed parent MLP/update data/reference
and exact45 test states; seeds then describe factual/Base subsampling, NOT
independent fitted models. This interpretation is being confirmed with user.
Draw nested populations from original test unfavorable cohort and associated
saved Wachter outputs. No Base regeneration or outcome filtering.
HELOC original selection r/b_ref0,beta1.1: report the expected identical r0
comparison AND a predeclared r/b_ref3,beta1.1 diagnostic to identify aggregation.
German original selection r/b_ref3,beta1.1. Never tune on test outcomes.
Optional n64 only if profiling supports it (German parent cohort has37 cases,
so n64 without replacement is unavailable and must not be fabricated).

Independent checks: original population objective/gradient and optimizer parity;
mean sensitivity identity; n1 equivalence; triangle inequality; finite-difference
individual gradients; chunk invariance; equal inputs/banks, assignment/geometry;
explicit denominator-zero convention for kappa. Infeasible final iterates remain
failures, not repaired. Peak memory and synchronized wall time recorded.
Report per-seed/per-future FV plus mean and sample SD across population draws,
not standard errors over45 correlated scenarios or independent training seeds.
