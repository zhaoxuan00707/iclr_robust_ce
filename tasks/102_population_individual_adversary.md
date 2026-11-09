# Controlled GRACE population-versus-individual adversary ablation

Requested by user on2026-11-08. Change only update-aware sensitivity aggregation,
not the support population, uncertainty coordinates/norm, update map, geometry,
optimizer/stopping, or assignment. The nominal loss is the existing smooth hinge.
Default mode remains population. No existing experiment outputs may be replaced.

Implementation and scientific choices are predeclared in
`results/102_population_individual_adversary/implementation_plan.md`.
Config: `experiments/configs/population_individual_adversary.json`.
Core mode: `mixed_update_aware.mixed_surrogate(adversary_mode=...)`.
Existing optimizer: `appendix_tabular.grace(adversary_mode=...)`.
Entry point: `experiments/run.py run population_individual_adversary`.
Controller106; independent reporter107. Use new external run directories only.

The original parent seed0 MLP, train/reference tensors, Wachter outputs and all45
held-out future checkpoints are frozen. Seeds3/4/5 select populations from the
saved unfavorable factual cohort; this isolates aggregation and makes exact bank
reuse possible. These are population-draw seeds, not three independent fitted
classifiers. This assumption was stated to the user before main computation.

Debug: Diabetes, n10, seed3, original r/b_ref3,beta1.2. Main: n32, HELOC/German,
seeds3/4/5. Original HELOC r0,beta1.1 gives a mandatory identity control; additionally
predeclare r/b_ref3,beta1.1, otherwise no aggregation effect could be measured.
German uses its original selected r/b_ref3,beta1.1. No recalibration or test-based
choice. German's parent test cohort has37 factuals, precluding a64-CF subset.

Raw base root:
`/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/`.
`20261108_profile_v1` is a preserved preparation-only argument-order failure.
`20261108_profile_v2` passes both three-step GPU preflights.
`20261108_debug_v1` uses the full original optimizer/stopping rule.

Tests:8 new aggregation/regression tests and3 existing mixed-domain tests pass.
Keep original source function fixture for optimizer parity. Report failures,
final-iterate feasibility, synchronized time, peak memory, kappa from identical
vectors (null for zero denominator), per-model and per-population metrics.
No standard error treating correlated/repeated future checkpoints as independent.

Completed main ablation; see results/102_population_individual_adversary/report.md and completion_check.json. Optional64-support extension not executed.
