# German Credit MLP PROPLACE license-size repair

Original task103 seeds1/2 selected delta=bias_delta=.005 from calibration.
Their test runs hit Gurobi restricted-license model-size error at factual28/41,
then skipped remaining20/6 cases respectively. Preserve the original artifacts.

Opt-in master fallback retains the same neural forward constraints, accumulated
adversarial cuts, empirical convex hull, standardized simplex and L1 objective.
It tries Gurobi first, switching only on license-size error10010 to SciPy HiGHS
MILP. The absolute-distance auxiliaries are explicitly continuous nonnegative;
this is equivalent to the nonnegativity implied by the original two absolute
value inequalities. No cuts are removed, no radius/architecture is reduced, and
no hyperparameter selection uses held-out models. Native adversary stays unchanged.
Default pipeline remains unchanged unless proplace_license_fallback=true.

Two solver tests pass: matching feasible integer optimum and a2100-constraint
problem beyond the restricted license size. Retry runner114 uses frozen selected
radius and only failed/skipped factuals, preserves successful outputs bytewise,
and evaluates the exact saved45 future models. Per-instance budget remains30s.
Any unresolved failures remain explicit; relaxed and hard results are separate.

New raw outputs:
/home/xu.zhao/robust_ce_results/iclr_runs/proplace_license_fix/20261109_seed1_v1
/home/xu.zhao/robust_ce_results/iclr_runs/proplace_license_fix/20261109_seed2_v1

Run commands: env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MPLCONFIGDIR=/tmp/task103_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python
experiments/114_fix_german_proplace.py --seed S --output NEW_PATH

Status: retries launched; do not treat this document as successful results yet.

## Same-budget outcome
The original30s limit repairs25/26 affected factuals. Seed1 all48 outputs are
native robust; seed2 has46/47 native robust and one remaining timeout. The
corrected relaxed mean across two seeds is FV98.9362±1.5045%, proximity
7.06549±.28003. Native checks and per-future scores are saved. Original successful
outputs are byte-identical; no blanket infrastructure failure remains.

One-case extended-budget diagnostic launched separately as seed2_v3,300s.
This is not the original30s benchmark. Keep its results distinct.

Extended v3 failed because HiGHS retained an internal30s cap. v4 propagates the
remaining per-factual300s budget to both master solvers. It is a separate
follow-up, not part of the original30s metrics. All3 solver tests pass, including
comparison to original master objectives on four small neural networks.
The full summary with only the German MLP PROPLACE row corrected at30s is
`all_methods_summary_30s_followup.csv`; original summary remains untouched.

## Extended-budget completion
Seed2_v4 solved factual41 in289.186s within the declared300s budget. The final
master used HiGHS; all seven master solves retain their solver diagnostics.
Reporter115 independently verified all95 relaxed CFs native robust and replayed
the saved45 future models. Zero remaining failures. Across seeds1/2, relaxed
CV/FV100±0%, proximity7.12888±.36967. See extended_budget/report.md. This row uses
an extended budget for one case and must NOT replace the30s row without disclosure.
Hard realization remains distinct: mean CV98.9583%, FV99.0278%, and proximity
7.58382; the relaxed native certificate does not automatically apply after rounding.
