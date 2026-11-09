# Task 002 preparation: specification and domain resolution

Date: 2026-09-16. Preparation completed; path-level solver implementation and
experimental evaluation have not started.

## Specification

No complete sections 1–9 were found in the current task, repository notes,
previous handoff or experiment documentation inspected. The original task is
preserved. `tasks/002_path_specification_supplement.md` supplies an explicitly
new specification, with configuration at
`experiments/configs/path_level_geodesic_wro_v1.yaml`. It is not recovered text.

The default objective is mean of the 21 lambda-column bank risks, each computed
as max-model population-mean hinge. Two intermediate target-mass control
distributions define an exact-plan piecewise Wasserstein path, parameterized by
arclength, constrained by one absolute total-length budget. Current validity is
required on every deployment-grid support point. Held-out refinement is batch
optimization with frozen method settings and construction bank, not reuse of a
training coupling or tuning on future outcomes. Original deployment lambda
selection stays unchanged. An optional aggregation preference was requested;
mean risk is explicitly declared as the default in the absence of a correction.

## Feature-domain resolution

Evidence: `heloc_diabetes_official_lift_v1.py` records all-mutable generation
with range violations audited; the previous `single_budget_geodesic_wro_v1.py`
applies the training box to learned controls. The frozen source populations are
current-valid, but lie outside that box with masses .84375 (train), .859375
(calibration), and .8515625 (test).

The new specification retains the box on learned control particles and records
complete-path range violations as diagnostics. It does not add an impossible
box requirement to unchanged source endpoints, enlarge the box, or claim
physical actionability. If full-path box compliance is required instead, this
fixed-instance design must be reported infeasible. Current validity remains a
hard requirement at controls and all deployment-grid points. No existing
scientific implementation/domain was modified by this preparation.

## Exact endpoint necessary checks

| Split | Exact Euclidean W2 to P1 | Slack at smallest fixed B_geo |
|---|---:|---:|
| Training | 4.8684782573 | 0.0000000000 |
| Calibration | 4.6477875161 | 0.2206907412 |
| Test | 4.7756652535 | 0.0928130038 |

All endpoints are current-valid. All twelve split/budget combinations pass
necessary endpoint conditions. Largest OT marginal residual is below 1e-14.
These checks do **not** prove a grid-valid or continuously valid path exists.

Because B_geo is an unchanged absolute training-derived budget, beta=1 imposes
the shortest-geodesic restriction only on training. Held-out anchor distances
are smaller, so the same budget permits some detour. Recomputing B_geo from each
batch would violate the fixed-budget requirement and is not done.

Only frozen input populations, current model and exact geometry were read;
future-model outcomes and labels were not used for choices. Historical raw
outputs and task text are unchanged. Existing known future outcomes remain
exploratory and cannot become fresh confirmation.

## Verification and next execution gate

Added reusable `path_preflight.py` and four focused tests for immutable budgets,
endpoint/domain conflicts, endpoint current validity and the distinction
between necessary checks and a path certificate. All 62 local tests pass.

```bash
cd /home/xu.zhao/iclr_robust_ce
PY=/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY experiments/004_path_preflight.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY experiments/run.py test
```

Details and input hashes: `004_path_preflight.json` and
`004_path_preflight.csv`. Next: implement the declared path solver and inference
rule, run synthetic and small construction-only smoke tests, and estimate full
compute. Fresh discovery and GPU replay are gated on actual path feasibility.
No driver for the new path-level method exists yet; do not launch the previous
single-budget driver under the new method's name.
