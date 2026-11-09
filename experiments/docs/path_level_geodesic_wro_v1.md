# Path-level constrained solver

Implementation: `src/lift_ce/counterfactuals/path_level_solver.py`.
Driver: `src/lift_ce/experiments/path_level_geodesic_wro_v1.py`.
Specification: `../../tasks/002_path_specification_supplement.md`.

The primary objective averages per-lambda population bank maxima over the same
21-point grid. The K=2 intermediate distributions retain target-anchor uniform
masses. Adjacent unregularized OT plans define the piecewise path. Later legs
are permutations; gluing them to the first split plan preserves factual IDs and
weights. This adapter explicitly rejects incompatible control representations.
It is not a general nonuniform-mass gluing implementation.

Total path length is the sum of adjacent exact W2 distances. Interpolation uses
cumulative Wasserstein arclength. Differentiation holds each exact plan fixed
and differentiates path coordinates, lengths and arclength fractions locally.
Plan switches are nonsmooth. Gradients of the smooth bank objective are streamed
in model chunks and reattached with a vector-Jacobian product. The reported
objective always uses true hinge and the population-first model reduction.

Numerical validity residual:

`g_valid = (mean_lambda E_Q[.5-f0(z)]_+ + mean_control [.5-f0(w)]_+) / 2`.

All masses are positive, so g_valid=0 imposes current validity everywhere in the
finite representation. The numerical optimizer has separate geometry/validity
duals and enforcement schedules. Final acceptance does **not** rely on a small
average residual: every grid support and control must have reference-CPU score
at least .5. Control particles are clipped to the inherited training box;
full-path box violations are diagnostics, not silently projected outputs.

Use the same solver for held-out batches with unchanged absolute B_geo and
frozen construction bank/method. New batch coordinates are inference variables.
Training coordinates serve only as a warm start. Reject endpoint impossibility
before optimization, and reject any final path with grid or budget violations.
This is an explicitly declared batch optimization rule, not a learned transport
map, continuous-validity certificate, or global optimum.

## Execution and gates

From the repository root with the `wm_gpu` Python environment, the local
launcher supports phases `prepare`, `smoke`, `benchmark`, `optimize`, `collect`,
`discover`, `freeze_eval_bank`, `select`, `future_models`, `evaluate`, `report`.
Use a fresh absolute output below `robust_ce_results/iclr_runs/`; later phases
require `--resume`. The driver retains historical controls verbatim.

The small-anchor smoke with an unchanged full-instance budget can reject the
subset on endpoint distance. `005_path_subset_smoke.py` separately tests actual
subset optimization using an explicitly subset-derived budget. It is excluded
from primary candidate selection. Full-support benchmark candidates are also
excluded. No partial-bank result is substituted for a primary full-bank run.

`006_run_path_jobs.py` records four visible GPU assignments and runs all 16
construction jobs, then collects exact-feasible incumbents. Its fixed output is
for the current experiment only; do not execute it again on completed artifacts.
`007_path_progress.py` reads live traces into a lightweight progress summary.

If every construction job fails retention, `report` records construction
rejection, gates fresh discovery and held-out deployment, and lists missing
scientific comparisons. This is a completed local-solver feasibility failure,
not a completed robustness comparison or proof of geometric impossibility.
If construction succeeds, complete cal/test batch optimization and freeze their
paths before shared-bank selection/evaluation. Do not interpret legacy controls'
known constraint failures as accepted deployment solutions.

Model checkpoints, population tensors, traces and job outputs stay in the new
external run directory. Repository `results/` contains lightweight summaries.
