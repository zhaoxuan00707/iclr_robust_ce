# Task 002: baseline audit and specification blocker

Status: baseline preflight complete; requested path-level implementation and evaluation are **not complete**. No new experiment was launched.

## Missing specification

`tasks/002_baseline.md` is 352 lines and starts with:

> (\widetilde R_{\rm path}) is a smooth search surrogate;

It includes numbered sections 10–20 but no sections 1–9. The beginning of the optimization section is also absent. The missing text has been requested from the user. The older `.agent/tasks/001_continue_current_work.md` describes a review/design task and does not supply the missing path-level objective or representation. It is not a substitute for the current task.

Implementation depends on the missing definitions:

- The true path-level objective and its aggregation across lambda. Minimizing a minimum, mean, or maximum over path positions defines different experiments; these cannot be inferred from the surrogate symbol.
- The path-control representation, default K, transport coupling/gluing rules, and what learned construction parameters remain fixed during held-out batch refinement.
- Exact path-length definition, interpolation parameterization, current-validity enforcement domain/grid, tolerances, and feature constraints.
- The first-run dataset/radius/budget scope and any optimizer settings specified in the missing sections.

No replacement scientific objective, constrained path algorithm, or new configuration has been invented.

## Preserved baseline and reproducibility

Implementation repository: `/home/xu.zhao/robust_ce/`.

Completed raw run:

`/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/`

All artifact names below are relative to that run:

- Config: `heloc/resolved_config.json`.
- Instance and preprocessing/identity references: `heloc/instance_manifest.json`, `heloc/data_audit.pt`, `heloc/training.pt`, and the train/cal/test population files.
- Base checkpoint: `heloc/current.pt`.
- Previous control checkpoints: `heloc/learned_controls.pt`, with per-control paths and hashes in `heloc/controls_manifest.json`.
- Old waypoint checkpoint: `heloc/frozen_waypoint.pt`; provenance in `heloc/waypoint_manifest.json`.
- Available expanded bank: `heloc/evaluation_bank.json`, containing 5,387 model/displacement references. Its recorded provenance is training-only, uses the fixed full-update seed, and passes the recorded radius checks. No historical full-model replay was repeated during this preflight.
- Original orchestration: `run_seven_gpu.py`, using module `lift_ce.experiments.single_budget_geodesic_wro_v1`. This script points at completed outputs and must not be rerun unchanged.
- Completion and checksums: `execution_completion.json` and `heloc/completion_audit.json`.

Baseline settings: HELOC seed 1; fixed `r/b_ref=3`; absolute data budget `8.188831830024718`; `D=4.868478257271079`; beta grid `[1,1.05,1.10,1.20]`; 21 lambda values from 0 to 1. Control optimizer: 120 Adam steps at learning rate .03, smoothing constants 50/100, three initializations, random seed 20260915. Official full Update uses the frozen initialization and fresh AdamW, learning rate .0002, weight decay .0001, 8 epochs, batch size 256. Exact settings remain in the original config/manifests.

Prior execution completed 12 optimization jobs, 54 fresh searches, 810 full model replays including 216 radial candidates, and 2,036,286 cross-evaluations on GPUs 0,1,2,3,5,6,7. These are prior results, not new task-002 results.

Reproduce this read-only preflight from the current repository:

```bash
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/002_baseline_audit.py
```

This writes only small audit summaries under `results/`; it does not invoke a model, optimizer, GPU experiment, or raw-output writer.

## Findings

All four learned controls were geometry-feasible and current-valid. Their true construction-bank hinges were .488634, .414663, .368982, and .306832, unchanged by the prior fresh discovery round.

All selected new test paths were rejected for deployment:

| beta | Selected lambda | Minimum f0 score | Invalid represented support points | Selected-output budget overrun |
|---|---:|---:|---:|---:|
| 1.00 | .65 | .228382 | 542 | None within numerical tolerance |
| 1.05 | .70 | .473985 | 96 | .623267 |
| 1.10 | .65 | .462608 | 85 | .671484 |
| 1.20 | .65 | .498542 | 4 | .598827 |

These overruns are distances of the **selected distribution**, not merely the total length of its containing path. New task-002 acceptance additionally requires checking the declared full path length and all deployment-grid outputs.

The old waypoint has higher named-future validity at higher cost. Its training path length is 8.147930, exceeding all previous single-budget caps. Its original training bank contained 851 models, so objective-only attribution or matched-cost superiority is not justified. Lightweight matched metrics are in `results/002_baseline_metrics.csv`.

A further preflight issue is the frozen source endpoint. Its mass outside the observed training feature box is approximately:

- Training: 84.375%.
- Calibration: 85.9375%.
- Test: 85.1563%.

The source endpoint is current-valid. Thus a new requirement that every path point, including lambda=0, lie inside that box would conflict with retaining these anchors. This is a conditional specification conflict, not proof that the admissible geometry is disconnected. Do not change endpoints or relax the domain silently.

The previous beta=1 path is current-invalid at interior grid points. This does not by itself prove that every shortest-geodesic realization is invalid; representation and potential coupling alternatives must be distinguished from genuine infeasibility.

## Checks and boundaries

The audit verified all artifact checksums listed in the previous completion manifest, frozen instance file checksums, unique expanded-bank IDs, recorded training-only provenance/update seeds/data budgets, exact control feasibility, nested control risk, and selected-output rejection. Checks passed. The detailed machine-readable audit is `results/002_baseline_audit.json`.

No raw outputs were modified or copied into this repository. No manuscript or production claims changed. Existing named-future outcomes remain exploratory: 30/45 scenarios pass the sufficient data-radius proxy, and 18 final model states overlap calibration model states. Scratch retraining remains separate stress evidence.

## Next steps

1. Obtain sections 1–9 of `tasks/002_baseline.md`, including the missing optimization formula and path representation.
2. Resolve the endpoint/domain requirement against the frozen source artifacts without silent relaxation.
3. Freeze the complete path-level method specification and fresh output location before implementation-dependent work.
4. Implement and test the specified path constraints and held-out batch refinement; run construction smoke checks before fresh discovery.
5. Freeze paths, perform matched training-only discovery, and cross-evaluate all controls without feedback into construction. Apply the stated rejection criteria to full paths and held-out outputs.

State is tracked in root `state.yaml`. The task must not be marked complete until its requested implementation and evaluation are completed or explicitly rescoped.
