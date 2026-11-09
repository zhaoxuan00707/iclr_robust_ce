# Core migration validation

Completed 2026-09-16. Authoritative core source:
`/home/xu.zhao/iclr_robust_ce/experiments/src/lift_ce/`.

Imported 57 project Python modules, 12 existing test modules, 89 vendor Python
files, vendor licenses/metadata, and historical HELOC/DIABETES YAMLs: 182 copied
files totaling approximately 1.1 MB. Four launcher regression tests were added.
The ten supported workflow entry points are listed by `experiments/run.py list`.
The closure includes older shared helpers; unrelated research prototypes remain
in the old workspace. No large artifacts or Git commits were created.

## Validation

- All 58 tests passed on CPU (54 imported tests and four migration tests).
- All ten supported CLI help commands returned successfully.
- Imported 94 project/vendor modules from this checkout without historical
  source fallback, using the existing `wm_gpu` environment.
- Loaded the frozen HELOC instance at
  `/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc`
  read-only. Verified training, training-population and current-model hashes,
  and the exact official Update function-source hash. Training shape: 6000 × 22.
- Verified original imported-file hashes remain unchanged. Local scientific AST
  is unchanged outside `main` argument parsing and `git_version` metadata.
- Historical outputs are rejected by the launcher; resume requires matching
  repository/module/code identity. A real CLI dry run created no output.

Commands (from repository root):

```bash
PY=/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY experiments/run.py check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY experiments/run.py test
$PY experiments/run.py run single_budget_geodesic_wro_v1 -- --help
```

Machine-readable details: `003_core_migration_checks.json`,
`003_core_migration_environment.json`, and
`../experiments/provenance/core_import.json`. Usage: `../experiments/README.md`.

## Scope and limitations

No experiment was rerun and no method, production scientific default, bank,
checkpoint, dataset, or manuscript claim changed. Raw inputs still live under
`/home/xu.zhao/robust_ce_results/` and `/home/xu.zhao/robust_ce_data/`.
New runs use `/home/xu.zhao/robust_ce_results/iclr_runs/<unique_run_id>/`.
The local source snapshot includes modified vendor working-tree files with their
hashes, rather than claiming pristine upstream revisions. Optional baseline
execution still requires appropriate solver licenses. Dependency-module legacy
CLIs are not supported launch targets. Full GPU experiment equivalence was not
retested; frozen-input compatibility and focused regressions were verified.

Task 002 remains awaiting its missing sections 1–9. Moving code does not resolve
the missing path objective/representation or the previously recorded full-path
feasibility issue.
