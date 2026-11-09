# Small opt-in mixed-domain baseline adaptation

The historical Table7 outputs are unchanged. This adaptation addresses missing
categorical domain support; it does not guarantee successful or robust CFs.

Changes:
- `roar_one(..., project=None)` preserves the canonical default. Opt-in ROAR
  projects each Adam iterate onto the same standardized categorical simplex as
  GRACE. Hard mode instead uses the existing one-hot projection and is a
  projected-gradient heuristic; small steps can leave categories unchanged.
- `mixed_proplace` subclasses the official PROPLACE implementation and adds
  affine category constraints in `_master_prob_add_inputs`: z=low+scale*p,
  sum(p)=1, p in[0,1] (relaxed) or binary (hard). Its objective, RNCE initialization,
  plausible convex hull and parameter-adversary loop are retained. A fresh
  instance is used per factual to avoid state leakage after a native failure.
- `appendix_tabular.generate(..., categorical_mode='unsupported')` preserves the
  original default. Explicit `relaxed`/`hard` opts into the new adaptations and
  refuses an existing output path. No historical unsupported row is overwritten.

Four unit tests pass, including nontrivially standardized one-hot/simplex
constraints solved with Gurobi. Calibration-only preflight on the first factual
of each German/FICO MLP/logistic cohort uses the first original grid parameter.
This is an implementation check, not calibration or a Table7 replacement.

Raw preflight:
/home/xu.zhao/robust_ce_results/iclr_runs/mixed_baseline_adapters/20261108_relaxed_preflight_v1

Reproduction:
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mixed_baselines_mpl \
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/run.py run mixed_baseline_preflight \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/mixed_baseline_adapters/NEW_PREFLIGHT \
  -- --categorical-mode relaxed
```

Full comparison requires new calibration with the original grids and evaluation
on the exact same saved future models. For relaxed baselines, show both relaxed
and separately hardened scores, just as for GRACE; rounding does not certify
native robustness. Restricted-Gurobi failures, timeouts and invalid outputs must
remain explicit even after the domain support is implemented.
