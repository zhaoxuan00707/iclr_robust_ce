# SUPERSEDED — objective integration check failed

Do not use the tables below as smooth-hinge results. Preparation used the requested smooth hinge, but optimizer calls accidentally fell back to hard hinge. The zero objective and selected r=0 in this run are invalid evidence about smooth-hinge behavior. Preserved for debugging; replacement results are in results/026_mnist_smooth_hinge/ after verification.

---

# MNIST corrected latent Base and smooth-hinge experiment

Calibration selects **r=0, beta=1**. Selected test Future Validity is **96.01%**, versus **96.01%** for the replayed Base and **96.01%** for historical Base. Selected current validity is **100.00%**; pixel cost is **5.5755** versus Base **5.5755**. These are single-checkpoint, 64-factual retrospective results, not a multi-seed benchmark.

## Why r was zero, and what changed

In task011, r=0 won because the positive-radius calibration settings performed worse after the pilot incorrectly used Enc(Base) instead of the generator's actual latent codes. That run also used hard hinge and exhausted its iteration budget. The subsequent hard-hinge preparation on corrected Base showed zero hinge and zero sensitivity, but did not complete optimization because of a retained-graph allocator-cap failure. Neither result establishes the behavior of the clarified smooth-loss method.

This run follows the user's explicit smooth-hinge clarification, overriding the hard-hinge numerical reading of the PDF. Use L_t(p)=t*softplus((0.5-p)/t), with **t=0.05 fixed in probability space**. Both nominal loss and update-aware sensitivity use this same smooth loss, and its complete outer gradient is differentiated. Training loss remains binary cross entropy; validity remains probability>=0.5. Temperature is an explicitly chosen implementation default, not selected using future outcomes. r=0 is retained as the nominal-loss ablation; it is not forced to win or excluded to make a robustness claim.

12 feasible settings tie at the highest calibration FV; lower pixel cost, then lower r and beta break the tie.

Best feasible calibration row per radius (chosen by calibration FV/cost only):

```
   r  beta   cal_FV  pixel_cost  surrogate  converged
0.00   1.0 0.954861    5.938225        0.0       True
0.25   1.0 0.954861    5.938225        0.0       True
0.50   1.0 0.954861    5.938225        0.0       True
```

## Base representation fix

The original generator produces images by decoding latent codes, but its CEResult discarded those codes. We now retain them, including the fallback branch, and use them directly as the Base geometry anchor. No re-encoding and no GRACE validity repair are applied. Replaying the same generator recovers calibration images exactly; historical test images differ by at most 0.0005848 per pixel, so we explicitly store newly replayed image/code pairs and preserve Historical-Base as a separate scored baseline. The first recovery attempt stopped at its strict exact-match check; no original artifact was changed.

Decoding retained codes reproduces the replayed Base exactly in this run and both calibration/test Base current validity are 100%. Compare this to task011's decoded Base current validity of 6.25%/12.5%. The correction is to the Base representation, not to optimized Q.

```
[
  {
    "split": "cal",
    "replay_vs_historical_image_max_error": 0.0,
    "historical_Base_replaced_by_explicit_generator_replay": true,
    "decode_original_image_max_error": 0.0,
    "current_validity": 1.0,
    "nominal_hinge": 0.017451206222176552,
    "sensitivity": 1.6141586303710938,
    "r05_objective": 0.8245305418968201,
    "r05_gradient_norm": 0.5464478731155396,
    "finite_difference": 0.5587935447692871,
    "analytic_directional": 0.5464478731155396,
    "relative_error": 0.022592587987138566
  },
  {
    "split": "test",
    "replay_vs_historical_image_max_error": 0.0005847811698913574,
    "historical_Base_replaced_by_explicit_generator_replay": true,
    "decode_original_image_max_error": 0.0,
    "current_validity": 1.0,
    "nominal_hinge": 0.01386812049895525,
    "sensitivity": 1.2401543855667114,
    "r05_objective": 0.6339452862739563,
    "r05_gradient_norm": 0.4940522313117981,
    "finite_difference": 0.49185752868652344,
    "analytic_directional": 0.4940522313117981,
    "relative_error": 0.004442248179807478
  }
]
```

## Numerical results

Primary FV is the mean over the same nine severity-1 model scenarios; sample std reflects scenario variability. Rotation/noise use three realizations; translation replicates are deterministic duplicates. Full severity0/0.5/1/1.5/2 outcomes are in future_per_model.csv. External severity is distinct from r. All test grid rows are predeclared diagnostics; only calibration determines the selected point.

```
        method  selected    r  beta  feasible  converged  current_validity  FV_mean   FV_std  pixel_cost  nominal  sensitivity  surrogate
   r0.25_beta1     False 0.25  1.00      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
r0.25_beta1.05     False 0.25  1.05      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
 r0.25_beta1.1     False 0.25  1.10      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
 r0.25_beta1.2     False 0.25  1.20      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
    r0.5_beta1     False 0.50  1.00      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
 r0.5_beta1.05     False 0.50  1.05      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
  r0.5_beta1.1     False 0.50  1.10      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
  r0.5_beta1.2     False 0.50  1.20      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
      r0_beta1      True 0.00  1.00      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
   r0_beta1.05     False 0.00  1.05      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
    r0_beta1.1     False 0.00  1.10      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
    r0_beta1.2     False 0.00  1.20      True       True               1.0 0.960069 0.119792    5.575461 0.000000     0.000000   0.000000
          Base     False 0.00   NaN      True       True               1.0 0.960069 0.119792    5.575460 0.013868     1.240154   0.013868
```

Calibration: 12/12 final iterates feasible, 12/12 meet the implemented stopping condition. Test: 12/12 feasible, 12/12 meet the stopping condition. Failed rows remain recorded and are not deployed or repaired. The stopping condition is exact-geometry feasibility plus small successive objective change, as in Algorithm1; it is not a global-optimality or stationarity certificate. Final objective-gradient norms and full traces are saved. A pure objective gradient can be nonzero at a constrained optimum; do not interpret it as a KKT residual.

## Fixed protocol and validation

The update operator, classifier, AE, training/factual/target IDs and all calibration/test future model checkpoints are unchanged from the prior pilot. Updates are the complete prescribed T=3 full-batch SGD steps (lr0.01) on a fixed balanced256-image empirical distribution, all classifier parameters trainable, frozen BN statistics, disabled dropout. This is not full-epoch MNIST retraining. Future checkpoints are reused read-only and hash-checked; no model retraining occurred. Previous final outcomes were already inspected, so this is explicitly retrospective.

Optimization and W2 are in the64-dimensional decoder latent domain. Perturbations underlying data-side sensitivity are in training-image pixel space. Dual norm is sqrt(N)*||dF/dh||_2 for the empirical RMS norm, not the unscaled tensor norm. Q has64 uniform support points and exact one-to-one Base→Q assignment. The augmented-Lagrangian settings remain primal step20 and rho1; maximum iterations increases to800 to test the previous iteration-limit problem. Exact unregularized OT rechecks each final geometry. Retained derivative graphs are released before final scalar metrics, fixing the earlier memory-cap error without changing objective values.

Ten focused tests cover returned-code preservation, smooth and hard objective gradients, dual scaling, geometry, and assignment. Actual CNN+decoder directional finite differences are recorded in base_recovery.json. Independent reporting verifies all24 permutations, unchanged Q under assignment, exact geometry totals and all source hashes. All8 GPUs are verified in worker logs; each worker caps its allocator at6.5% of device memory. Frozen inputs and existing runs remain untouched.

## Artifacts and reproduction

Task: tasks/012_mnist_latent_base_fix.md. Config: experiments/configs/mnist_latent_base_fix.json. Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/20260924_smooth_v1`. Code: experiments/src/lift_ce/experiments/mnist_latent_base_fix.py; shared objective counterfactuals/update_aware.py; generator code bookkeeping counterfactuals/vision.py. Summary CSVs and provenance live here; model checkpoints remain at their original external paths. Figures: figures/mnist_latent_base_fix/comparison.pdf and qualitative.pdf (first16 fixed factuals, including poor visual cases).

```
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/056_run_mnist_latent_base_fix.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_fix_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/057_report_mnist_smooth.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/NEW_RUN_ID
```

Use a new raw path and a separate checkout for report reproduction, since the reporter refuses to overwrite its existing report. Remaining limits include a single old classifier/AE,64 factuals, short model updates, nonsmooth CNN activations despite smooth hinge, the first-order/local nature of the uncertainty surrogate, and absence of independent semantic digit labels for CF outputs. No claim of certified worst-case validity or global optimization is justified.
