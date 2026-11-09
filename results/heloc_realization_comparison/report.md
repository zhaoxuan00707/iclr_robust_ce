# HELOC fixed-Q realization comparison

Completed the audit, archived OLD reproduction, and the mathematically compatible **weighted** realization comparison for all four betas. The requested exactly-one-CF-per-factual, Q-preserving assignment is **not implementable on these saved artifacts**: each split has 128 equally weighted Base CFs, while Q has 3,336 distinct-support particles. We did not change Q, resample it, duplicate factuals into fictitious observations, or barycentrically collapse transport atoms. A 128-row output cannot have this 3,336-particle marginal. NEW-DIRECT below is therefore explicitly NEW-DIRECT-weighted, not the requested permutation variant. No deterministic OLD-1CF rule exists in the evaluated old pipeline; none was invented.

Raw outputs: `/home/xu.zhao/robust_ce_results/iclr_runs/heloc_realization_comparison/20260918_v1`.
Newest fixed Q source: `/home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2/heloc`.
Archived completed OLD source: `/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc`.

The archive used earlier Q supports. Its selected paths were reconstructed with the exact existing `glued_arclength_path` and compared elementwise to saved paths. Current validity, recourse cost and all 45 aligned plus 45 scratch future-model outcomes per beta reproduce with maximum absolute error 4.44e-16. The original 5,387-model bank risks reproduce within 5.55e-17. Details: old_reproduction.csv and raw beta*/archived_bank_reproduction.json.

The newest raw-Q run had no completed OLD path or evaluation. Thus the primary OLD rows are **OLD-native-replayed on newest Q**, using the actual archived path function and `exact_lambda_choices` calibration rule, with all methods evaluated on the common newest 5,927-model bank. Lambda is selected on the existing calibration factual population over 0:0.05:1, minimizing exact bank hinge and breaking exact ties toward smaller lambda. No current-validity filter was added to OLD. This is a disclosed replay, not a claim that historical metrics existed for the newest Q. No single beta was calibrated in the archive; beta remains a diagnostic axis. Consequently there is no legitimately selected beta row to highlight.

## Primary numerical comparison

Operator: existing aligned full update. Mean and sample standard deviation are over the same 45 future models; this is variation across heterogeneous shift scenarios, not a standard error or 45 independent dataset seeds. Only HELOC seed 1 is available. Scratch-trained stress models are kept separate in summary.csv.

```
 beta_geo                    method  current_validity  future_validity_mean  future_validity_std  recourse_cost  base_displacement  train_bank_risk  W2_deploy_Q  runtime_seconds
     1.00       OLD-native-replayed          0.845792              0.673337             0.248790       3.375761           2.979286         0.480328 1.671483e+00         0.325776
     1.00       NEW-DIRECT-weighted          1.000000              0.744691             0.244538       4.884089           4.583517         0.488634 0.000000e+00         0.159404
     1.05       OLD-native-replayed          0.961031              0.833790             0.172618       4.130930           3.761792         0.408748 8.766837e-02         0.631077
     1.05       NEW-DIRECT-weighted          0.962830              0.836438             0.170728       4.210471           3.845955         0.405891 0.000000e+00         0.165527
     1.05 NEW-DIRECT-VALID-weighted          1.000000              0.845651             0.176142       4.107244           3.719196         0.405946 7.467729e-01         1.645604
     1.10       OLD-native-replayed          0.972722              0.875080             0.132300       4.259888           3.876264         0.365321 3.376966e-02         1.535376
     1.10       NEW-DIRECT-weighted          0.971523              0.875733             0.130529       4.254918           3.870606         0.363122 3.943031e-08         0.173297
     1.10 NEW-DIRECT-VALID-weighted          1.000000              0.882856             0.134654       4.168237           3.765524         0.363148 6.963469e-01         1.394743
     1.20       OLD-native-replayed          0.982614              0.915695             0.091742       4.195458           3.771277         0.319224 2.193051e-01         2.776526
     1.20       NEW-DIRECT-weighted          0.982914              0.920903             0.086820       4.395767           3.982203         0.303359 3.606193e-08         0.166595
     1.20 NEW-DIRECT-VALID-weighted          1.000000              0.924523             0.088968       4.337348           3.911145         0.303392 6.190667e-01         0.941167
```

All rows are weighted empirical populations. Current validity, future validity, risk and costs use their transport masses, exactly as OLD-native. Recourse cost is mean Euclidean factual-to-final-CF distance in the inherited imputed/normalized 22-feature representation. Base displacement uses the factual's own Base CF. Per-factual cost plots use conditional expected cost given factual identity, not an invented deterministic output.

## Interpretation of the completed comparison

Direct transport preserves the learned Q and its construction-bank risk exactly up to numerical tolerance. It does not uniformly improve the cost–robustness tradeoff. At beta=1, it increases future validity from 67.33% to 74.47% and current validity from 84.58% to 100%, but increases cost from 3.376 to 4.884; its bank risk is slightly worse than OLD (0.4886 versus 0.4803). At beta=1.05/1.10/1.20, direct future validity gains over OLD are only 0.26/0.07/0.52 percentage points. Pure direct assignment takes about 0.16–0.17 seconds versus 0.33–2.78 seconds for native OLD path construction, with the timing caveats below.

For beta=1.05 and 1.10, the corrected weighted diagnostic is empirically promising: current validity is 100%, future validity improves from 83.38% to 84.57% and from 87.51% to 88.29%, respectively, while cost falls from 4.131 to 4.107 and from 4.260 to 4.168. At beta=1.20, correction improves future validity from OLD's 91.57% to 92.45%, restores 100% current validity, and lowers bank risk from 0.31922 to 0.30339, but cost rises from 4.195 to 4.337. These are descriptive comparisons, not a calibrated beta selection or evidence of statistical significance.

Correction affects only 3.72%, 2.85%, and 1.71% of transport mass, yet is **not negligible for affected cases**: their conditional mean alpha is 0.670, 0.681, and 0.651, and maximum movement is 10.71–11.44 normalized L2 units. W2 from corrected deployment to Q is 0.747, 0.696, and 0.619. Bank-risk increases relative to pure Q are small (0.000055, 0.000026, 0.000034), but that does not imply geometric or individual preservation. Correction time brings total realization to 1.65, 1.39, and 0.94 seconds.

All four saved Q distributions satisfy their training transport-sum budget. For beta greater than 1, the test Base-to-Q-to-P1 sums are 6.028, 6.247, and 6.535, exceeding the inherited training budgets 5.112, 5.355, and 5.842. Thus these comparisons cannot claim a test-population geometry guarantee. At beta=1, Q equals P1 in this artifact, explaining its zero second-leg distance.

Conclusion: exact weighted direct transport is a simpler way to retain population Q, and Base-directed correction improves the measured tradeoff at beta=1.05/1.10. There is no universal winner and no demonstrated deterministic individual-recourse solution. The cardinality mismatch, non-negligible corrections on affected cases, one dataset seed, and failed transfer of the geometry budget limit the conclusion.

## Construction and safeguards

OLD solves Base->Q and Q->P1 plans and uses the existing normalized-arc-length interpolation. NEW-DIRECT solves only Base->Q, retains every positive transport atom with its mass, and returns the corresponding unchanged Q point. POT's network simplex is the correct unregularized quadratic OT solver for these unequal populations; the existing SciPy minimum-cost assignment branch applies only to equal-count uniform populations. All source marginals are 1/128, target marginals are 1/3336, and every factual ID is retained. NEW-DIRECT's W2(Q_deploy,Q) is checked with an independent OT solve and its bank risk agrees with learned Q within numerical tolerance. Its assignment cost agrees with OLD's first-leg optimal cost. P1 is not an input to direct deployment or correction.

For betas with current-invalid direct mass, NEW-DIRECT-VALID uses the existing `minimal_restore` on each transport atom and its own source Base CF. Valid atoms remain bitwise unchanged. The routine enumerates ReLU-affine validity intervals in float64 and refines the first crossing with reference float32 predictions; alpha bracket tolerance is 1e-7. This avoids a monotonicity assumption. All Base anchors were verified valid. It is a numerical first-crossing search, not an interval-arithmetic certificate. Pure direct outputs were saved and evaluated before correction. No P1 and no Q reoptimization are used. Correction changes the deployed distribution, not the saved Q.

Correction sizes:

```
 beta_geo  invalid_mass  changed_atoms  alpha_mean_all  alpha_conditional  alpha_max  mean_correction_displacement  max_correction_displacement  alpha_bracket_max
     1.05      0.037170            127        0.024899           0.669871   0.994992                      0.126760                    10.711428       7.890625e-08
     1.10      0.028477             99        0.019382           0.680619   0.994802                      0.105082                    10.636411       7.890625e-08
     1.20      0.017086             59        0.011118           0.650701   0.996563                      0.071058                    11.444563       8.593750e-08
```

Q's training two-anchor geometry is identical across methods. The absolute budget is beta times the training Base-to-P1 distance, as before. Test-batch Q geometry and final deployed-population geometry are also reported; neither feasibility is silently inferred from training. See fixed_Q_geometry.csv and summary.csv. Reported inherited_fresh_subset_risk uses historical fresh-discovery models that are already included in the construction bank, so it is not independent fresh confirmation. Held-out future-bank risk is the maximum hinge across the existing future models; no new attacks were trained.

Realization timings exclude Q training, bank scoring, calibration selection and future evaluation. OLD timing includes its two OT solves plus the native 21-point path construction; NEW includes its single OT solve and atom construction; corrected NEW additionally includes first-crossing correction. These are single wall-clock measurements under concurrent workers, not precision runtime benchmarks.

## Paper/implementation inconsistencies

1. The active manuscript includes `paper/Sections/03_methods_restructured.tex`, whose Method describes a Wasserstein lens, barycentric family and held-out proxy operating-point selection. The audited newer code instead optimizes free continuous Q under a single transport-sum budget and realizes weighted two-leg paths.
2. `paper/Sections/new_method.tex` is a separate draft and is not the active Method include. Its valid-only training-bank operating-point selection and target-directed Q repair do not describe the archived evaluated OLD pipeline's calibration-bank selection without an f0 filter. The newest repair run produced only the beta=1 repaired support and no finished paths/evaluation; its proposed safeguards cannot be treated as completed deployment behavior.
3. The task's equal-cardinality assumption is contradicted by saved artifacts (128 versus 3,336). The actual OLD output is a weighted conditional CF distribution. One deterministic CF per factual cannot be asserted without an additional, distribution-changing rule.
4. The 22-dimensional continuous representation has no integer or immutable-feature actionability guarantee. Future scenarios are previously inspected retrospective models, not untouched confirmation. These limitations are retained; manuscript claims were not edited.

## Artifacts and reproduction

Summary/per-model CSVs, correction statistics, calibration choices, per-factual costs, counts, hashes and checks are in `results/heloc_realization_comparison/`. The six requested plots, separately and together as comparison.pdf, are in `figures/heloc_realization_comparison/`. Detailed transport atoms, frozen outputs, per-model bank risks, model hashes and logs stay in the external raw run. Existing outputs were preserved.

Commands (use a **new** raw run path to preserve this completed run):

```bash
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/051_run_heloc_realization.py --output /home/xu.zhao/robust_ce_results/iclr_runs/heloc_realization_comparison/NEW_RUN_ID
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/heloc_realization_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/052_report_heloc_realization.py --run /home/xu.zhao/robust_ce_results/iclr_runs/heloc_realization_comparison/NEW_RUN_ID
```

The report script refuses to overwrite this completed repository report; archive it or use a separate checkout for independent reproduction. Controller audit runs on CPU; four beta workers use physical GPUs 1,2,3,5 through experiments/run.py. Config: experiments/configs/heloc_realization_comparison.json. Exact worker commands are logged. Core: experiments/src/lift_ce/experiments/heloc_realization_comparison.py; the small direct adapter is counterfactuals/direct_realization.py. Existing geodesic_excess.py, population_path.py, minimal_validity_repair.py, and fixed_risk_geometry_v1.py supply unmodified OLD, weighting, correction and lambda selection logic.
