# ElliCE warning diagnosis and corrected follow-up

The observed German/logistic warning is not removable at fixed epsilon0.05 by
running longer. All62 calibration cases remain native-nonrobust with default
1000iterations/patience50, extended3000/patience300, and lr.03/3000/patience300.
All remain currently valid. At epsilon.005 and.01 all62 are native robust without
warnings under all three optimizer configurations.

Independent fixed-ellipsoid mixed-integer conic optimization over the complete
numeric training box and legal one-hot categories yields maximum robust logit
-0.2565163, global upper bound -0.2565138, within solver numerical tolerances.
Thus no hard-category robust CF exists at this epsilon for that classifier and
domain. Continuous relaxation attains+1.07706; rounding that relaxed optimum
produces-0.41276. Relaxed success is not evidence of legal categorical robustness.
See hard_feasibility.json, feasibility_bound.json and scripts099/100.

A separately declared fix admits only calibration-native-feasible epsilon
settings, then uses the original calibration FV/cost rule. It selects epsilon.01
with the official default optimizer and explicit requires='robust'. This changes
the selection eligibility rule; it is not a claim that epsilon.05 was solved.
Settings and43 new CFs were frozen before accessing final future models.
All43 test outputs have legal categories and positive independently recomputed
ElliCE worst-case logit; test FV100%, mean normalizedL2 cost10.73994.
Raw: /home/xu.zhao/robust_ce_results/iclr_runs/ellice_convergence/20260925_feasible_native_v1.

Additional calibration audit: Diabetes/MLP epsilon.05 passes44/50, German/MLP
5/52. Both pass all cases at epsilon.005/.01; their original calibration-selected
settings are already .005 and.01 respectively, so those high-epsilon warnings do
not invalidate the selected configurations. No infeasibility proof is claimed
for these nonlinear classifiers. Other Task100 workers are still running.

Original raw results and current live core are preserved. Reporter095 now records
native ElliCE robust fraction separately from future validity, preventing a
nonempty currently valid output from being mistaken for native robust success.
