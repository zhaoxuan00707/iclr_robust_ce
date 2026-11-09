# ElliCE convergence diagnostic

User requested fixing ElliCE warnings. Preserve Task100 live source and outputs.
Use only seed0 German/logistic calibration factuals and the frozen Task100 model,
training representation, native epsilon grid .005/.01/.05, regularization1e-4,
feature bounds and categorical constraints. Do not inspect test scores to tune.
Run official continuous generator with three optimizer configurations:
- default: lr.1, patience50, max1000;
- longer: lr.1, patience300, max3000;
- smaller_step: lr.03, patience300, max3000.
Pair within-case randomness (seed90000+case index) across configurations.
Independently measure returned nominal logit, analytic worst-case ellipsoid logit,
hard-category legality, L2 cost, warnings, and wall time. A nonempty returned
DataFrame is not sufficient to establish validity or robust convergence.
Do not shrink epsilon or increase ellipsoid regularization just to remove warnings.
Select optimizer configuration by calibration robust-success count, then mean
robust margin and cost. Keep all cases; no hidden fallback or validity repair.
Save raw artifacts externally and diagnostics in results/101_ellice_convergence.
This bounded screen diagnoses the observed German/logistic warning. Extend only
if needed; do not silently overwrite the main comparison with diagnostic outputs.
