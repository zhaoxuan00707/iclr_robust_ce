# FFHQ result interpretation

All four beta values return the same retained **geodesic midpoint** population (`geodesic0.5`). It already has zero finite-bank hinge at beta1.00. Thus this run provides **no evidence of benefit from additional geometry freedom**. Calibration selected r=1, beta=1.00.

At nominal external severity, prompt-relative Future Validity is45.83% for Base,83.33% for LeapFactual and100% for selected GRACE. Mean pixel224L2 costs are38.74,54.64 and92.91 respectively; GRACE costs approximately2.40× Base. LPIPS is0.133,0.202 and0.382. All test comparisons use the same8 factuals.

The fixed qualitative grid was inspected after numerical verification. The GRACE columns show substantial changes in facial appearance, hair, clothing and background. This is a material limitation, not evidence of identity preservation. No identity model was available; the identity CSV explicitly reports unavailable values. The original CLIP prompt pair also misclassifies the visibly smiling audit face relative to literal smile appearance, so validity is only the original classifier's prompt-relative outcome.

Selected GRACE remains100% under the evaluated brightness/noise levels, but falls to87.5% at the strongest blur. Base validity increases under stronger noise updates; therefore greater external severity is not uniformly more damaging in this fixed projection-update experiment. No confidence interval is inferred from deterministic replica copies.

Construction wall time is538.55 seconds(8.98 minutes),67.32 seconds per test CF, and0.584 summed worker GPU-hours. It includes generated-data preparation, both Base/LeapFactual generation, the attack bank, calibration, all optimization and freezing; it excludes pretraining and final future evaluation. Generator-only worker times for both splits are saved below and are not directly comparable to the complete GRACE wall interval.

{
  "candidate_origins": {
    "1.0": "geodesic0.5",
    "1.05": "geodesic0.5",
    "1.1": "geodesic0.5",
    "1.2": "geodesic0.5"
  },
  "identical_beta_outputs": true,
  "no_observed_benefit_from_extra_beta": true,
  "selected_pixel224_cost_ratio_to_Base": 2.398142819489171,
  "Base_generator_worker_seconds_both_splits": 254.70296335220337,
  "LeapFactual_generator_worker_seconds_both_splits": 390.88114047050476,
  "identity_metric": "unavailable",
  "qualitative_review": "Substantial changes in facial appearance, hair, clothing and background; visual grid does not establish identity preservation.",
  "shift_result": "Selected GRACE1.0 on all brightness/noise levels,0.875 at strongest blur; Base noise validity rises with severity, so severity is not monotonic damage under this projection update."
}
