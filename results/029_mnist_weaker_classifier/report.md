# MNIST weaker-classifier matched comparison

Weak checkpoint selected before CF generation: seed153, step16; validation accuracy89.84%, original99.61%. Official digit4/9 test diagnostics after checkpoint freeze: {'strong': {'accuracy': 0.996986448764801, 'recall4': 0.9969450235366821, 'recall9': 0.9970267415046692, 'bce': 0.011470149271190166}, 'weak': {'accuracy': 0.8869914412498474, 'recall4': 1.0, 'recall9': 0.7770069241523743, 'bce': 0.528203010559082}}.

Mean direct image L2 change between r0 andr.25 is 0.138931 for the strong arm and 0.000013 for the weak arm (ratio 9.33e-05). Mean absolute per-pixel changes are 0.001011 and 1.07e-07. Larger measured differences do not establish better-looking or semantically correct CFs.

## Main finding and stopping limitation

This experiment does **not** support the proposed visual-improvement hypothesis. The weaker model produces different Base digits, but almost no additional difference between radii under the fixed protocol. It also needs a larger Base factual-to-CF pixel cost: 6.4254 versus 5.4035 for the strong control. That larger initial edit must not be attributed to GRACE.

Weak Base already has100% FV across the prescribed36 test scenarios and sensitivity0.002805, before refinement. Every weak calibration/test solve stops after two objective evaluations (one support update): the absolute objective change is below1e-6 because the implementation divides by max(1,abs(previous objective)). Its gradient is small but nonzero. Therefore the near-zero visual change is partly a stopping-scale limitation, and is **not evidence that a fully optimized weak-classifier problem must have no refinement**. The objective is not zero, and no global optimum or stationarity is established. Checkpoint weakness, different Base locations, and early stopping are all involved; accuracy alone is not isolated as a causal explanation.

The weak arm selects r0 because calibration FV is100% for all radii and lower cost breaks the tie. The strong arm selects r.25, improving test FV from85.55% (Base) through97.87% (nominal-only) to100%. Weak-model100% FV is only validity under its own short future updates, not superior predictive performance or a robustness certificate.

Recommended next experiment, if needed: a separately predeclared **matched solver-tolerance audit** for both arms, using a scale-sensitive stopping rule or a fixed minimum iteration count, before attributing weak-arm flatness to classifier quality. Do not choose a different weak checkpoint based on these images or enlarge r solely to force visibility. For current presentation use the strong-arm sensitivity/FV/difference-map figure, and keep this weaker-arm result as an honest diagnostic.

## Population results

Same64 factuals per split, same decoder/reference population/update data, beta1.02 and r0/.05/.10/.25. Sensitivity and FV below are population-level; they are not scores of individual images. Each arm selects radius on its own calibration FV with cost tie-break before test scoring.

```
   arm         method  selected  feasible  converged  current_validity  FV_mean   FV_min  pixel_cost  nominal  sensitivity  surrogate  strong_classifier_agreement
strong r0.05_beta1.02     False      True       True               1.0 0.996962 0.906250    5.452536 0.000072     0.005400   0.000342                          1.0
strong  r0.1_beta1.02     False      True       True               1.0 0.998698 0.953125    5.460044 0.000046     0.003147   0.000361                          1.0
strong r0.25_beta1.02      True      True       True               1.0 1.000000 1.000000    5.476827 0.000025     0.001437   0.000384                          1.0
strong    r0_beta1.02     False      True       True               1.0 0.978733 0.531250    5.434035 0.000305     0.029656   0.000305                          1.0
strong           Base     False      True       True               1.0 0.855469 0.031250    5.403529 0.009974     1.063635   0.275883                          1.0
  weak r0.05_beta1.02     False      True       True               1.0 1.000000 1.000000    6.425387 0.002173     0.002805   0.002313                          1.0
  weak  r0.1_beta1.02     False      True       True               1.0 1.000000 1.000000    6.425387 0.002173     0.002805   0.002453                          1.0
  weak r0.25_beta1.02     False      True       True               1.0 1.000000 1.000000    6.425390 0.002173     0.002805   0.002874                          1.0
  weak    r0_beta1.02      True      True       True               1.0 1.000000 1.000000    6.425386 0.002173     0.002805   0.002173                          1.0
  weak           Base     False      True       True               1.0 1.000000 1.000000    6.425374 0.002173     0.002805   0.002173                          1.0
```

Calibration metrics:
```
   arm    r   cal_FV  pixel_cost  nominal  sensitivity  surrogate  feasible  converged
strong 0.05 0.977865    5.632762 0.000076     0.005794   0.000366      True       True
strong 0.10 0.990451    5.639774 0.000049     0.003395   0.000389      True       True
strong 0.25 0.997830    5.653274 0.000027     0.001565   0.000418      True       True
strong 0.00 0.919705    5.612719 0.000340     0.033051   0.000340      True       True
  weak 0.05 1.000000    6.415926 0.002155     0.002788   0.002295      True       True
  weak 0.10 1.000000    6.415927 0.002155     0.002788   0.002434      True       True
  weak 0.25 1.000000    6.415930 0.002155     0.002787   0.002852      True       True
  weak 0.00 1.000000    6.415926 0.002155     0.002788   0.002155      True       True
```

Direct image-difference summaries:
```
   arm         method   reference  mean_L2  median_L2     mean_MAE  maximum_pixel_change
strong r0.25_beta1.02 r0_beta1.02 0.138931   0.123754 1.011116e-03              0.274224
  weak r0.25_beta1.02 r0_beta1.02 0.000013   0.000011 1.072542e-07              0.000014
```

## Design and limits

Task tasks/015_mnist_weaker_classifier.md and config experiments/configs/mnist_weaker_classifier.json were written before running. Raw run `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/20260924_v1`. The weak SmallCNN is trained on128 balanced images, Adam lr.001/batch32, seeds150–157, checkpoints4/8/16/32/64/128. All48 candidates are retained. Selection uses512 balanced validation images excluded from weak training, update data and final P1; closest to90% accuracy among80–95% and at least3pp below original, ties fewer steps then seed. Neither appearance nor CF robustness chooses the checkpoint. Validation may overlap the historical strong classifier's training; its exact training provenance is unknown. One selected weak model is a designed stress test, not evidence that classifier accuracy alone causally controls visual quality.

The matched factuals are true4 images rejected by both models, prioritizing old IDs then remaining fixed pool order. Conditioning on agreement excludes some difficult cases; do not extrapolate to all digits. Common true9 references are accepted by both models in pixel/decoded space. Each arm generates new Base images/codes with the identical96-step latent prototype generator, because the current classifier has changed. No old Base/output is overwritten; Base regeneration is distinct from post-optimization repair. The autoencoder and256-image update distribution are frozen. Thus the matched strong control is newly generated too, and task014 headline scores are not a paired comparison.

Each arm uses the complete prescribed3 full-batch SGD steps lr.01 with frozen BN/dropout disabled, BCE training loss and smooth probability hinge t=.05 for nominal and sensitivity. All64 Q atoms are uniform; exact one-to-one assignment; latent64-D W2; training perturbations pixel-space with sqrt(256) empirical RMS dual scaling. ALM primal20/rho1/max800, feasibility1e-5*max(1,D) and objective-change1e-6. Final failures excluded, no repair. Small objective change is not a stationary/global-optimum certificate.

Paired future shifts use the same new cal51000/test61000 seeds, rotation/translation/noise, severity .5/1/1.5/2, three realizations.144 new checkpoint scenarios (72 per arm), with deterministic translations duplicated across seeds/splits. Both calibration choices are frozen before final scoring. FV is measured under each arm's own updated classifier;100% weak-model FV need not mean stronger semantic validity. Frozen strong-classifier agreement is a non-independent diagnostic, not human digit-label ground truth. Short-update, single-pair, retrospective data/decoder study; no certified or arbitrary-retraining robustness claim.

## Checks, figures and reproduction

Ten focused tests pass. Actual CNN outer derivatives are in each arm's base_recovery.json. Independent checks verify16 assignments/geometry/objective arithmetic, decoded images/current validity/cost, direct pixel differences, paired IDs and populations, checkpoint selection and hashes, frozen evaluations and GPUs0–7. Allocator cap6.5% per worker. Large training candidates and raw outputs remain external. No manuscript claims changed.

Figures: figures/mnist_weaker_classifier/qualitative.pdf shows fixed-first8 paired cases plus signed difference maps; all64_qualitative.pdf includes all64 test examples; radius_qualitative.pdf shows every feasible radius on the same first8 cases; population_metrics.pdf shows joint population sensitivity, FV and measured image change. Shared difference scale is recorded; difference maps magnify subtle changes and must not be presented as ordinary CF images. No attractive-case filtering.

```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/062_run_mnist_weaker_classifier.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_weak_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/063_report_mnist_weaker_classifier.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/NEW_RUN_ID
```
Use a separate checkout for report reproduction; existing report is protected.
