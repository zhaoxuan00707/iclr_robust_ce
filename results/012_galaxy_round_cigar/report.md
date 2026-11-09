# Galaxy GRACE: original LeapFactual multiclass assets



Completed on one server and 4 GPUs assigned to this run (eight across the concurrent pair runs). No historical runs were modified; no base classifier, encoder or decoder was retrained. This is a finite-bank/local-solver replication, not an exact WRO solution.

## Asset reuse and task

Original VGG16_BN trained on the 20% Galaxy10 training folder, checkpoint epoch47-step624; WAE_MMD64 version8 last checkpoint (epoch299); original conditional flow MLP finetune weights and progressive LeapFactual generator. Exact paths/hashes: model_manifest.json and galaxy_asset_inventory.md. The configured source-target cohorts preserve the original ten-way classifier. Each has 16 calibration factuals, 16 test factuals and 16 target-class training references; cohorts follow the fixed random seed, without pair-outcome selection.

Source classes by target: {"2": 4, "4": 2}. These are pair-restricted recourse cohorts, not newly trained binary classifiers. Calibration/future checkpoints are reused from /home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_full_v1 after verifying identical training-data hashes, original-model hashes and update/shift configurations. Prior ten-class outcomes did not tune this pair run; this is not a fresh unseen-shift replication. Runtime includes new pair construction and loading reused calibration models, but excludes historical shared model training.

## Cohort preparation

{
  "reason": "Cigar class has zero original training-complement calibration images",
  "policy": "For Cigar source only, first16 seeded test samples reserved for calibration; next16 for final testing; no overlap, no outcome selection. Original f0 used old test folder for validation, as already disclosed.",
  "source_code": "/home/xu.zhao/iclr_robust_ce/experiments/023_prepare_cigar_calibration.py",
  "source_sha256": "466f8e641ce34a2cac565d827880b6e0c553922ce7b2a53c63014459affa5a54",
  "utc": "2026-09-16T19:59:57.852123+00:00",
  "original_test_cigar_count": 33
}

Initial prepare failed because Cigar had no calibration-complement samples; corrected using the documented disjoint original-test reservation. Existing partial outputs were preserved; preparation then completed.

## Calibration-selected configuration

| Task                         |   target |   r_star |   r_index |   beta_geo_star |   calibration_max_mean_hinge |
|:-----------------------------|---------:|---------:|----------:|----------------:|-----------------------------:|
| Round Smooth Galaxies        |        2 |        2 |         0 |            1    |                    0.398465  |
| Cigar Shaped Smooth Galaxies |        4 |        4 |         1 |            1.05 |                    0.0108451 |

For each target, r_star and beta_geo_star minimize worst calibration-model mean hinge, with the predeclared tie-break. All calibration grid rows are saved. These configurations remain fixed across every final severity.

## Main result at nominal external severity

| method         |   Future_Validity |   Argmax_Validity |    Cost |   Phi_hat_r |   Construction_Seconds |   Seconds_per_test_CF | Runtime_Convention                                     |
|:---------------|------------------:|------------------:|--------:|------------:|-----------------------:|----------------------:|:-------------------------------------------------------|
| Base           |          0.507812 |          0.570312 | 13.5332 |  0.326451   |                129.733 |               4.05416 | sum generator GPU seconds for both splits              |
| GRACE_beta1.00 |          0.648438 |          0.679688 | 19.4034 |  0.0919537  |               2184.24  |              68.2575  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_beta1.05 |          0.75     |          0.75     | 15.8627 |  0          |               2184.24  |              68.2575  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_beta1.10 |          0.75     |          0.75     | 15.8627 |  0          |               2184.24  |              68.2575  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_beta1.20 |          0.75     |          0.75     | 15.8627 |  0          |               2184.24  |              68.2575  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_selected |          0.742188 |          0.742188 | 17.0869 |  0.00367211 |               2184.24  |              68.2575  | shared complete pipeline wall seconds; do not sum rows |
| LeapFactual    |          0.6875   |          0.6875   | 18.8244 |  0.126471   |                156.981 |               4.90564 | sum generator GPU seconds for both splits              |

Macro means over the configured target tasks, two shift families and two realizations. Primary validity requires target softmax probability ≥0.5; target-argmax validity is separately reported, preserving the original multiclass diagnostic. Cost is factual-to-returned-image pixel L2. The full operating pipeline and calibration grid are shared across GRACE rows, so their runtimes must not be summed. Base/LeapFactual generator times are GPU-worker seconds; GRACE is complete construction wall time; these different runtime conventions are explicit rather than treated as a speedup comparison.

## Geometry budget ablation

|   beta_geo |   Future_Validity |    Cost |   Phi_hat_r |   shared_construction_seconds |
|-----------:|------------------:|--------:|------------:|------------------------------:|
|       1    |          0.648438 | 19.4034 |   0.0919537 |                       2184.24 |
|       1.05 |          0.75     | 15.8627 |   0         |                       2184.24 |
|       1.1  |          0.75     | 15.8627 |   0         |                       2184.24 |
|       1.2  |          0.75     | 15.8627 |   0         |                       2184.24 |

Per-task nonincreasing retained shared-bank risk: {'2': True, '4': True}. Nested candidate pooling enforces the ordering among retained feasible candidates; it does not prove global optimality. All exact W2 checks use the declared relative tolerance 1e-6 × max(1,D). No current-model validity constraints or repair are used.

## Representation and image quality

All geometry is unregularized empirical W2 in the same original64-dimensional Euclidean latent space. Classifiers always score decoded RGB images. Asset audit reconstruction MSE=0.002179; classifier reconstruction decision agreement=70.31%; audit classifier accuracy=67.19%. These modest semantic checks limit interpretation despite small pixel error. Per-task Base round-trip validity, reconstruction MSE, pixel cost, latent displacement and PSNR are saved. The published qualitative panel uses the first frozen factual for each target, not outcomes. No quantitative morphology-validity claim is inferred from visual appearance alone.

## External shifts

Exposure and additive imaging noise are a NEW predeclared evaluation suite, not an established LeapFactual robustness benchmark. Fixed updates train on shifted original training data; returned CF pixels are frozen. Exposure/noise severity is unrelated to internal r. Brightness replicas and severity0 are deterministic duplicates, not independent statistical replication. No uncertainty bands across those copies are claimed.

brightness: Base: 59.38% at severity 0, 34.38% at severity 2; GRACE_selected: 93.75% at severity 0, 90.62% at severity 2. These endpoints describe this fixed update protocol; intermediate levels are reported in shift_severity.csv.

noise: Base: 59.38% at severity 0, 21.88% at severity 2; GRACE_selected: 93.75% at severity 0, 59.38% at severity 2. These endpoints describe this fixed update protocol; intermediate levels are reported in shift_severity.csv.

## Compute and runtime

{
  "construction_wall_seconds": 2184.2389578819275,
  "construction_wall_seconds_per_test_cf": 68.25746743381023,
  "construction_GPU_hours": 1.667844516105122,
  "future_GPU_hours": 0.05283463769488864,
  "all_worker_GPU_hours": 1.7223665057288275,
  "runtime_scope": "full grid and configured-task pipeline including compatible baseline; shared costs counted once; pretrained training and final future evaluation excluded from construction"
}

Construction runtime starts from pretrained assets and includes original Base/LeapFactual generation, bank construction, calibration models, all calibration solves, r/beta selection and final refinement/freezing. It excludes the preliminary audit and final future evaluation. GPU-hours are summed worker occupancy wall time, not a hardware utilization integral. Detailed per-job timing and commands are saved.

## Failures, limitations and checks

All 43 jobs and independent checks passed. Sixteen full-update attack models are persisted; each candidate is scored against every radius-feasible model in the shared bank. All final future values and finite-bank risks were recomputed from stored checkpoints/pixels (max errors 0, 9.62772e-17). One-step minibatch attack hypergradients, only128 mutable training images per restart, and two-epoch complete updates restrict the adversary/model-update family. These approximations are explicit. Original checkpoint validation reused the old test folder; new pair-cohort future evaluations occur only after freezing, but the pretrained classifier is not a new clean-split benchmark. Sixteen factuals per target and a single frozen f0 limit statistical generalization. GPU memory caps preserve coexistence with existing workloads; no jobs were killed.

Raw results: `/home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_pairs_round_cigar_v1`. Tables, configuration, manifests and publication PDF/PNG figures are saved. No manuscript scientific claims were changed.
