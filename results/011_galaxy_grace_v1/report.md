# Galaxy GRACE: original LeapFactual multiclass assets



Completed on one server and eight GPUs. No historical runs were modified; no base classifier, encoder or decoder was retrained. This is a finite-bank/local-solver replication, not an exact WRO solution.

## Asset reuse and task

Original VGG16_BN trained on the 20% Galaxy10 training folder, checkpoint epoch47-step624; WAE_MMD64 version8 last checkpoint (epoch299); original conditional flow MLP finetune weights and progressive LeapFactual generator. Exact paths/hashes: model_manifest.json and galaxy_asset_inventory.md. Ten target-label recourse tasks preserve the original ten-way classifier. Each has 16 calibration factuals, 16 test factuals and 16 target-class training references; no test outcomes select those cohorts.

## Calibration-selected configuration

| Task                             |   target |   r_star |   r_index |   beta_geo_star |   calibration_max_mean_hinge |
|:---------------------------------|---------:|---------:|----------:|----------------:|-----------------------------:|
| Disturbed Galaxies               |        0 |        4 |         1 |            1.05 |                    0.0277283 |
| Merging Galaxies                 |        1 |        4 |         1 |            1.05 |                    0.157582  |
| Round Smooth Galaxies            |        2 |        2 |         0 |            1    |                    0.445821  |
| In-between Round Smooth Galaxies |        3 |        2 |         0 |            1.05 |                    0.0881809 |
| Cigar Shaped Smooth Galaxies     |        4 |        2 |         0 |            1.05 |                    0.028407  |
| Barred Spiral Galaxies           |        5 |        4 |         1 |            1.05 |                    0.417245  |
| Unbarred Tight Spiral Galaxies   |        6 |        4 |         1 |            1.05 |                    0.418831  |
| Unbarred Loose Spiral Galaxies   |        7 |        2 |         0 |            1.1  |                    0.327677  |
| Edge-on Galaxies without Bulge   |        8 |        2 |         0 |            1    |                    0.393476  |
| Edge-on Galaxies with Bulge      |        9 |        2 |         0 |            1    |                    0.412507  |

For each target, r_star and beta_geo_star minimize worst calibration-model mean hinge, with the predeclared tie-break. All 80 grid rows are saved. These configurations remain fixed across every final severity.

## Main result at nominal external severity

| method         |   Future_Validity |   Argmax_Validity |    Cost |   Phi_hat_r |   Construction_Seconds |   Seconds_per_test_CF | Runtime_Convention                                     |
|:---------------|------------------:|------------------:|--------:|------------:|-----------------------:|----------------------:|:-------------------------------------------------------|
| Base           |          0.285938 |          0.3125   | 16.7346 | 0.293168    |                647.237 |               4.04523 | sum generator GPU seconds for both splits              |
| GRACE_beta1.00 |          0.475    |          0.498437 | 26.8872 | 0.131474    |               5375.89  |              33.5993  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_beta1.05 |          0.614062 |          0.61875  | 21.9325 | 0.00145877  |               5375.89  |              33.5993  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_beta1.10 |          0.610938 |          0.614062 | 22.0027 | 0.000920628 |               5375.89  |              33.5993  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_beta1.20 |          0.610938 |          0.614062 | 22.0027 | 0.000920628 |               5375.89  |              33.5993  | shared complete pipeline wall seconds; do not sum rows |
| GRACE_selected |          0.6      |          0.603125 | 24.1702 | 0.01952     |               5375.89  |              33.5993  | shared complete pipeline wall seconds; do not sum rows |
| LeapFactual    |          0.496875 |          0.521875 | 34.2753 | 0.162958    |                785.315 |               4.90822 | sum generator GPU seconds for both splits              |

Macro means over ten target tasks, two shift families and two realizations. Primary validity requires target softmax probability ≥0.5; target-argmax validity is separately reported, preserving the original multiclass diagnostic. Cost is factual-to-returned-image pixel L2. The full operating pipeline and calibration grid are shared across GRACE rows, so their runtimes must not be summed. Base/LeapFactual generator times are GPU-worker seconds; GRACE is complete construction wall time; these different runtime conventions are explicit rather than treated as a speedup comparison.

## Geometry budget ablation

|   beta_geo |   Future_Validity |    Cost |   Phi_hat_r |   shared_construction_seconds |
|-----------:|------------------:|--------:|------------:|------------------------------:|
|       1    |          0.475    | 26.8872 | 0.131474    |                       5375.89 |
|       1.05 |          0.614062 | 21.9325 | 0.00145877  |                       5375.89 |
|       1.1  |          0.610938 | 22.0027 | 0.000920628 |                       5375.89 |
|       1.2  |          0.610938 | 22.0027 | 0.000920628 |                       5375.89 |

Per-task nonincreasing retained shared-bank risk: {'0': True, '1': True, '2': True, '3': True, '4': True, '5': True, '6': True, '7': True, '8': True, '9': True}. Nested candidate pooling enforces the ordering among retained feasible candidates; it does not prove global optimality. All exact W2 checks use the declared relative tolerance 1e-6 × max(1,D). No current-model validity constraints or repair are used.

## Representation and image quality

All geometry is unregularized empirical W2 in the same original64-dimensional Euclidean latent space. Classifiers always score decoded RGB images. Asset audit reconstruction MSE=0.002179; classifier reconstruction decision agreement=70.31%; audit classifier accuracy=67.19%. These modest semantic checks limit interpretation despite small pixel error. Per-task Base round-trip validity, reconstruction MSE, pixel cost, latent displacement and PSNR are saved. The published qualitative panel uses the first frozen factual for each target, not outcomes. No quantitative morphology-validity claim is inferred from visual appearance alone.

## External shifts

Exposure and additive imaging noise are a NEW predeclared evaluation suite, not an established LeapFactual robustness benchmark. Fixed updates train on shifted original training data; returned CF pixels are frozen. Exposure/noise severity is unrelated to internal r. Brightness replicas and severity0 are deterministic duplicates, not independent statistical replication. No uncertainty bands across those copies are claimed.

brightness: Base: 43.75% at severity 0, 38.75% at severity 2; GRACE_selected: 96.25% at severity 0, 75.62% at severity 2. These endpoints describe this fixed update protocol; intermediate levels are reported in shift_severity.csv.

noise: Base: 43.75% at severity 0, 19.38% at severity 2; GRACE_selected: 96.25% at severity 0, 30.94% at severity 2. These endpoints describe this fixed update protocol; intermediate levels are reported in shift_severity.csv.

## Compute and runtime

{
  "construction_wall_seconds": 5375.885952472687,
  "construction_wall_seconds_per_test_cf": 33.599287202954294,
  "construction_GPU_hours": 9.390480188330015,
  "future_GPU_hours": 0.21200986484686535,
  "all_worker_GPU_hours": 9.606756628221936,
  "runtime_scope": "full grid and ten-task pipeline including compatible baseline; shared costs counted once; pretrained training and final future evaluation excluded from construction"
}

Construction runtime starts from pretrained assets and includes original Base/LeapFactual generation, bank construction, calibration models, all calibration solves, r/beta selection and final refinement/freezing. It excludes the preliminary audit and final future evaluation. GPU-hours are summed worker occupancy wall time, not a hardware utilization integral. Detailed per-job timing and commands are saved.

## Failures, limitations and checks

All75 jobs and independent checks passed. Sixteen full-update attack models are persisted; each candidate is scored against every radius-feasible model in the shared bank. All final future values and finite-bank risks were recomputed from stored checkpoints/pixels (max errors 0, 8.32667e-17). One-step minibatch attack hypergradients, only128 mutable training images per restart, and two-epoch complete updates restrict the adversary/model-update family. These approximations are explicit. Original checkpoint validation reused the old test folder; final future scenarios were untouched until freezing, but the pretrained classifier is not a new clean-split benchmark. Sixteen factuals per target and a single frozen f0 limit statistical generalization. GPU memory caps preserve coexistence with existing workloads; no jobs were killed.

Raw results: `/home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_full_v1`. Tables, configuration, manifests and publication PDF/PNG figures are saved. No manuscript scientific claims were changed.
