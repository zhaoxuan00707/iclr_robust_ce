# FFHQ hard-content bottleneck GRACE

Raw run: `/home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1`. Parent frozen run: `/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1`. Completed on eight GPUs. This is a matched follow-up on eight generated faces with previously evaluated future models, not a new unseen-shift benchmark.

## LeapFactual reuse and decomposition

Original StyleGAN3, CLIP, flow-generated Base CFs, factual IDs, targets, training/calibration populations, attack bank, calibration models and future models are reused. No original FFHQ task/content decomposition or identity model exists; see identity_bottleneck_inventory.md. New ArcFace is an evaluation component only. No identity loss or current-validity branch is present.

Training-only gradient SVD retains 29 of512 dimensions at95% energy. Task coordinates u=hU; immutable content v=h_fact−h_fact UUᵀ; synthesis h′=v+u′Uᵀ. Target geometry uses only target u. The optimizer changes only u, with max_model mean_CF hinge and the sole beta×D geometry constraint. Original Base can change when realized with factual v; Conditional_Base is reported separately in all_results.csv.

## Bottleneck diagnostics

|   smile_probe_u |   smile_probe_u_balanced |   smile_probe_v |   smile_probe_v_balanced |   task_dimension |   content_dimension |   latent_dimensions |   gradient_energy |   calibration_majority_accuracy |   reconstruction_error |   reconstruction_max_error |   reconstruction_latent_max_error |
|----------------:|-------------------------:|----------------:|-------------------------:|-----------------:|--------------------:|--------------------:|------------------:|--------------------------------:|-----------------------:|---------------------------:|----------------------------------:|
|        0.765625 |                 0.765625 |          0.5625 |                   0.5625 |               29 |                 483 |                 512 |          0.950744 |                             0.5 |                      0 |                          0 |                                 0 |

Probes use the same512-input logistic architecture, C=1, no feature scaling, training256/calibration64. These are CLIP pseudo-labels from prompts “smiling face” versus “face”, not annotated smile labels. Complementary content is not assumed classification-independent. Reconstruction uses the64 calibration latents and the original generator.

## Calibration

{
  "r_star": 2.0,
  "r_index": 1,
  "beta_geo_star": 1.05,
  "bottleneck_type": "approximate training-gradient SVD",
  "bottleneck_dimension": 29,
  "basis_path": "/home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1/checkpoints/task_basis.pt",
  "basis_sha256": "ef23089404ddd704f45e2d609f59f1538fc07c66f6462b0d0b6b6ba48f61cb5c",
  "selected_utc": "2026-09-17T08:50:34.297003+00:00",
  "future_outcomes_used": false,
  "rule": "minimum maximum calibration-model mean hinge rounded to1e-6; ties lower pixel224L2, r, beta"
}

Both r and beta were selected using only four saved calibration models. Full-latent GRACE is matched to the same selected r, beta and bank; its stored output is reused when available. Original full-latent candidates and new task-space candidates use the same solver settings and candidate-pooling policy.

## Table 1: matched results at nominal severity

| method           |   Future_Validity |   pixel_L2_224 |   Identity_Similarity |   identity_std |   identity_p10 |   LPIPS_squeeze |   runtime_worker_seconds |
|:-----------------|------------------:|---------------:|----------------------:|---------------:|---------------:|----------------:|-------------------------:|
| Base             |          0.458333 |        38.7439 |              0.788858 |      0.160239  |       0.625445 |        0.132648 |                  124.68  |
| Full_GRACE       |          1        |        92.9133 |              0.20711  |      0.0935036 |       0.113464 |        0.382032 |                  427.204 |
| Bottleneck_GRACE |          1        |       128.818  |              0.244136 |      0.038651  |       0.213216 |        0.48935  |                  468.338 |

Cost is factual-to-output pixel L2 at224RGB. Identity is cosine of512-dimensional ArcFace embeddings after independent five-landmark alignment; std uses ddof=0, low percentile is10th percentile. Largest detected face is used. Missing detections are reported in identity_metrics.csv; per-example detection counts and similarities are retained. Similarity is a metric, not proof of identity.

## Table 2: task-space geometry budget

|   beta_geo |   Future_Validity |   Phi_hat_r |   pixel_L2_224 |   Identity_Similarity |   LPIPS_squeeze |   runtime_worker_seconds |
|-----------:|------------------:|------------:|---------------:|----------------------:|----------------:|-------------------------:|
|       1    |                 1 |           0 |        128.818 |              0.244136 |         0.48935 |                  468.338 |
|       1.05 |                 1 |           0 |        128.818 |              0.244136 |         0.48935 |                  468.338 |
|       1.1  |                 1 |           0 |        128.818 |              0.244136 |         0.48935 |                  468.338 |
|       1.2  |                 1 |           0 |        128.818 |              0.244136 |         0.48935 |                  468.338 |

## Robustness versus identity tradeoff

{
  "selected_r": 2.0,
  "selected_beta": 1.05,
  "task_dimension": 29,
  "full_FV_gain_over_base": 0.5416666666666667,
  "bottleneck_FV_gain_over_base": 0.5416666666666667,
  "retained_fraction_of_full_FV_gain": 1.0,
  "identity_similarity_gain_vs_full": 0.03702594339847559,
  "LPIPS_reduction_vs_full": -0.10731776058673859,
  "hard_content_max_drift": 1.029543798578266e-07
}

Positive identity-similarity gain versus full GRACE indicates less measured drift. The retained robustness-gain fraction compares gains above the original Base at nominal severity; it is descriptive on this fixed small cohort and is not a statistical guarantee. Negative or small gains must not be described as successful identity preservation.

## Runtime

New construction wall time: 315.28s. Total new worker occupancy: 0.6554 GPU-worker hours, including reconstruction and evaluation. Table1 runtime is worker-seconds for the entire eight-CF population: original Base generation, full-latent test optimization/pooling, or bottleneck test optimization/pooling. Shared basis/bank/calibration costs are excluded from method rows and itemized in runtime.csv. Reused historical costs are reported, not rerun; bottleneck beta rows share pooled candidates, so runtime must not be summed across beta.

## Qualitative panels

All eight prespecified test IDs are shown, in fixed order, in ffhq_identity_qualitative.pdf and ffhq_identity_beta.pdf. The panels expose facial appearance, hair, pose, background and smile changes. No favorable-example selection was used. Visual identity-preservation claims require inspection and are not inferred from latent freezing.

## Limitations

The SVD split is approximate and classifier-dependent; task directions can still encode appearance and content directions can retain smile information. Generated FFHQ-style faces, prompt-relative labels, small populations, projection-only Update and finite attack bank limit generalization. Face alignment/detection and ArcFace domain behavior affect identity scores. The historical brightness/blur replicas include deterministic duplicates. Reusing previously evaluated future models yields a controlled follow-up, not independent replication. The central GRACE objective/theory and StyleMNIST/Galaxy runs are unchanged.

## Independent verification

{
  "status": "passed",
  "jobs": 56,
  "gpu_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "source_hashes_unchanged": true,
  "basis_training_only": true,
  "basis_energy_rule_recomputed": true,
  "calibration_selection_recomputed": true,
  "all_geometry_feasible": true,
  "max_content_coordinate_drift": 1.029543798578266e-07,
  "max_bank_risk_error": 3.576278686245882e-07,
  "max_future_probability_error": 1.1682510375976562e-05,
  "max_identity_cosine_error": 1.1102230246251565e-16,
  "max_decoded_image_error": 0.0,
  "identity_missing_pairs": 45,
  "all_test_ids_displayed": true
}
