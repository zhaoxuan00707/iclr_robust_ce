# FFHQ hard-content bottleneck GRACE — final report

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
| Base             |          0.458333 |        38.7439 |              0.827244 |      0.140103  |      0.625144  |        0.132648 |                  124.68  |
| Full_GRACE       |          1        |        92.9133 |              0.272283 |      0.0869766 |      0.174025  |        0.382032 |                  427.204 |
| Bottleneck_GRACE |          1        |       128.818  |              0.198316 |      0.0902299 |      0.0962242 |        0.48935  |                  468.338 |

Cost is pixel L2 at224RGB; lower is better. Identity is ArcFace cosine similarity after five-landmark alignment; higher is better. All methods have8/8 valid identity pairs with the audited320px detector scale, confidence threshold0.5. Identity std uses ddof=0 and the low percentile is10th percentile. LPIPS uses native1024 images and SqueezeNet. Runtime is summed worker-seconds for eight-CF test generation/refinement, excluding shared costs.

## Table 2: task-space geometry budget

|   beta_geo |   Future_Validity |   Phi_hat_r |   pixel_L2_224 |   Identity_Similarity |   identity_std |   identity_p10 |   runtime_worker_seconds |
|-----------:|------------------:|------------:|---------------:|----------------------:|---------------:|---------------:|-------------------------:|
|       1    |                 1 |           0 |        128.818 |              0.198316 |      0.0902299 |      0.0962242 |                  468.338 |
|       1.05 |                 1 |           0 |        128.818 |              0.198316 |      0.0902299 |      0.0962242 |                  468.338 |
|       1.1  |                 1 |           0 |        128.818 |              0.198316 |      0.0902299 |      0.0962242 |                  468.338 |
|       1.2  |                 1 |           0 |        128.818 |              0.198316 |      0.0902299 |      0.0962242 |                  468.338 |

All four test budgets selected the same task-anchor endpoint (geodesic1.0), so no beta effect was observed. Full GRACE selected geodesic0.5. Shared-pool beta runtimes must not be summed.

## Outcome: success criterion not met

The bottleneck retains the full nominal robustness gain (FV1.0 versus Base0.45833), but worsens mean identity similarity relative to full GRACE (0.19832 versus0.27228; paired difference−0.07397 across all8 factuals). Base similarity is0.82724. LPIPS increases from0.38203 to0.48935 and pixel224L2 from92.91 to128.82. Thus this approximate classifier-gradient bottleneck does not achieve useful identity preservation on the tested cohort.

The content restriction is numerically enforced (maximum drift1.03e-7), and factual reconstruction is exact in the saved calibration check. This does not make the complement a validated identity representation. The29 task directions retain substantial appearance information; moving them to target anchors can change facial appearance, hair, clothing and background despite freezing483 complementary coordinates.

## Conditional-Base diagnostic

| method           |   Future_Validity |   pixel_L2_224 |   Identity_Similarity |   LPIPS_squeeze |
|:-----------------|------------------:|---------------:|----------------------:|----------------:|
| Base             |          0.458333 |        38.7439 |              0.827244 |        0.132648 |
| Conditional_Base |          0.354167 |        38.1071 |              0.836328 |        0.128864 |

Projecting the Base task code while restoring factual content gives slightly higher mean identity similarity but lower Future Validity. Subsequent task-space robust refinement selects much larger changes. This diagnostic separates the conditional realization from its robust optimization.

## Detector-scale audit and authoritative identity files

The initial640px detector missed5/8 factual faces, making initial identity averages incomplete and unsuitable for matched comparison. A read-only factual-image audit found all8 at320px, at the unchanged0.5 confidence threshold. That scale was fixed before reevaluating all CF methods; identity scores were not used to choose it. All factual/CF images are now detected. Aligned crops were visually checked. The original640px measurements remain intact for traceability.

Use `/home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1/identity_detection_audit/identity_metrics.csv`, `full_vs_bottleneck.csv`, `geometry_ablation.csv`, `identity_per_example.csv`, `identity_paired.csv`, `protocol.json` and `review.json` for the corrected identity evaluation. They are also stored in `results/015_ffhq_identity_v1/identity_detection_audit/`. Original root identity summaries and the initial report are superseded for identity comparisons only. Optimization, future predictions and qualitative images are unchanged.

## Qualitative inspection

The complete fixed-order eight-row panel shows substantial changes in facial appearance, hairstyle, accessories and background in several bottleneck cases. The result does not support an identity-preservation claim. No case was removed or selected for appearance. Face-embedding similarity is an evaluation signal rather than an identity guarantee.

## Runtime

New construction wall time: 315.28s. Total new worker occupancy: 0.6554 GPU-worker hours, including reconstruction and evaluation. Table1 runtime is worker-seconds for the entire eight-CF population: original Base generation, full-latent test optimization/pooling, or bottleneck test optimization/pooling. Shared basis/bank/calibration costs are excluded from method rows and itemized in runtime.csv. Reused historical costs are reported, not rerun; bottleneck beta rows share pooled candidates, so runtime must not be summed across beta.

## Qualitative panels

All eight prespecified test IDs are shown, in fixed order, in ffhq_identity_qualitative.pdf and ffhq_identity_beta.pdf. The panels expose facial appearance, hair, pose, background and smile changes. No favorable-example selection was used. Visual identity-preservation claims require inspection and are not inferred from latent freezing.

## Limitations

The SVD split is approximate and classifier-dependent; task directions can still encode appearance and content directions can retain smile information. Generated FFHQ-style faces, prompt-relative labels, small populations, projection-only Update and finite attack bank limit generalization. Face alignment/detection and ArcFace domain behavior affect identity scores. The historical brightness/blur replicas include deterministic duplicates. Reusing previously evaluated future models yields a controlled follow-up, not independent replication. The central GRACE objective/theory and StyleMNIST/Galaxy runs are unchanged.

## Original-run independent verification (before detector audit)

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


## Final audit verification

{
  "status": "passed",
  "all_factuals_detected": true,
  "missing_pairs": 0,
  "mean_paired_identity_gain_vs_full": -0.07396605575922877,
  "paired_count": 8,
  "wall_seconds": 9.924326419830322,
  "original_run_review": "/home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1/completion_review.json"
}

Five focused unit tests passed. Both qualitative PDFs are one-page documents; image panels and aligned face crops were inspected. Original source hashes and all experiment outputs are preserved.
