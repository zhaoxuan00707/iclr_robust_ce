# FFHQ-generated face GRACE

See [result interpretation](interpretation.md) for the identical midpoint outputs and substantial appearance-change limitation.

Completed with original pretrained StyleGAN3, CLIP and conditional flow. This is a synthetic-face, prompt-relative experiment; it is not a supervised real-FFHQ smile benchmark.

## Task and asset reuse

The original classifier compares literal prompts "smiling face" (index0, target) and "face" (index1, source), with probability threshold0.5. Original StyleGAN3-r-ffhq-1024x1024.pkl synthesizes RGB1024 faces from W512 repeated over16 layers; there is no encoder or inversion. The original KarrasUnet1D flow checkpoint and progressive lift/land implementation generate Base CFs with10 iterations at0.8/0.8. LeapFactual comparator applies5 additional iterations at0.8/0.83. GRACE starts from Base, never the robust comparator. Exact paths and hashes are in model_manifest.json and ffhq_asset_inventory.md.

## New fixed update and generated cohorts

256 generated training faces receive frozen original CLIP pseudo-labels. Separate64-image generated pools supply8 non-smiling calibration factuals and8 test factuals, selected in seeded order. Eight smiling training references are fixed before evaluation. All original generator/flow/CLIP backbone/text weights are frozen. Update trains the entire768×512 visual projection for5 epochs with fresh AdamW(lr1e-4,weight_decay0.01), batch32, fixed seed700000. This deliberately limited update family is newly introduced because the original zero-shot classifier has no supervised retraining protocol. It is not full-backbone adaptation. Failed Base CFs remain in the denominator.

## Selected configuration

r_star=1.0, beta_geo_star=1.0. Selection uses only the8 calibration grid rows and four predeclared calibration models; the operating point is fixed for all future severities.

## Main results at nominal external severity

| method         |   Future_Validity |   pixel_L2_224 |   pixel_L2_1024 |   LPIPS_squeeze |   Phi_hat_r |
|:---------------|------------------:|---------------:|----------------:|----------------:|------------:|
| Base           |          0.458333 |        38.7439 |         188.093 |        0.132648 |   0.113816  |
| GRACE_beta1.00 |          1        |        92.9133 |         428.993 |        0.382032 |   0         |
| GRACE_beta1.05 |          1        |        92.9133 |         428.993 |        0.382032 |   0         |
| GRACE_beta1.10 |          1        |        92.9133 |         428.993 |        0.382032 |   0         |
| GRACE_beta1.20 |          1        |        92.9133 |         428.993 |        0.382032 |   0         |
| GRACE_selected |          1        |        92.9133 |         428.993 |        0.382032 |   0         |
| LeapFactual    |          0.833333 |        54.6383 |         257.456 |        0.202362 |   0.0327701 |

Future Validity averages over3 shift families and2 realizations at severity1, with8 factuals. Pixel cost is distinct from distribution-level geometry. Identity score is unavailable: no compatible original face identity metric was found, and none was invented. SqueezeNet LPIPS, SSIM and PSNR follow the original image-quality metric families. No identity-preservation claim is made.

## Geometry budget

|   beta_geo |   Future_Validity |   pixel_L2_224 |   Phi_hat_r |   LPIPS_squeeze |
|-----------:|------------------:|---------------:|------------:|----------------:|
|       1    |                 1 |        92.9133 |           0 |        0.382032 |
|       1.05 |                 1 |        92.9133 |           0 |        0.382032 |
|       1.1  |                 1 |        92.9133 |           0 |        0.382032 |
|       1.2  |                 1 |        92.9133 |           0 |        0.382032 |

All distances are exact equal-mass Euclidean W2 in unnormalized W512. Feasibility tolerance1e-6×max(1,D). The objective is max_model mean_CF hinge. Each candidate is cross-evaluated against all compatible shared-bank models; beta candidates are pooled after optimization. Nonincreasing retained bank risk=True. This is finite-bank/local optimization, not exact WRO optimality. No current-validity constraint or repair is used.

## Future shifts

New brightness, additive noise and Gaussian blur stresses alter canonical224 training images before the same fixed projection Update. CFs are frozen before all30 future models. External severity differs from internal Wasserstein radius. Brightness/blur replicas and severity0 contain deterministic duplicates; they do not provide independent replication or uncertainty estimates.

## Image diagnostics and qualitative figure

The first four prespecified test factuals appear in every column. Native1024 returned images and source W codes are retained. The generator is deterministic; no encoder reconstruction error is applicable. Latent displacement, pixel costs, original LPIPS, SSIM and PSNR are saved. Visual inspection is descriptive and cannot establish identity preservation.

## Runtime

{
  "construction_wall_seconds": 538.5463206768036,
  "seconds_per_test_CF": 67.31829008460045,
  "construction_GPU_hours": 0.583607006404135,
  "future_GPU_hours": 0.10786510917875503,
  "servers": 1,
  "gpu_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ]
}

End-to-end construction includes generated-data preparation, Base and compatible-baseline generation, shared bank, calibration, both-start/beta refinement and output freezing. It excludes original pretraining and future evaluation. Shared construction time is repeated in method rows for context and must not be summed. GPU-hours measure summed worker occupancy.

## Checks, failures and limitations

{
  "status": "passed",
  "jobs": 75,
  "attack_models": 16,
  "future_models": 30,
  "max_decoded_image_error": 0.0,
  "max_future_validity_error": 0.0,
  "max_bank_risk_error": 2.3096799842137195e-07,
  "all_geometry_feasible": true,
  "nonincreasing_bank_risk": true,
  "configuration_selection_recomputed": true,
  "all_future_after_freeze": true,
  "gpu_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ]
}

Initial audit attempts using reference StyleGAN operators exceeded the memory cap. Original fused CUDA operators were built in an isolated environment; checkpointed synthesis was then verified exactly equal to original inference and latent gradients were finite. Audit failures produced no experiment results. The data adversary edits32 of256 canonical training images per restart and uses12 one-step surrogate iterations; each accepted model receives the complete fixed5-epoch projection update. These restrictions and8 test factuals limit generalization. The classifier and labels are both prompt-relative CLIP, and "face" is not an explicit non-smiling annotation. The fixed audit image visibly smiles while its smiling probability is0.45375; this observed semantic mismatch limits any literal non-smiling-to-smiling interpretation. All historical outputs were preserved; no manuscript claims were changed.

Raw run: `/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1`.
