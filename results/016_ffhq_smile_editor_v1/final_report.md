# Calibration-only localized smile-editor screen

Raw run: `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3`. Read-only parent: `/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1`. All eight GPUs used. Final test images/results and future models were not used.

## Question and approach

The previous29-dimensional smile-gradient subspace did not preserve identity. This screen tests whether a much smaller StyleGAN3 post-affine control can increase the original smile score while retaining facial appearance. This is a new adaptation inspired by [StyleSpace](https://arxiv.org/abs/2011.12799), not reuse of pretrained StyleGAN2 channel IDs.

## Training-only control discovery

Use all256 saved training W codes for per-channel standard deviations and a512-input logistic smile-direction baseline. On the first32 training codes, differentiate the original smiling softmax score with respect to standardized style channels. Estimate outside-region image-Jacobian RMS using three fixed-seed Rademacher projections per example. Rank eligible channels by absolute mean smile gradient divided by estimated outside-Jacobian RMS plus1e-5, require75% gradient-sign agreement, and select1/4/8 channels. Layer indices3–11 are eligible; other modulation channels and the factual Fourier input transform remain fixed. A scalar α moves the selected channels along the normalized mean-gradient direction. The W baseline is a single unit-norm logistic-regression direction and is not eligible as the hard-style-bottleneck realization.

## Predeclared calibration gates

{
  "parent_run": "/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1",
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
  "seed": 20260917,
  "discovery_training_count": 32,
  "style_counts": [
    1,
    4,
    8
  ],
  "eligible_layers": [
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11
  ],
  "outside_jacobian_probes": 3,
  "minimum_gradient_sign_agreement": 0.75,
  "alpha_grid": [
    0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0
  ],
  "calibration_count": 8,
  "arcface_min": 0.8,
  "facenet_min": 0.8,
  "lpips_max": 0.15,
  "outside_rms_max": 0.04,
  "smile_score_min": 0.5,
  "smile_score_gain_min": 0.05,
  "required_passing_factuals": 6,
  "selection": "Per factual, smallest alpha satisfying all gates; editor with most passing factuals, ties highest minimum ArcFace similarity then fewer mutable channels. Missing detections fail. A qualitative review is also required before GRACE.",
  "mask": "Fixed lower-face ellipse x=0.5,y=0.70,rx=0.27,ry=0.20 on canonical224. Coarse localization proxy, not semantic segmentation.",
  "identity": "ArcFace buffalo_l with320 detector; independent FaceNet VGGFace2 with MTCNN160 crop. Similarity0.8 is a conservative engineering gate, not a calibrated same-person verification threshold.",
  "scope": "Training/calibration only; no final test images, outputs, future models or scores used. Only a passing localized editor proceeds to GRACE. Thresholds frozen before measurement; no adaptation after failure."
}

These are engineering screen thresholds, not validated same-person verification operating points. The fixed lower-face ellipse is a coarse localization proxy, not a semantic mouth mask. A smile score is the original CLIP probability for “smiling face” versus “face”; it cannot establish a real smile on its own. A visual check must accompany the numerical gate. Missing detections fail.

## Results

| editor   |   mutable_channels |   passing_factuals |   total_factuals | numerical_gate_passed   |   mean_smile_gain |   mean_smile_score |   mean_arcface |   min_arcface |   mean_facenet |   min_facenet |   mean_LPIPS |   mean_outside_rms |
|:---------|-------------------:|-------------------:|-----------------:|:------------------------|------------------:|-------------------:|---------------:|--------------:|---------------:|--------------:|-------------:|-------------------:|
| style1   |                  1 |                  0 |                8 | False                   |         0.0558706 |           0.294489 |       0.957901 |      0.909655 |       0.980285 |      0.95126  |   0.00750699 |         0.00335109 |
| style4   |                  4 |                  0 |                8 | False                   |         0.0305878 |           0.269206 |       0.974687 |      0.946992 |       0.986917 |      0.976311 |   0.00703766 |         0.00429543 |
| style8   |                  8 |                  0 |                8 | False                   |         0.0459157 |           0.284534 |       0.964793 |      0.952681 |       0.984655 |      0.975582 |   0.0090766  |         0.00578691 |
| w_linear |                512 |                  1 |                8 | False                   |         0.0669333 |           0.305552 |       0.968949 |      0.940611 |       0.987801 |      0.96916  |   0.0256095  |         0.0267329  |

{
  "numerical_gate_passed": false,
  "selected_editor": null,
  "required_passing_factuals": 6,
  "qualitative_review_required": true,
  "final_test_accessed": false,
  "future_models_accessed": false,
  "GRACE_started": false,
  "thresholds_relaxed": false
}

Per factual, use the smallest edit strength passing every gate. For failed cases the figure shows the strongest smile score among quality-preserving candidates, or the zero edit if no quality-preserving candidate exists. FAIL labels remain visible. Thus the panel is a disclosed calibration selection, not a test-performance figure. All192 candidates and every rejection remain in all_calibration_metrics.csv.

## Independent identity evaluation

ArcFace buffalo_l uses the previously audited320px detector and five-landmark alignment. Independent FaceNet uses [facenet-pytorch2.6.0](https://github.com/timesler/facenet-pytorch), VGGFace2 InceptionResnetV1, MTCNN160 crops and512-dimensional cosine similarity. FaceNet is neither differentiated nor used in channel discovery. Its calibration threshold participates in editor screening, so it is not an untouched final test metric. Checkpoint hashes and acquisition source are retained in facenet_manifest.json.

## Next-stage rule

Run GRACE only if a localized control passes the frozen numerical gates and qualitative review. Otherwise stop after reporting this screen. No threshold relaxation or repeated test-guided search is allowed. Any later hard identity constraint changes the domain-specific feasible set; it does not modify max_model mean_CF hinge or data-side WRO.

## Checks

{
  "status": "passed",
  "jobs": 18,
  "calibration_images": 8,
  "evaluated_edits": 192,
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
  "training_only_controls": true,
  "numerical_gate_recomputed": true,
  "identity_cosines_recomputed": true,
  "zero_edit_identity_check_passed": true,
  "missing_arcface_pairs": 0,
  "missing_facenet_pairs": 0,
  "final_test_accessed": false,
  "future_models_accessed": false
}

{
  "zero_edit_max_error": 0.0,
  "autograd_directional_derivative": 0.026443906128406525,
  "finite_difference_derivative": 0.02551693469285965,
  "passed": true
}

The style wrapper is checked against original synthesis at zero edit; its gradients are checked by finite differences and a checkpointed toy-network test. Immutable channels are asserted exactly equal on every style edit. Calibration thresholds, controls and input hashes were frozen before evaluation. No existing experiment output was overwritten.


## Final reviewed outcome

The screen is complete. Localized edits retained high measured identity similarity (ArcFace means0.958–0.975; FaceNet0.980–0.987) with very small perceptual changes. However, each localized editor passed0/8 cases; the W baseline passed1/8. Six passing cases were required. No editor is accepted for the GRACE follow-up, and no final test or future models were accessed. These results show a useful small-edit preservation regime, not a successful counterfactual method.

Qualitative inspection of all8 calibration cases shows mostly preserved appearance and some small mouth changes, but not a reliable smile transition. Some factuals already visually smile despite original CLIP probabilities below0.5; therefore the smile pseudo-label model also needs independent validation before treating these scores as semantic smile success. This observation did not change thresholds, outputs or the gate. The next scientific question is whether a validated smile signal and a stronger localized control can jointly satisfy the fixed preservation constraints; that is not established here.

{
  "status": "complete",
  "screen_passed": false,
  "GRACE_started": false,
  "reason": "No localized editor satisfied smile and preservation gates on any of the8 calibration factuals;6 were required",
  "visual_review": "Most facial appearance, hair, accessories and background stay similar. Some localized mouth changes are visible, but no reliable non-smiling-to-smiling behavior across the cohort. Some factuals already visually smile despite low original CLIP scores, so the original pseudo-label classifier remains a semantic limitation.",
  "arcface_mean_range": [
    0.9579007402062416,
    0.9746873006224632
  ],
  "style_passing_counts": [
    0,
    0,
    0
  ],
  "w_linear_passing_count": 1,
  "final_test_accessed": false,
  "future_models_accessed": false,
  "thresholds_relaxed": false,
  "focused_tests_passed": 7,
  "pdf_inspected": true,
  "completed_utc": "2026-09-17T10:44:35.903368+00:00"
}
