# Extended calibration-only smile-editor comparison

Raw run: `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v2/20260917_extended_v1`. Previous screen: `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3`. Stronger scalar settings and independent-channel optimization were authorized after inspecting the first screen, so this is a sequential calibration study, not independent confirmation. The FFHQ final test set and all future models remain untouched.

## What changed

Reuse the same training-selected1/4/8 style channels. Add scalar strengths6,8,12,16; retain old0–4 candidates without regeneration. If scalar controls fail the6/8 gate, allow the selected4 or8 standardized channel coefficients to move independently. Start from each factual’s best preservation-feasible scalar edit, maximize original CLIP smile probability using40 normalized-gradient steps with adaptive step size, and bound each coefficient to±16. Accept a trial only when its original score is nondecreasing and every measured preservation gate passes. Rejected trials remain saved. This is constrained editor construction, not GRACE or a current-validity repair branch.

## Unchanged limits

Every accepted edit requires ArcFace≥0.8, FaceNet≥0.8, LPIPS≤0.15 and outside-region RMS≤0.04. Smile success still requires original CLIP probability≥0.5 and increase≥0.05. Six of eight calibration factuals must pass. The fixed mouth/cheek ellipse remains a proxy, not segmentation. Nonselected styles and the input transform stay factual. No identity penalty was added.

## Results

| editor       |   passing_factuals | numerical_gate_passed   |   mean_CLIP_smile |   mean_CLIP_gain |   mean_ArcFace |   minimum_ArcFace |   mean_FaceNet |   mean_LPIPS |   mean_outside_RMS |   supervised_positive_outputs |   supervised_smile_transitions |   CLIP_pass_and_supervised_positive |
|:-------------|-------------------:|:------------------------|------------------:|-----------------:|---------------:|------------------:|---------------:|-------------:|-------------------:|------------------------------:|-------------------------------:|------------------------------------:|
| independent4 |                  6 | True                    |          0.484528 |        0.245909  |       0.85847  |          0.800738 |       0.925453 |    0.0299547 |          0.0179876 |                             3 |                              2 |                                   2 |
| independent8 |                  6 | True                    |          0.49147  |        0.252852  |       0.838581 |          0.800454 |       0.929894 |    0.0391422 |          0.0260883 |                             2 |                              1 |                                   1 |
| style1       |                  4 | False                   |          0.480469 |        0.241851  |       0.852718 |          0.806425 |       0.931782 |    0.0232221 |          0.010756  |                             3 |                              2 |                                   1 |
| style4       |                  1 | False                   |          0.315406 |        0.0767878 |       0.924653 |          0.857322 |       0.960153 |    0.0232786 |          0.0156945 |                             2 |                              1 |                                   0 |
| style8       |                  3 | False                   |          0.393263 |        0.154645  |       0.875963 |          0.837533 |       0.946046 |    0.0348264 |          0.0244414 |                             2 |                              1 |                                   0 |

{
  "numerical_gate_passed": true,
  "passing_counts": {
    "independent4": 6,
    "independent8": 6,
    "style1": 4,
    "style4": 1,
    "style8": 3
  },
  "supervised_factual_positive_count": 1,
  "qualitative_review_required": true,
  "GRACE_started": false,
  "final_FFHQ_test_accessed": false,
  "future_models_accessed": false,
  "preservation_limits_relaxed": false
}

Selected examples use the minimum pixel cost among candidates passing all original gates; failures display the highest-score preservation-feasible candidate. This is explicitly post-selection on calibration data. The complete candidate table contains rejected large edits and preserves all failures.

## Independent supervised smile check

{
  "checkpoint": "/tmp/ffhq_lennon_state.pt",
  "source": "https://huggingface.co/AiresPucrs/LeNNon-Smile-Detector",
  "revision": "e34d94965363927f60dea5a5b76770a8966048c5",
  "model_card_mapping": "argmax(logits)>0 means smiling; ten-logit head retained",
  "validation": "first32 annotated eurecom-ds/celeba test rows; diagnostic sample, overlap with training unknown",
  "role": "diagnostic only; never changes original classifier or selection objective",
  "converted_state_sha256": "c024c004569e8d409d663a90ed051d9bf43e2000dd30acfa6755214cfd50f2eb",
  "original_checkpoint_sha256": "61da9befc0b3f13b22ae3b4a380a0b6c9d5ac8e7b73a2aeac7c594db21470030",
  "conversion": "weights_only load with audited module/standard nn allowlist, then state_dict; inference keys loaded strictly",
  "source_sample_accuracy": 0.96875,
  "source_positive_count": 14,
  "source_extra_class_predictions": 0,
  "small_nonindependent_source_audit_not_generalization_estimate": true,
  "validation_dataset_revision": "e9092e921951927418a9e8f87822020206b292ce"
}

The [released LeNNon model](https://huggingface.co/AiresPucrs/LeNNon-Smile-Detector) was trained on CelebA smile labels. Architecture and upstream preprocessing were audited; it resizes RGB to100×100 with values[0,1]. The checkpoint retains10 logits despite the binary task. We preserve upstream argmax>0 mapping and report1−P(class0), raw class and extra-class mass. Only classes0/1 were predicted on the32 annotated source-audit examples. The source check scored31/32 but training overlap is unknown; it is not a generalization claim or calibration of probabilities. This small supervised model is a diagnostic, not a replacement for the original classifier or a ground-truth annotator. Disagreements and visually ambiguous smiles must be reported.

|   calibration_index |   supervised_smile_probability |   supervised_smile_label |   supervised_raw_class |   extra_class_mass |
|--------------------:|-------------------------------:|-------------------------:|-----------------------:|-------------------:|
|                   0 |                    0.00425702  |                        0 |                      0 |        2.93184e-25 |
|                   1 |                    0           |                        0 |                      0 |        4.80531e-21 |
|                   2 |                    0           |                        0 |                      0 |        7.70791e-35 |
|                   3 |                    1.19209e-07 |                        0 |                      0 |        4.4167e-29  |
|                   4 |                    0           |                        0 |                      0 |        9.065e-41   |
|                   5 |                    0.999989    |                        1 |                      1 |        2.13077e-25 |
|                   6 |                    7.71284e-05 |                        0 |                      0 |        6.13176e-20 |
|                   7 |                    0.0126137   |                        0 |                      0 |        2.48111e-16 |

Source-audit examples are the first32 test rows from eurecom-ds/celeba, an external dataset—not the FFHQ final test cohort. Neither these examples nor the independent smile scores affect channel discovery, the optimization objective, or the original numerical gates.

## Interpretation rule

Only proceed to GRACE if an editor achieves the numerical gate and the qualitative/independent smile check supports meaningful edits. Do not relax identity limits or claim success solely from larger classifier scores. If original source labels already disagree with visible/independent smiles, report the cohort/label limitation rather than silently replacing labels or changing success thresholds.

## Reproducibility and checks

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
    6.0,
    8.0,
    12.0,
    16.0
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
  "scope": "User-approved sequential calibration follow-up. Frozen controls, stronger amplitudes6/8/12/16; if scalar gate fails test independently movable4/8 selected channels. Original preservation limits unchanged. No final FFHQ test or future models.",
  "previous_screen": "/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3",
  "independent": {
    "counts": [
      4,
      8
    ],
    "steps": 40,
    "initial_step_size": 1.0,
    "maximum_step_size": 2.0,
    "minimum_step_size": 0.001,
    "step_growth": 1.1,
    "rejection_factor": 0.5,
    "coordinate_bound": 16.0,
    "initialization": "highest original smile score among all preservation-feasible scalar candidates including the previous screen",
    "objective": "maximize original smile probability under unchanged measured preservation gates; no identity penalty",
    "acceptance": "all preservation gates and nondecreasing original smile score"
  },
  "supervised_smile": {
    "checkpoint": "/tmp/ffhq_lennon_state.pt",
    "source": "https://huggingface.co/AiresPucrs/LeNNon-Smile-Detector",
    "revision": "e34d94965363927f60dea5a5b76770a8966048c5",
    "model_card_mapping": "argmax(logits)>0 means smiling; ten-logit head retained",
    "validation": "first32 annotated eurecom-ds/celeba test rows; diagnostic sample, overlap with training unknown",
    "role": "diagnostic only; never changes original classifier or selection objective"
  }
}

{
  "status": "passed",
  "jobs": 27,
  "eight_gpus_used": true,
  "previous_scalar_candidates_reused": 144,
  "new_scalar_candidates": 96,
  "independent_candidates": 656,
  "all_accepted_vector_iterates_preserve_gates": true,
  "maximum_identity_cosine_error": 1.1102230246251565e-16,
  "supervised_prediction_recompute_error": 1.1102230246251565e-16,
  "missing_ArcFace_pairs": 0,
  "missing_FaceNet_pairs": 0,
  "source_hashes_unchanged": true,
  "final_FFHQ_test_accessed": false,
  "future_models_accessed": false
}

Every independent-channel accepted iterate is checked against the true measured preservation gates; missing detections fail. All identity similarities are recomputed from saved embeddings, and independent smile predictions are re-evaluated with a fresh model instance. Raw coordinates, rejected candidates, PNGs, commands, seed, model hashes and phase timing are retained. Previous outputs and the manuscript are unchanged.


## Final reviewed outcome

Independent movement of4 or8 selected channels reaches the original numerical gate on6/8 calibration faces without relaxing any preservation limit. The4-channel candidate is preferable within this calibration comparison: mean ArcFace0.8585 versus0.8386, FaceNet0.9255 versus0.9299, LPIPS0.0300 versus0.0391. These means include all8 selected calibration outputs; they should not be compared directly to prior results on different test factuals.

This is real progress in constrained editor feasibility, but not established semantic smile success or robust recourse. Of six original-gate successes, the supervised model marks only two of the4-channel and one of the8-channel selected outputs smiling. Qualitative inspection shows useful mouth changes in several cases, but one numerical success has little visible smile change and another factual already appears smile-like. The supervised classifier also disagrees with some visibly smile-like images; its small source-domain audit does not validate it on FFHQ. It is not adopted as a replacement classifier.

The4-channel control is saved as a promising calibration-selected candidate, not a validated final editor. GRACE is not launched because the prior prerequisite—convincing smiles under preservation limits—has not been established consistently. The next unresolved work is a dependable smile-label/evaluation protocol on these generated faces; no labels, thresholds or existing outputs were silently replaced.

Per-case observations in qualitative_review.csv are descriptive assistant review, not blinded human annotations, adjudicated ground truth or face-identity recognition. They must not be used as a numerical accuracy benchmark.

{
  "status": "complete",
  "numerical_gate_passed": true,
  "candidate_editor": "independent4",
  "candidate_reason": "Same6/8 original-gate success as independent8, with higher mean ArcFace and lower LPIPS; calibration-selected candidate only",
  "candidate_mean_ArcFace": 0.8584702387452126,
  "candidate_mean_FaceNet": 0.9254525974392892,
  "candidate_mean_LPIPS": 0.0299547255272045,
  "semantic_smile_validation_established": false,
  "GRACE_started": false,
  "reason_for_no_GRACE": "Original score, supervised diagnostic and qualitative appearances disagree; consistent non-smiling-to-smiling interpretation is not established on the required cohort",
  "preservation_limits_relaxed": false,
  "final_FFHQ_test_accessed": false,
  "future_models_accessed": false,
  "focused_tests_passed": 9,
  "qualitative_panels_inspected": true,
  "completed_utc": "2026-09-17T11:37:07.106276+00:00"
}
