# Galaxy update-aware directed-pair pilot

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/20260924_v1`. Task: `tasks/017_galaxy_update_aware.md`. Config: `experiments/configs/galaxy_update_aware.json`. Controller069; independent report070.

## Results

| Direction       |    r |   beta |   Base FV (%) |   r=0 FV (%) |   Selected FV (%) |   Selected cost |
|:----------------|-----:|-------:|--------------:|-------------:|------------------:|----------------:|
| Merging → Round | 0    |   1.02 |         69.92 |          100 |               100 |          12.786 |
| Round → Cigar   | 0.25 |   1.02 |         50    |          100 |               100 |          14.193 |

Future Validity averages the same 16 held-out shifted-training head models for each method. Nominal-only uses the selected geometry budget; its gain is separated from the additional radius term. Selection uses calibration only. All 16 cases per direction are retained.

## Implementation

Original ten-class VGG16-BN and WAE64; three full-batch SGD updates (lr 1e-4) of the last linear layer only, on a fixed 40-image class-balanced training subset. Earlier weights and BN statistics are frozen. All three updates are differentiated. The smooth probability hinge has threshold0.5 and temperature0.05. Pixel sensitivity is the exact chain-rule pullback through the frozen VGG features, with sqrt(40) empirical-RMS dual scaling; it is not feature-space uncertainty. Q has16 equal-mass latent supports; geometry uses exact latent W2 and deployment is a one-to-one assignment.

Eight (r,beta) settings per split/direction, 160 updates each, best feasible iterate retained including Base. No validity repair. A retained feasible iterate does not establish optimizer convergence or a robustness certificate. Fresh brightness/noise future models use the identical prescribed head update; these stress tests are not certified inner maximizers and are not restricted to every r-ball.

## Runtime and solver checks

Construction wall time: 23.6 minutes; optimization worker-hours: 1.94; peak allocated optimizer memory: 5.21 GiB. Saved-output feasibility: True; final-iterate feasibility: True. This excludes historical Base/classifier training and is not a kernel-only GPU-time measurement.

## Shift scale

|                        |     min |     max |
|:-----------------------|--------:|--------:|
| ('cal', 'brightness')  | 6.18703 | 21.5122 |
| ('cal', 'noise')       | 4.42722 | 17.0622 |
| ('test', 'brightness') | 4.43694 | 21.0927 |
| ('test', 'noise')      | 4.43145 | 17.0726 |

These are paired transport upper bounds, not exact class-preserving OT distances. The stress shifts can lie outside the optimized radius; interpret FV as shared held-out stress performance.

## Checks and provenance

32 saved solves checked independently: geometry, assignment, cost, fresh decoded-image/model FV, per-example FV, best feasible retention, surrogate decomposition and calibration selection. Sixteen focused tests passed (three new pixel-pullback tests, seven OT tests, six update-aware tests). The new derivative tests compare the memory-bounded calculation against direct autodiff. Galaxy directional derivative audits are saved under each raw pair directory. All original inputs are hashed and verified after evaluation. GPU4 handles future-model work; GPUs0,1,2,3,5,6,7 handle the heavier optimization jobs.

Assets and cohorts are historical; see input_manifest.json and asset_audit.csv. Original classifier checkpoint and AE paths are recorded there. Current deterministic decoder replay differs slightly from original images: Historical_Base is reported separately; Base=Dec(original raw latent), with no re-encoding or new generation. Audit_v1/v2 stopped on strict replay tolerances and remain preserved. Original TF32/batch4 replay is exactly equal for all four cohorts; the float32 discrepancy is reported rather than hidden.

## Reference realization

| pair          |   decoded_reference_target_validity |   reference_reconstruction_pixel_L2 |
|:--------------|------------------------------------:|------------------------------------:|
| merging_round |                                   1 |                             9.88987 |
| round_cigar   |                                   1 |                            10.5518  |

The favorable anchors are encoded observed target images; decoder reconstruction need not preserve their original classifier validity. These diagnostics do not filter anchors or alter the experiment.

## Limits

This is a small retrospective head-adaptation pilot, not all-parameter retraining or an independent end-to-end benchmark. Original Base and reference/factual predictions can fail the requested target/source conditions; see asset_audit.csv and the separate initially-valid Base subset metrics. Target probability≥0.5 is classifier validity, not human morphological validation. No claim of identity preservation or a globally optimal robust solution follows from these images. The old VGG checkpoint used the original test folder for validation, so these historical cohorts are not a pristine unseen benchmark. One Merging→Round test factual is already target-positive; it remains included by the prespecified no-filtering rule.

## Figures

All16 fixed test cases per direction are in figures/galaxy_update_aware/*_all16.pdf. Each page shows factual, Base, matched-beta nominal-only, selected and r4, plus two common-scale heatmaps. If calibration selects r=0, the selected and nominal-only columns are necessarily identical. Sensitivity/FV/cost plots use all predeclared test grid cells as diagnostics; they do not change calibration selection.

## Interpretation

Both selected test populations and their matched-beta nominal-only controls reach
100% Future Validity. Thus this run provides no additional held-out FV gain from
the sensitivity term. Round→Cigar's r=.25 improves calibration FV (93.75%→100%)
and lowers test sensitivity (0.001448→0.000141), but the test validity metric is
already saturated at r=0. At beta1.02, r=4 instead gives93.75% test FV for
Merging→Round and68.75% for Round→Cigar; reducing sensitivity alone can coincide
with confidently invalid outputs. These observations do not justify selecting
a larger radius using test results.

Numerical replay affects boundary cases: the unchanged Historical_Base has
74.61% Merging→Round test FV, compared with69.92% for the exact float32 decoder
replay used by optimization. Round→Cigar is50% for both versions. The difference
must remain visible when quoting Base-to-refinement gains.

Visual inspection of the first all-case pages shows that Base-to-refinement
edits are generally subtle. Some inherited Base images have strong reconstruction
artifacts (for example Round→Cigar ID3); those artifacts persist in refined
outputs despite high classifier validity. These are not human-validated galaxy
morphology results. The complete PDFs retain every case.
