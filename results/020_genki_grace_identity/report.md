# GENKI-head GRACE with identity and background preservation

This run uses GRACE, with a new GENKI-supervised smile head and rebuilt data-side
attack/future-update models. It starts from newly generated nominal Base CFs.
It does not rescore the earlier CLIP edits and call them GRACE.

Raw run: `/home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1`. All 79 jobs completed on eight GPUs. Selected configuration:
**r=2, beta=1.2**, chosen on calibration shifts only.

## Main result

| method         |   current_validity |   ArcFace_mean |   ArcFace_min |   FaceNet_mean |   LPIPS_mean |   outside_RMS_mean |   outside_RMS_max |   pixel_L2_224 |   future_validity |   worst_future_model_validity |   Phi_hat_r | calibration_selected   |
|:---------------|-------------------:|---------------:|--------------:|---------------:|-------------:|-------------------:|------------------:|---------------:|------------------:|------------------------------:|------------:|:-----------------------|
| Base           |               0.25 |       0.922661 |      0.800097 |       0.937857 |    0.0144713 |         0.00567567 |         0.0131672 |        6.324   |          0.316667 |                             0 |    0.148015 | False                  |
| GRACE_beta1.00 |               0.25 |       0.922661 |      0.800097 |       0.937857 |    0.0144713 |         0.00567567 |         0.0131672 |        6.324   |          0.316667 |                             0 |    0.148015 | False                  |
| GRACE_beta1.05 |               0.5  |       0.919918 |      0.800367 |       0.941926 |    0.0147557 |         0.00583399 |         0.0128942 |        6.51206 |          0.45     |                             0 |    0.144902 | False                  |
| GRACE_beta1.10 |               0.5  |       0.919918 |      0.800367 |       0.941926 |    0.0147557 |         0.00583399 |         0.0128942 |        6.51206 |          0.45     |                             0 |    0.144902 | False                  |
| GRACE_beta1.20 |               0.5  |       0.919918 |      0.800367 |       0.941926 |    0.0147557 |         0.00583399 |         0.0128942 |        6.51206 |          0.45     |                             0 |    0.144902 | True                   |

The selected GRACE result has future validity **45.00%**,
versus **31.67%** for Base on the same 30 future models.
Mean ArcFace similarity is **0.9199**, mean LPIPS
**0.0148**, and maximum outside-region RMS
**0.0129**. All 40 Base/beta outputs pass the declared
preservation limits. Failed nominal Base cases remain in every population and
in all figures; they are not excluded from validity averages.

Primary validity and bank risk above use a uniform CPU float32 evaluation of the
actual saved PNG faces, performed as a post-hoc consistency audit. The original
float-feature evaluation is preserved separately in cached_future_validity.csv
and cached_finite_bank_risk.csv. Base and beta1 have byte-identical PNGs, yet their
original cached scores differed slightly because of numerical realization paths.
Base stops near the threshold, so small realization/quantization changes can
change labels. The uniform saved-image audit fixes this comparison inconsistency
without changing a face, model, radius, beta or configuration selection. Every
PNG existed before future-model training. Identical Base/beta1 images now have
identical probabilities under every evaluated model. Geometry/optimization traces
remain results for the original floating-point search; identity/LPIPS/outside-RMS
metrics remain the saved native-render diagnostics (identity detection itself
used the same quantized RGB as the PNGs). This additional evaluation was not
predeclared before the numerical discrepancy was discovered.

These are classifier-defined validity results on generated FFHQ-style faces.
Human smile correctness and real-world identity verification are not established.
Neither GENKI test accuracy nor preservation scores alone establish meaningful
non-smiling-to-smiling changes. Inspect the figures together with the numbers.

## What was kept fixed

The selected GENKI linear head uses the original frozen CLIP image encoder,
without text prompts. The pretrained StyleGAN3 generator remains fixed. Only the
four previously selected post-affine style channels can change. All other styles
and the factual Fourier input transform are held fixed. Absolute selected styles
divided by their frozen training standard deviations define the four-dimensional
coordinates u. Factual content is paired with each returned support throughout.

Every returned candidate must have ArcFace>=0.8, independent FaceNet>=0.8,
native-image LPIPS<=0.15, and RGB224 RMS outside the fixed mouth/cheek ellipse<=0.04.
Missing detections fail. Coordinate displacement is bounded by 16 standard
deviations per channel. These are hard final-candidate constraints, not soft
identity penalties. The mask is a localization proxy, not segmentation; an RMS
bound is not a guarantee that every background pixel is unchanged. Similarity
0.8 is an engineering limit, not a calibrated same-person verification threshold.

## GRACE objective and geometry

The finite-bank objective is max_model mean_CF [0.5-p(smile)]_+. It is not
mean_CF max_model loss. All populations at a radius use the full compatible
shared bank. The source is the same frozen nominal Base population for every beta.
The reference is the first eight generated training faces predicted smiling by
the new head, encoded into u. D=W2(Base,reference); enforce
W2(Base,Q)+W2(Q,reference)<=beta*D using exact Euclidean W2 in u for all terms.

The reference label belongs to the original training face. Transplanting its four
coordinates into another factual's frozen content need not preserve that label.
This is the task-space/frozen-content realization from the identity experiment,
not full-W geometry or a claim that the channels perfectly disentangle smiles.

The existing solve_raw optimizer uses its smooth bank-loss approximation and
geometry-only primal/dual updates. Twelve steps from Base and a near-Base start,
plus eight declared geodesic points, define the numerical search. All evaluated
iterates enter the finite candidate pool, including those not retained by the
solver's unconstrained risk incumbent. Pools are shared across betas at each
radius. Final selection uses exact bank hinge, hard geometry/preservation and
pixel-cost tie-breaking. Preservation is screened after proposal generation;
this is a finite-search solution, not a certificate of a global constrained optimum.
No nominal-validity repair, nominal-validity penalty or extra target penalty enters GRACE.

|   r |   beta_geo |       D |   B_geo |   total_length |        slack |   Phi_hat_r |   pixel_L2_cost |   candidate | preservation_feasible   | geometry_feasible   |
|----:|-----------:|--------:|--------:|---------------:|-------------:|------------:|----------------:|------------:|:------------------------|:--------------------|
|   2 |       1    | 10.5929 | 10.5929 |        10.5929 | -1.77636e-15 |    0.147391 |         6.32398 |           0 | True                    | True                |
|   2 |       1.05 | 10.5929 | 11.1225 |        10.9103 |  0.212263    |    0.144694 |         6.51201 |          75 | True                    | True                |
|   2 |       1.1  | 10.5929 | 11.6522 |        10.9103 |  0.741908    |    0.144694 |         6.51201 |          75 | True                    | True                |
|   2 |       1.2  | 10.5929 | 12.7115 |        10.9103 |  1.8012      |    0.144694 |         6.51201 |          75 | True                    | True                |

The geometry table's Phi_hat_r is the optimization's cached-feature bank risk;
the main table's Phi_hat_r is the uniformly evaluated saved-image bank risk.

## Compatible Base and model construction

Factuals come from the previous generated calibration/test pools in their seeded
order, requiring nominal non-smile and both identity detections before CF
generation. Eligibility exclusions are saved. The images are generated faces,
not inversions of GENKI photographs. The pools have appeared in earlier work,
so this is a matched new-classifier experiment, not a new untouched face benchmark.

Base uses nominal smile-gradient ascent only, with up to 40 steps and actual
preservation checks. On the first observed crossing of 0.5, twelve local bisection
steps refine that crossing; this is not a global minimum-distance certificate.
If no crossing is found, keep the best preservation-feasible nominal candidate.
All Base outputs freeze before bank construction.

Only the 2798 GENKI training images enter Update. Start from the selected
GENKI head each time; use 120 deterministic full-batch SGD steps, lr0.1, L2=0.0003,
frozen image encoder, and unchanged human training labels. Each of eight attacks
changes pixels of 32 training images, using 12 ascent steps with a one-step
hypergradient surrogate. Fractions0.5/1 produce sixteen exact full-update heads;
a clean full update supplies the seventeenth. Accepted weights come from Update,
not direct parameter perturbations. The short attack surrogate is not the full
120-step unroll and does not certify worst-case uncertainty coverage.

Data-side radius is RGB224 pixel Euclidean W2 with preserved labels. Every
saved perturbation has a paired-coupling upper bound within its radius. This
certifies membership conservatively; exact 2,798-image OT is not computed. The
selected radius has 17 compatible bank entries. The old CLIP-labeled flow,
old attack bank and old future models are not used for this classifier.

The distance and attacks operate on preprocessed floating-point canonical RGB
tensors. Antialiased bicubic resizing produces values from
-0.111786 to 1.117817, including slight overshoot
outside [0,1]. Proposal clipping occurs before budget rescaling; final shifted
values stay within the per-pixel convex hull of the original tensor and [0,1].
The independent audit verifies this envelope and the paired transport bound.
This is not a claim that every attacked tensor is a quantized, range-clipped RGB
photograph. An initial report check that assumed all canonical values were in
[0,1] failed; correcting that audit assumption did not change any training data,
attack, model or counterfactual.

## Calibration and future evaluation

Four declared shifted-training calibration updates select r/beta by minimum
maximum population mean hinge (rounded1e-6), then pixel cost, radius and beta.
Selection is frozen before test refinement. All test CFs freeze before future
models are trained. Future stress covers brightness, noise and blur at five
ordered severities and two realizations. External severity is not internal r.
The 30 future updates contain **18 distinct weight states**;
deterministic shifts and zero severity repeat models. FV is an empirical average
over this correlated stress suite, not 30 independent trials or a deployment guarantee.

## Runtime and checks

Construction wall time: 7.09 minutes, including Base,
banks, calibration and refinement; excluding pretrained classifier/generator
training and final future evaluation. Allocated construction GPU time:
0.324 GPU-hours. Full phase timings are saved.

Seven focused tests pass. Independent checks recompute exact geometry, shared-bank
population risk, all future predictions, validation selection, pixel-shift
certificates and identity cosines from saved embeddings. All parent hashes remain
unchanged. No previous output or manuscript claim was changed.

## Reproduce and inspect

```sh
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/041_run_genki_grace.py --output /home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/audits/genki_grace_saved_images.py --run /home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1 --output /tmp/genki_grace_saved_image_audit_20260917
PYTHONPATH=/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/042_report_genki_grace.py --output /home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1
```

Workers run through experiments/run.py with source fingerprints and completion
markers. Exact commands are in raw logs; the initial command is in
/tmp/genki_grace_initialize.log. Config, seed, hardware, source/checkpoint hashes,
factual IDs, coordinates, proposals, bank perturbations, updated heads, quality
screens and frozen outputs are persisted. Raw arrays/checkpoints remain outside Git.

Figures: qualitative.pdf shows all eight Factual/Base/selected-GRACE pairs;
qualitative_all_betas_v2.pdf shows every beta for every fixed case. Smile scores
are nominal classifier predictions; per-face FV averages the common future models.


## Descriptive visual review

Both pages of the selected comparison were inspected, with all eight fixed cases retained. Backgrounds, clothing and hair generally appear stable; the additional GRACE changes are often subtle. Cases2 and7 have visible Base mouth/teeth changes but remain failures under all future models. Cases3 and5 already appear smile-like in the factual images, so the new classifier has not eliminated the semantic source-label concern. These are assistant descriptions, not blinded human labels or face-identity verification. Stronger blur shifts produce zero validity for every method in this stress suite. See qualitative_review.csv and the shift-severity figure.


The final all-beta figure is v2, with corrected header/footer spacing. The initial draft remains preserved. This layout change does not regenerate images or alter results.
