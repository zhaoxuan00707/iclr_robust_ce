# FFHQ update-aware original-CLIP experiment

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/ffhq_update_aware/20260924_v1`. Task: `tasks/018_ffhq_update_aware.md`.
Controller072, report073, config `experiments/configs/ffhq_update_aware.json`.

## Results

| Method          |   Test FV (%) |   Current validity (%) |   Pixel cost |
|:----------------|--------------:|-----------------------:|-------------:|
| Historical_Base |         82.29 |                   87.5 |      38.7439 |
| Base            |         82.29 |                   87.5 |      38.7441 |
| r0_beta1.02     |        100    |                  100   |      47.5033 |
| r0.25_beta1.02  |        100    |                  100   |      48.0834 |

Calibration selected r=0, beta=1.02; test settings were frozen.
The additional selected-versus-nominal test FV gain is 0.00 percentage points.
Do not attribute the full Base-to-refinement change to sensitivity. Every row
uses the same eight test faces and24 held-out future projection checkpoints.

## Method and reuse

Original StyleGAN3-r FFHQ1024 generator, W512 repeated at16 layers, original CLIP
ViT-B/32 with literal prompts `smiling face` and `face`. Target is index0.
The original nominal Base codes are reused, not LeapFactual robust codes.
No GENKI classifier, fixed-content editor, identity/background constraint, or
validity repair. Exact W512 W2 geometry and one-to-one deployment assignment.
Mean smooth hinge (.5 probability threshold, .05 temperature) plus r times the
empirical-RMS pixel sensitivity norm. Training uncertainty is on RGB224 pixels,
not cached features. The complete prescribed update is three full-batch SGD
steps at lr1e-4 on CLIP's768x512 visual projection, with32 balanced generated
training images and original frozen pseudo-labels. Text, logit scale and visual
backbone remain fixed. Both nominal and sensitivity Q gradients pass through
all three steps; streamed pixel pullbacks and generator VJPs bound memory.
This is short projection adaptation, not full-CLIP retraining.

## Evaluation and solver

r={0,.25,1,4}, beta={1.02,1.1};16 solves,160 primal-dual updates each, lr5/rho1,
gradient norm capped at.2. Lowest-surrogate exactly geometry-feasible iterate
retained, including Base. No convergence certificate. Calibration and test each
use24 fresh models (brightness/noise/blur, four severities, two realizations).
Shift seeds differ; original future checkpoints are not reused. These stress
scenarios are not certified inner maximizers or confined to every r-ball.
Selection maximizes calibration FV, then minimizes cost, r and beta.

## Validation

16 independently checked outputs: exact geometry, one-to-one assignment,
re-decoding, pixel cost, surrogate decomposition, fresh-model predictions and
per-example probabilities. Selection ordering, input hashes and freeze hashes
also pass. Re-decoded reference validity is 100.0%; details are saved in reference_audit.json. Twelve focused tests passed (2 projection,3 pullback,7 OT). Projection-update unit tests check both standard SGD equivalence and
full pixel/Q derivatives against direct autodiff. Full-network directional checks
are in gradient_audit.json. Runtime: 54.1 minutes,
6.86 optimization worker-hours. Saved-output
feasibility: True; final-iterate feasibility: False.
All eight GPUs are used for both future models and concurrent optimization.
A streamed audit measured2.35GiB peak memory; workers are capped at6.5% of an80GiB GPU.

## Interpretation limits and figures

These are FFHQ-generated faces, not real-photo inversions; smile validity is
relative to the original CLIP prompts. Identity or background may change under
full W refinement. No human smile/morphology validation or global robustness
certificate is claimed. Historical Base and exact current decoder replay Base
are scored separately; see asset_audit.csv for numerical differences.

`figures/ffhq_update_aware/qualitative_all8.pdf` contains all fixed test IDs.
Its last two columns are abs(r.25-r0) and abs(r0-Base), at the selected beta,
with the same absolute RGB224 scale on every page. The largest-difference PDF
is explicitly post-hoc top-four selection by pixel L2 for each contrast, not
a representative sample or a method-selection criterion. Rankings are saved.

## Completion review

Both qualitative PDFs have two pages and were visually inspected. All16 saved
outputs satisfy the declared geometry tolerance; the calibration r0/beta1.02
solve retains iteration155 because iteration160 is infeasible. Only6/16 solves
retain their last iterate. This is best-feasible retention, not convergence.
The primary end-to-end gradient audit reaches0.597% relative error.

At beta1.02, test sensitivity decreases from0.012014 at r0 to0.002436 at r.25,
while FV remains100% and cost increases from47.5033 to48.0834. Largest-difference
IDs are[6,7,5,0] for both contrasts, although the heatmaps are not identical.
Visual changes include hair/background; some CLIP-valid faces do not exhibit
an unequivocal smile. The figures do not establish human-rated smile validity.
