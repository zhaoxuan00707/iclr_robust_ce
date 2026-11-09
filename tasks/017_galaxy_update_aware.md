# Task 017: update-aware Galaxy directed-pair pilot

## Request and authority
Reuse the original Galaxy infrastructure and classifiers for Merging (1) → Round
Smooth (2), and Round Smooth (2) → Cigar Shaped Smooth (4). Execute after writing
this task. Use eight GPUs. Scientific authority: paper/iclr_robust_ce-62.pdf
Section 3 / Algorithm 1, with the user's smooth-hinge clarification (temperature
0.05, threshold 0.5). This supersedes the historical finite-bank objective.

## Frozen inputs
Original ten-way VGG16-BN checkpoint, WAE64 and preprocessing from
lift_ce.galaxy.assets. Reuse target02 from
/home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_pairs_merging_round_v1
and target04 from
/home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_pairs_round_cigar_v1.
Keep all 16 calibration and 16 test factuals per direction, their IDs and original
Base images. Use base_raw_generator_latents, NEVER Enc(Base). No original output
is overwritten or regenerated. Hash all inputs. Report nominal Base failures
(previous test validity: 75% and 50%), factual predictions, reference predictions,
and decoder recovery error; no favorable-outcome filtering or validity repair.
References retain the historical 16 target-class training examples. Check all
factual/train/reference split overlaps and disclose historical checkpoint reuse.

## Complete prescribed update
Use a deterministic seed-1701 class-balanced training subset: four examples per
class (40 total), excluding target-reference IDs. Keep the original VGG feature
extractor and first two fully connected layers frozen. Update ONLY the original
4096→10 final layer (weights and biases): three full-batch SGD steps, lr=0.0001,
10-class cross entropy, no decay, dropout disabled, BN statistics frozen.
This is a new, explicitly limited head-adaptation pilot, not the historical
all-parameter/two-epoch AdamW protocol. Every nominal, calibration and future
model uses exactly this update; no old future checkpoints are reused.

## Objective and derivative
Mean smooth hinge of target softmax after the full update plus r times empirical
RMS-dual pixel sensitivity: sqrt(40) ||dF/d(training pixels)||_2. Labels fixed.
Q consists of 16 equal-mass 64-D latent supports, initialized at the original
Base codes. Decoder remains differentiable. Training uncertainty is on canonical
RGB128 pixels, not on cached features. To reduce memory, compute the exact
frozen-feature Jacobian pullback norm in microbatches and its derivative with
respect to the training-feature adjoint. This is the chain rule for the full
pixel sensitivity, NOT a feature-space norm or a truncated model update.
Compare with direct autodiff on a small nonlinear model and run a Galaxy
finite-difference directional audit before full optimization. CNN/OT gradients
are local derivatives where defined, not a global smoothness certificate.

## Predeclared grid and solver
r={0,0.25,1,4}; beta_geo={1.02,1.10}: eight configurations per direction/split.
160 fixed primal-dual support updates, primal lr=5, rho=1. Exact unregularized
latent W2 constraint W2(Base,Q)+W2(Q,P1)<=beta_geo W2(Base,P1).
Tolerance 1e-5*max(1,D). Persist the best feasible evaluated iterate, including
the initial Base; rank by the same surrogate, then earlier iteration. Report
final-iterate feasibility separately, retained iteration, objective traces and
lack of convergence certification. No loss-scale early stop. Geometry-only
feasibility governs retention; current classifier validity is never a gate.
Deploy Q by exact equal-mass one-to-one Base→Q assignment, then decode.

## Calibration and final evaluation
Generate new shared head checkpoints for brightness and Gaussian pixel-noise
training shifts, severities {0.5,1,1.5,2}, two realizations each (16 per split).
Brightness multiplier 1 - .2*severity*U(.8,1.2); noise std .04*severity.
Clamp shifted training pixels to [0,1]; save paired RMS displacement certificates.
Calibration and test seeds differ (171000 vs181000); all configurations see the
same models within a split and both directions share each model. These are
predeclared stress distributions, not certified worst-case searches or samples
restricted to every candidate r-ball. Nominal no-shift update is also reported.
Choose per direction by calibration mean Future Validity, ties pixel L2 cost,
then r, beta. Freeze selection before test optimization/evaluation. Test outcomes
cannot alter parameters. Include Base, all grid cells, selected GRACE and a
matched-beta r=0 ablation. Primary validity is target probability >=0.5;
report target argmax and initial-Base-valid subset as separate diagnostics.

## Outputs and checks
New raw root /home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/<run_id>.
Config/code under experiments; summaries under results/031_galaxy_update_aware;
qualitative all-case PDFs and sensitivity/robustness plots under
figures/galaxy_update_aware. Include factual, Base, nominal-only, selected,
and matched-beta r=4; shared-scale difference heatmaps. Record seeds, commands,
GPU IDs, checkpoint and input hashes, runtimes, FV, cost, sensitivity and surrogate.
Run relevant derivative/geometry tests, strict checkpoint and code-recovery checks,
independently recompute saved-output geometry and FV, and verify selection/freeze
ordering. Update AGENTS.md, notes/decisions.md, state.yaml on completion.

Limitations: single historical classifier, small retrospective cohorts and small
update population, head-only short updates, potentially invalid Base/reference
predictions, no human morphology validation, no global robustness certificate.
Do not interpret successful job completion or surrogate reduction as proof of
robust semantic counterfactuals.

### Preparation audit amendment (before any optimization)
The original raw codes decode with a maximum 0.000751 pixel difference on the
first calibration cohort under the current deterministic float32 settings. The
strict 2e-5 replay check stopped audit_v1; it is preserved. This is much smaller
than an encoder round trip but must be visible. Audit_v2 additionally observed max error 0.002067 on Round→Cigar test.
Require deterministic-float32 max error <0.005 and independently require
original TF32-enabled/batch4 replay error <2e-5, record
max and mean per-image L2, retain Historical_Base as a separate scored control,
and use Dec(original raw codes) as Base for exact optimizer initialization. No
code changes, label corrections or Base search are performed. Store both image
versions only in the new run and report any validity difference.

### Post-audit solver stabilization, before the grid
Both full Galaxy directional checks pass (<0.9% relative error). At r=1 the
Merging→Round initial gradient norm is14.85, so an unbounded lr5 step would be
large relative to the latent population. Clip the total primal gradient L2 norm
at0.2 (maximum joint support displacement1 per iteration), retaining lr5, rho1
and160 updates. This is a numerical solver safeguard, independent of future
validity, with all radii sharing the same rule. Three derivative unit tests pass.
