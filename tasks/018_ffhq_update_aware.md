# Task 018 — FFHQ face → smiling face, original CLIP, full W512

## User request and authority
Write this task before execution; then run and save results and heatmaps. User
explicitly selected the ORIGINAL CLIP classifier and removed fixed-content
requirements. Use eight GPUs. Reuse original FFHQ assets; use the same revised
method as MNIST/Galaxy: paper/iclr_robust_ce-62.pdf Section3/Algorithm1 plus smooth
hinge temperature0.05, probability threshold0.5. Do not use the GENKI head, the
four-channel editor, identity/background gates, or current-validity correction.

## Immutable assets and population
Parent: /home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1.
Original StyleGAN3-r FFHQ1024, original CLIP ViT-B/32, literal prompts
['smiling face','face']; target index0. Eight calibration and eight test factuals,
eight training reference W codes, and actual original nominal Base W512 codes
are retained. The original robust/LeapFactual comparator is NOT the Base.
No inversion, re-encoding, new Base generation, or old-output modification.
Record input hashes, class predictions, factual IDs, and decoder replay errors.
Retain all fixed cases, including failed Base outputs. Score Historical_Base
separately from exact current decoder replay Base if numerical differences occur.
These are generated FFHQ-style faces, not inversion of real FFHQ photographs.

## Complete prescribed model update
Select32 original generated training images,16 per original frozen CLIP
pseudo-label, with seed1801; exclude the eight reference IDs. Freeze CLIP visual
backbone, text embeddings and logit scale. Update its original768x512 visual
projection only: three full-batch SGD steps at lr0.0001, two-class cross entropy,
no weight decay, BN/inference statistics fixed, dropout off. Differentiate all
three steps; this is a complete short projection-adaptation protocol, not the
old five-epoch AdamW update. All future models use this identical operator.

## Objective and realization
Optimize eight equal-mass supports Q in full unconstrained W512; each W is
repeated over the original16 StyleGAN layers. Start at actual Base codes.
F(Q)=mean .05*softplus((.5-p_smile(Update(f0,P0),Dec(Q)))/.05).
Minimize F(Q)+r*sqrt(32)*||dF/d(training RGB224 pixels)||_2.
Labels are fixed. Use exact pixel pullback through the frozen CLIP encoder,
streamed in microbatches (the audited Galaxy chain-rule helper), and streamed
StyleGAN feature VJPs to bound memory. This is not feature-space uncertainty;
there is no truncated update or old finite-bank worst-model objective.
Validate the complete Q gradient by finite differences before the full grid.
For higher-order training-pixel derivatives, use the mathematical attention
backend; record numeric differences from saved historical inference if present.

Exact unregularized Euclidean W2 in raw W512 for all three geometry distances:
W2(Base,Q)+W2(Q,P1)<=beta_geo*W2(Base,P1). Deploy using an exact one-to-one Base→Q
assignment, then the same global decoder. No factual-content conditioning or
identity restrictions. Images may change identity; report this honestly.

## Predeclared grid and optimizer
r={0,.25,1,4}, beta_geo={1.02,1.10}: eight settings per split,16 solves total.
160 fixed primal-dual updates; lr5, rho1, total gradient L2 clipped at.2;
geometry tolerance1e-5*max(1,D). Retain the lowest-surrogate EXACTLY geometry-
feasible evaluated iterate, including Base; no validity-based gate or early stop.
Save final-iterate feasibility separately and do not claim convergence.

## Calibration and future evaluation
New common projection checkpoints for brightness, noise and blur training shifts;
severities {.5,1,1.5,2}, two realizations each,24 calibration and24 test models.
Seeds191000/201000; brightness factor1-.2*severity*U(.8,1.2), noise std.04*severity,
blur sigma.8*severity*U(.8,1.2),9x9 Gaussian kernel. Clamp canonical pixels to[0,1]
for brightness/noise. Save paired displacement certificates. These are shared
held-out stress scenarios, not certified inner maximizers or guaranteed samples
inside every r-ball. Include a clean-update diagnostic. Select by calibration
mean FV, ties pixel224 L2 cost, r, beta; freeze before test optimization/scoring.
Score all configurations, Base and Historical_Base on identical future models.
No test-informed parameter changes. No eight-hour-long model retraining claim.

## Outputs and completion checks
Raw outputs: /home/xu.zhao/robust_ce_results/iclr_runs/ffhq_update_aware/<new_run_id>.
Config/code in experiments; concise summaries in results/032_ffhq_update_aware;
figures in figures/ffhq_update_aware. Report FV, recourse cost, nominal smooth
loss, sensitivity, surrogate, geometry and runtimes. Show all eight test faces:
Factual | Base | r0 | r.25 | selected | abs(r.25-r0) | abs(r0-Base), matched beta.
Heatmaps use mean absolute RGB differences on canonical224 images, with one
shared absolute scale. Also provide clearly labeled largest-difference examples
ranked by pixel L2, without changing any experiment choice. Preserve historical
outputs and disclose artifacts and classifier-relative smile semantics.
Run derivative/OT unit tests, full-network finite-difference audit, input/replay
checks, independently recompute saved geometry/assignment/cost and predictions,
and visually inspect figures. Update AGENTS.md, notes/decisions.md, state.yaml.
