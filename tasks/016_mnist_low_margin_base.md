# MNIST lower-margin Base CFs: controlled robustness and visual refinement

Status: execution authorized by the user; implementation and checks in progress. See state.yaml for live run status.

## Purpose and scope

The user requests another round with worse Base CF quality, following task015. Interpret “worse” as less classification margin and potentially less future validity, not deliberately corrupted images or invalid CFs. Test whether GRACE can recover robustness from these weaker starting populations with useful visual changes. Lower margin does not guarantee lower future validity or worse perceptual quality; measure these separately and report a negative result if the hypothesis fails.

Follow AGENTS.md, Section3/Algorithm1 of paper/iclr_robust_ce-62.pdf and the user's smooth-hinge clarification: t*softplus((0.5-p)/t), t=.05. No post-GRACE validity correction, bank-max objective, pixel-space optimization, or image selection for attractive results. Use eight GPUs when executing. Preserve all existing outputs.

## Freeze the classifier and matched inputs

Use the original strong SmallCNN and frozen64-dimensional decoder, not the weaker classifier from task015. Task015 showed that weaker classification accuracy did not produce less robust Base CFs, and its optimization stopped after one support update. Do not change classifier quality again in this experiment.

Reuse the strong arm's population from:
`/home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/20260924_v1/strong/population.pt`.

Reuse its checkpoint/config provenance from the same directory's resolved_config.json. Keep the same64 calibration factuals,64 test factuals,64 favorable reference points,256 update images, labels, IDs, encoder/decoder and classifier. Preserve its Base image/latent pairs byte-for-byte as the historical control. Its historical85.55% Base FV is a reference from that run's scenarios, not a promised value for newly generated future scenarios.

## Construct four Base conditions before any future scoring

1. **Original Base:** the frozen task015 strong-arm Base, generated with target probability .52 and a96-step latent path. Its achieved scores can exceed .52 because of discrete first-crossing steps.
2. **Precise .52:** Base on the same factual-to-original-Base latent path with probability just above .52. This controls for the original generator's probability overshoot.
3. **Low margin .505:** same path, target probability .505.
4. **Near boundary .5001:** same path, target probability .5001.

For each factual, define z_start=Enc(factual), z_end=the saved original Base code, and z(alpha)=(1-alpha)z_start+alpha*z_end. Decode these actual codes; do not encode the saved Base image. Use one fixed path for all probability levels and search using f0 only. No future classifier, sensitivity score or visual judgment may affect Base construction.

For each target probability tau, scan alpha=0,1/4096,...,1 in increasing order, find the first sampled below/above bracket, then do24 bisection steps, preserving a below-tau lower endpoint and an at-least-tau upper endpoint. Return the upper endpoint's code and decoded image. Neural probability need not be monotonic along this path: describe this as a refined first *sampled* crossing, not a proven globally first or shortest crossing. Save brackets, endpoint probabilities and achieved error. Require tau <= p <= tau+1e-4 for successful refined cases; audit recomputation in the frozen inference mode.

Predeclare exceptions rather than silently dropping cases: if the decoded starting point is already >=tau, or the original endpoint does not bracket tau, retain that factual's original valid Base for that condition and flag the case as unchanged/unbracketed. If numerical refinement cannot meet the probability tolerance, retain original Base and report the failure. Never return an invalid CF, substitute another factual, or change the target level after inspecting results. Keep all64 factual IDs in every condition; report achieved-margin distributions and the fraction actually modified, and distinguish achieved low-margin cases from unchanged ones in diagnostics. If fewer than48/64 cases attain a condition's target on either split, report that condition as an inadequate manipulation rather than claiming a full low-margin experiment.

This is construction of a new Base generator condition before GRACE, not validity repair of optimized Q. Lower-margin images may look more ambiguous; that is not itself a success.

## Refinement design and solver control

For each of four Base conditions, use absolute MNIST r={0,.05,.10,.25}, fixed beta=1.02. Four conditions times four radii give16 configurations per split,32 solves across calibration/test. Start Q at the condition's own Base codes, with64 equal-mass supports. Use the same P1 for all arms; calculate D=W2(Base,P1) separately for each condition. Equal beta does not imply equal absolute budget, so report D, beta*D and actual transport distances.

Keep the complete prescribed T=3 full-batch SGD update, lr.01, BCE, all classifier parameters updated, frozen BN statistics and disabled dropout. Training uncertainty is in pixel space, with empirical-RMS sensitivity sqrt(256)*norm(dF/dh). CF optimization and exact W2 geometry are latent-space. Differentiate both nominal loss and sensitivity through all update steps.

Use the same ALM primal/dual update equations, primal step20, rho1 and exact final feasibility tolerance1e-5*max(1,D). To avoid the task015 one-update stopping artifact, predeclare a **fixed800-update primary comparison for every condition and radius**. Record when the historical small-objective-change criterion would have fired, but do not terminate on it. This changes the stopping policy uniformly, not the objective or update equations; do not claim numerical comparability with task015's early-stopped outputs. Record full traces, final objective/constraint gradients and final-step changes. A fixed iteration count is not convergence or global optimality. Reject nonfinite runs; retain failed outputs and report failures without post-hoc hyperparameter tuning.

Deploy only final iterates that pass independent exact geometry and one-to-one Base-to-Q assignment checks. Do not substitute earlier iterates or restore current validity after optimization. If all radii in a condition fail, report no deployable result for that condition; do not choose a favorable image instead.

## Future models and calibration

Generate one common set of72 future checkpoints for all four Base conditions:36 calibration and36 test, starting from the frozen strong f0 and using the exact same T=3 update. Rotation/translation/noise nominal strengths15/2/.08, severities .5/1/1.5/2, three realizations; seeds71000+rep for calibration and81000+rep for test. Reuse each future checkpoint across every condition/radius so comparisons are paired. Deterministic translations are duplicates across repetitions/splits; record distribution/checkpoint hashes and unique counts. These are72 scenarios, not72 independent dataset seeds.

Score each condition's Base as well as refined CFs. Choose r separately per condition using mean calibration FV across all36 scenarios; ties lower factual pixel L2 cost then lower r. Freeze all choices before test scoring. Evaluate every predeclared radius as a diagnostic, but never revise selection based on test outcomes. New random realizations do not remove the retrospective nature of previously inspected factuals/classifier/decoder.

## Required measurements and interpretation

For Base and every condition/radius, record:
- Current validity and f0 probability/margin distribution, nominal smooth loss, sensitivity dual norm, Phi_r, and common-radius nominal+.25*sensitivity. Raw Phi_r values at different r are different objectives and must not be ranked as one common risk.
- FV_cal and Cost_cal, per-model/per-severity future validity and smooth future loss, test FV and factual pixel L2 cost.
- Direct CF-minus-own-Base and CF-minus-own-r0 distances: per-factual pixel L2, mean absolute difference and max absolute difference, with paired IDs.
- Exact geometry, absolute budget, final feasibility, actual update count, historical stopping-criterion crossing, final objective change, gradient diagnostics and runtime.
- Margin-target success/fallback counts and statistics for all64 cases plus clearly labeled successful-construction subsets. Main results must retain the full matched population.

First establish whether the manipulation worked: achieved f0 margins should shrink in the intended cases. Then measure whether Base FV actually falls. Only then evaluate recovery. Report separately (a) nominal refinement gain Base→r0 and (b) incremental robust gain r0→positive r. A larger visual edit does not establish better digit semantics or better recourse. The fixed classifier's own confidence cannot independently verify semantic quality.

Produce a population plot joining sensitivity, FV and cost; fixed-first8 qualitative panels; all64 test examples; signed difference maps with a common scale across conditions/radii; and explicit population-level labels for sensitivity/FV. Do not attach population sensitivity to one image as if it were an individual score. No cherry-picking of malformed/attractive examples or rescaling each difference image separately. It is acceptable to find lower Base FV but only subtle visual refinement.

## Validation and deliverables

Before optimization, verify all source hashes, paired IDs, original Base-code reconstruction, target-reference validity, common update data, and the boundary-search brackets. Add meaningful unit tests for a nonmonotonic probability path, valid-endpoint preservation, already-positive decoded starts, and unbracketed targets. Run existing smooth-hinge/dual-gradient, latent-code and exact-assignment tests plus an actual classifier/decoder directional-gradient audit. Independently verify all32 final geometries and assignments, frozen calibration selection, image metrics and all future-model hashes. Preserve old raw outputs and checkpoints; do not commit large data/model files.

Planned config: `experiments/configs/mnist_low_margin_base.json`.
Planned core: `experiments/src/lift_ce/experiments/mnist_low_margin_base.py`, using `experiments/run.py` and `experiments/README.md`.
New raw root: `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/<unique_run_id>/`.
Report: `results/030_mnist_low_margin_base/report.md`.
Figures: `figures/mnist_low_margin_base/`.
Record exact commands/config/seed/checkpoint/server paths and update AGENTS.md, notes/decisions.md and state.yaml with actual status. Do not mark results complete until execution and independent checks finish. No manuscript scientific claims should change on the basis of this plan.
