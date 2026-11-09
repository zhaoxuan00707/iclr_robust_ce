# HELOC update-aware GRACE with smooth hinge

## Authority and objective

User requests a HELOC run of the corrected method, saved results, AGENTS.md/state.yaml updates, and explicitly eight GPUs. Section3 and referenced Algorithm1 of paper/iclr_robust_ce-62.pdf are authoritative, with the user's smooth-hinge clarification overriding the hard-hinge numerical formula. Preserve all previous artifacts; do not treat old bank-based Q supports or results as the new method.

## Frozen assets and representation

Reuse HELOC seed1's saved current TabularMLP(22,[64,32]), full6000-row normalized/imputed training distribution, full3336-point favorable training reference P1, and128 saved Base CFs/factual IDs per calibration/test split. Current/training artifacts: /home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2/heloc/. Calibration/test populations: /home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc/.

Optimize in the inherited22-dimensional continuous normalized feature space, not an image latent space. All features remain mutable; use identity projection on R^22, preserving the historical continuous relaxation. Report training-range excursions; do not claim integer/immutable-feature actionability. Do not regenerate Base. Verify current Base validity, P1 favorability, IDs, dimensions, classifier compatibility and hashes.

Use M=n=128 equal-mass Q supports, initialized at the corresponding Base population, for each split. P1 may have3336 supports; only Base/Q need equal cardinality for deployment. Each split uses its own D=W2(Base,P1). Train new Q for this method; do not reuse the old3336-particle bank-trained Q. Deploy via exact minimum-cost one-to-one Base→Q assignment and verify permutation/marginal preservation.

## Objective and update operator

Smooth loss: L_t(p)=t*softplus((0.5-p)/t), t=0.05 fixed in probability space. Use it consistently in nominal population loss and sensitivity. Objective: F_Q(theta_nom)+r*||d_h F_Q(theta_T(h))|h=0||_H*. Perturb normalized training covariates with fixed labels; with empirical RMS norm, dual sensitivity equals sqrt(N)*||ordinary tensor gradient||_2. Differentiate the full prescribed update and the full outer objective, including sensitivity.

Prescribe10 full-batch gradient-descent steps, lr0.01, binary cross entropy, starting from frozen f0 and updating all parameters, using all6000 training examples. This complete10-step operator defines the experiment; it is not a truncation of the previous eight-epoch AdamW update. Reuse classifier weights and historical shift specifications, but produce new future checkpoints using this same10-step operator. No pre-existing AdamW/scratch model is primary evidence for the new operator.

No finite-model-bank maximum, two-leg path deployment, barycentric collapse, current-validity constraint or post-hoc validity repair.

## Predeclared experiment

- r/b_ref in {0,0.25,0.5,1,3}, b_ref inherited from seed1 metadata (approximately2.72961). r=0 is nominal-only control, not forced to win or excluded.
- beta_geo in {1,1.05,1.10,1.20}. Report all20 combinations, with one calibration-selected operating point.
- Augmented-Lagrangian primal–dual optimization, max800 iterations, primal step1.0, rho1, geometry tolerance1e-5*max(1,D), successive objective-change tolerance1e-6. Verify exact unregularized discrete OT on the final iterate. Do not deploy infeasible final iterates or substitute a repaired/earlier iterate. Distinguish stopping criterion from global/stationary convergence. Save full traces and final gradients.
- Calibration uses the existing45 calibration future-shift specifications across all three environments, severities0.5/1/2.5, five realizations. Use the full predeclared severity range to avoid selecting r solely on a saturated mild shift. Maximize mean calibration Future Validity; ties use lower normalized factual L2 cost, then lower r, then beta. Freeze selection before final scoring.
- Final scoring uses the existing45 final future-shift specifications with their original distinct IDs/seeds, rebuilt using the new operator. All grid rows are predeclared diagnostics; never select based on final outcomes. Disclose deterministic duplicate shifts and historical factual/scenario reuse as retrospective.

## Validation and reporting

First audit assets and write resolved configuration/input hashes. Verify smooth-hinge derivatives/dual scaling using existing unit tests plus actual HELOC directional finite differences. Require each optimizer's initial objective to match an independently audited Base smooth nominal+r*sensitivity, catching hard-hinge fallback. Release unused graphs before scalar evaluation. Check deterministic model updates, f0/nominal predictive diagnostics, exact geometry and assignments, threshold current validity, normalized factual cost, Base displacement, nominal smooth loss, sensitivity, common-radius surrogate comparisons, per-model/aggregate Future Validity and worst smooth future loss, runtime and any out-of-training-range outputs.

Use all eight GPUs for independent optimization/future-model jobs, one worker per GPU, with memory caps and separate job outputs. Do not stop others' processes. Freeze code/config before launching; preserve failed attempts and existing outputs. No large data/model files in Git.

Raw outputs: new directory under /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/. Config/code in experiments/. Summaries/report in results/027_heloc_smooth_hinge/. Figures in figures/heloc_smooth_hinge/: cost–FV, severity curves, radius/beta calibration table, and fixed-first-factual feature-change panel. Report exact commands, all input/output paths, seed, limitations and failures. Update AGENTS.md, notes/decisions.md and state.yaml only with verified status/results.
