# MNIST first experiment with the revised update-aware GRACE Method

Authority: Section 3 of `paper/iclr_robust_ce-62.pdf` (2026-09-24 user instruction), with its referenced Algorithm 1 and appendix. Earlier bank-based tasks are historical, not specifications for this run.

## Scope and reusable assets

Run the existing MNIST 4→9 task with style shifts. Reuse the frozen SmallCNN and, for base generation only, StrongConvAutoencoder from `/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt`. Reuse the 64 factual test images and original Base CF images from `/home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1/population.pt`. Preserve all previous outputs.

This first run is a controlled single-seed pilot, not a full-data benchmark. Use a fixed label-stratified 256-image subset of the old training population as the empirical update distribution, 64 separately generated calibration factuals, 64 test factuals, and 64 training references favorable under f0. Freeze IDs and hashes before configuration selection. Record historical checkpoint split provenance limits.

## Exact numerical interpretation

- Per user correction, use the existing 64-dimensional encoder latent space for CF support optimization and Euclidean W2 geometry. Decode supports through the frozen decoder for all classifier evaluations and pixel costs. Report original Base and decoded-encoded Base separately, including their validity/representation gap. Training-distribution perturbations remain in image pixel space; distinguish data-side and CF-side metrics.
- Keep M=n=64 uniform support points, initialized to Enc(Base). P1 is fixed across configurations and splits. Compute each split's D from its own Base population.
- Prescribe T=3 full-batch gradient-descent updates, step size 0.01, binary cross entropy, all classifier parameters updated, frozen BatchNorm statistics and disabled dropout. The complete specified update (not a truncation of another update) is differentiated. Use this same operator for calibration and primary future evaluation. Older AdamW future checkpoints are incompatible and must not be reused as primary evidence.
- Optimize nominal updated-model mean hinge plus r times update-aware sensitivity. Perturb training pixels with labels fixed. For the unweighted tensor derivative g=dF/dh, the empirical RMS perturbation norm gives dual norm sqrt(N)*||g||_2. Differentiate through updates and sensitivity when optimizing Q; do not detach the sensitivity term.
- Use all eight GPUs for independent jobs. Use the Algorithm 1 augmented-Lagrangian primal–dual update with identity projection on the unconstrained latent domain R^64. Geometry is sum of exact unregularized empirical W2 distances; use optimal plans to differentiate squared cost and take its square root. No finite-bank maximum objective, two-leg path deployment, current-validity constraint, or validity repair.
- Predeclare r in {0,0.25,0.5}, beta in {1,1.05,1.10,1.20}, primal step size 20, rho=1, maximum 80 iterations, geometry tolerance 1e-5*max(1,D), relative objective tolerance 1e-6. Numerical smoke tests may motivate documented implementation corrections before calibration. Report solver failures honestly; do not substitute a repaired/selected historical iterate for Algorithm 1's final iterate.
- Verify final geometry using exact OT. Infeasible configurations are failures, excluded from calibration selection. Always retain Base as a separate baseline.
- Deploy feasible Q using exact minimum-cost one-to-one Base→Q assignment; verify permutation, optimal cost, and unchanged empirical Q.

## Calibration and evaluation

Re-use rotation, translation, noise transformations from the existing MNIST infrastructure. Predeclare severity multipliers {0,0.5,1,1.5,2}, nominal strengths {15 degrees,2 pixels,0.08}. Calibration uses independent transformation seeds (11000-series); evaluation uses 21000-series with three realizations per family/severity. Keep deterministic duplicate scenarios labeled; do not call them independent replicates.

Select r and beta by calibration Future Validity at severity 1; ties within numerical precision use lower pixel cost, then lower r, then lower beta. Freeze selection before test future-model scoring. Evaluate Base, the selected operating point, and beta diagnostics at the calibrated r. Do not pick a test winner. External severity is distinct from r.

## Checks and outputs

Before experiments, verify sensitivity dual normalization with an analytic example, directional finite differences through the update, and finite differences of the full objective with respect to Q, including the sensitivity gradient. Verify source checkpoints unchanged, disjoint calibration/test factuals, favorable references, identical factuals/Base for comparisons, and deterministic update reproducibility.

Save isolated raw outputs externally, configs/code under experiments/, summaries under results/024_mnist_update_aware/, and figures under figures/mnist_update_aware/. Include per-model and aggregate Future Validity, current validity, pixel L2 cost, nominal hinge, sensitivity norm, surrogate, exact geometry/feasibility, assignment checks, runtime, solver traces, selection and model manifests, and fixed-index qualitative PDF. Report failures, degenerate zero-hinge solutions, nonsmooth-CNN/local-surrogate limitations and historical-data reuse. Do not claim exact worst-case robustness from a first-order surrogate.

Finish by recording verified results and paths in AGENTS.md, notes/decisions.md and state.yaml. Do not mark failed or unfinished numerical work successful.

User steering (2026-09-24): latent-space optimization and eight GPUs supersede the initial pixel-space pilot plan. Preserve the initial pixel preparation/smoke attempt as superseded; its bitwise CUDA reproducibility check failed before experiments. Enable deterministic CUDA kernels for the revised run and recheck.
