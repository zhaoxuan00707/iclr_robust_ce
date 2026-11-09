# MNIST latent-space update-aware GRACE pilot

Calibration selected r=0, beta=1.05. On the fixed 64 test factuals at severity 1, mean Future Validity is 51.9097% versus original Base 96.0069% and decoded Base 21.5278%; pixel cost is 5.9430 versus 5.5755/5.9167. Current validity is 53.1250%. This is a single-checkpoint descriptive pilot, not statistical evidence of general superiority.

Authority: Section 3 of `paper/iclr_robust_ce-62.pdf`, task `tasks/011_mnist_update_aware.md`. This run implements the new update-aware first-order objective, not the old finite-bank maximum. No current-validity correction is applied. Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_update_aware/20260924_latent_v2`.

## Results

Future Validity mean/sample standard deviation below are across nine severity-1 scenarios (three families × three realizations). Deterministic translation repeats are duplicates; these are not nine independent training seeds. The full 45-scenario severity suite is saved separately.

```
      method  selected  feasible  current_validity  FV_mean   FV_std  pixel_cost  nominal  sensitivity  surrogate
    r0_beta1     False     False           0.15625      NaN      NaN    5.931027 0.355299     1.032921   0.355299
 r0_beta1.05      True      True           0.53125 0.519097 0.129974    5.943000 0.211037     0.347317   0.211037
  r0_beta1.1     False      True           0.53125 0.519097 0.129974    5.943000 0.211037     0.347317   0.211037
  r0_beta1.2     False      True           0.53125 0.519097 0.129974    5.943000 0.211037     0.347317   0.211037
        Base     False      True           1.00000 0.960069 0.119792    5.575523 0.000000     0.000000   0.000000
Base-decoded     False      True           0.12500 0.215278 0.132301    5.916695 0.375588     1.006962   0.375588
```

Calibration selected r=0.0, beta=1.05 on separate calibration factuals and calibration transformation seeds; all operating parameters were frozen before final scoring. 3 of 12 calibration configurations failed final exact-OT geometry. 1 test diagnostic configurations failed geometry. Failed configurations are retained with solver metrics, excluded from deployment, and not repaired or replaced with earlier iterates.

## What this first run establishes

**This pilot does not demonstrate successful robust CF generation.** Calibration selects r=0, so the deployed configuration has no sensitivity penalty. The positive-radius calibration runs perform worse: mean Future Validity is 7.47% at r=0.25 and 6.42% at r=0.5, versus 31.08% at r=0 (for feasible beta>1). These results do not establish a general failure of the proposed objective: the latent Base representation is already badly compromised, all 16 optimizations exhaust 80 iterations without satisfying the objective-change stopping tolerance, and only one short prescribed update is tested.

The major observed compatibility issue is Enc(Base)→Dec: current validity drops from 100% to 6.25% on calibration and 12.5% on test before GRACE optimization. The selected output recovers some validity but remains worse and more expensive than the original Base. The original Base has zero nominal hinge and zero first-order sensitivity in this run, yet less than 100% future validity under finite style shifts—an explicit example of the local hinge surrogate's limited predictive scope. Both f0 and the nominal three-step model retain 100% accuracy on the fixed update set (model_diagnostics.json); this is not held-out accuracy and does not certify generalization.

All three calibration beta=1 runs and the test beta=1 run fail final exact geometry. The feasible beta=1.05/1.10/1.20 outputs coincide at each radius because their geometry budgets do not constrain the observed trajectories. Do not interpret this as evidence that beta has no effect in general. The plots omit infeasible deployments, while tables retain their failure metrics.

The next useful experiment is to verify a Base generator that returns valid CFs directly on the existing decoder manifold, preserving its latent codes instead of re-encoding the output, and establish solver convergence on calibration data. This is a recommendation, not an unreported modification of this completed run; any subsequent experiment should be versioned and calibrated anew.

## Implementation and spaces

Frozen SmallCNN and StrongConvAutoencoder(64) are reused from `/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt`. Existing test factuals and Base images are reused from `/home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1`. The task is digit 4→9, not six classifiers numbered 4–9. Exact input hashes and IDs are recorded.

Q has 64 equal-weight 64-dimensional latent supports, initialized to Enc(Base). P1 contains encoded training images favorable under f0. Optimize latent support coordinates and latent Euclidean W2 geometry, decode with the frozen decoder for classifier evaluation, and report factual-to-decoded-image pixel L2 cost. One-to-one minimum-cost assignment in latent space preserves Q exactly up to permutation; every factual retains one image. Latent projection is identity on R^64. Decoder pixels remain in [0,1]. Data-side perturbations are label-preserving training-image pixel perturbations, with an empirical RMS Euclidean norm; these are distinct from the latent CF geometry.

The prescribed update is three complete full-batch gradient-descent steps, lr=0.01, binary cross entropy, on 256 fixed label-stratified training examples, starting from frozen f0. All classifier parameters update; BatchNorm statistics are frozen and dropout disabled. This exact update is differentiated and used for all primary future models. It is not a truncated derivative of an eight-epoch AdamW update. Nominal plus r times sensitivity is optimized; sensitivity is sqrt(N) times the ordinary tensor-gradient norm. Reverse-mode differentiation includes the sensitivity gradient with respect to Q.

Algorithm 1's augmented-Lagrangian primal–dual update is used, with exact OT plans and latent-distance gradients, 80 maximum iterations, primal step 20, rho=1, relative feasibility tolerance 1e-5 and objective-change tolerance 1e-6. Final unregularized OT feasibility is checked. The last iterate is retained. There is no historical bank, path interpolation, validity filter, or validity repair. Raw logs/trace.csv distinguish stopping-tolerance convergence from exhausting iterations.

## Representation and validation

```
{
  "train_n": 256,
  "cal_n": 64,
  "test_n": 64,
  "target_n": 64,
  "pixel_space": false,
  "latent_dim": 64,
  "target_current_validity": 1.0,
  "cal_base_seconds": 0.5954429134726524,
  "disjoint_cal_train_ids": true,
  "cal_test_disjoint_by_official_MNIST_split": true,
  "checkpoint_training_split_provenance": "historical source defaults, exact original invocation unknown",
  "cal_original_base_validity": 1.0,
  "cal_decoded_base_validity": 0.0625,
  "cal_decode_pixel_gap": 1.8545259237289429,
  "test_original_base_validity": 1.0,
  "test_decoded_base_validity": 0.125,
  "test_decode_pixel_gap": 1.7127666473388672
}
```

Original Base and decoded Base are both reported because decoding Enc(Base) can change images and validity. A latent empirical-population guarantee is not a guarantee that the original generator images remain unchanged.

Three analytic/numerical unit tests verify dual normalization, the full outer gradient including sensitivity, and geometry derivatives. The actual CNN+decoder directional finite-difference check has relative error 0.009871; deterministic repeated updates agree bitwise. Independent checks confirm all 16 saved assignments are permutations and preserve Q, images are finite/in-domain, and original input hashes are unchanged. Eight GPUs were used for independent calibration workers, with a per-worker allocator cap of 6.5% of device memory. Existing other-user processes were not stopped.

## Limits and interpretation

This is a controlled first pilot with one historical classifier and 64 test factuals, a 256-example empirical update distribution, and a three-step update horizon. It is not a full-data/full-training MNIST result. Original checkpoint training provenance is only partially recoverable. Calibration/test images are disjoint by the official MNIST split, but historical experiments inspected these assets previously; this is retrospective evaluation. CNN ReLU/max-pool/hinge nonsmoothness limits smooth local asymptotic interpretations, and r=0.25/0.5 is not certified to lie in an asymptotic regime. The surrogate is not a certified exact worst-case risk. Report any zero hinge/sensitivity degeneracy and geometry failures rather than claiming successful robustness. No independent semantic classifier or human label audit certifies decoded digit identity.

The prior pixel-space preparation/smoke attempt is preserved as superseded; it failed a bitwise CUDA check before experiments. The first latent smoke passed; the final run adds deterministic kernels, representation reporting, memory limits and the eight-GPU controller. Final baseline scoring initially exhausted the deliberately small GPU allocator cap by retaining unused derivative graphs. The standalone `experiments/055_finalize_mnist_update_aware.py` releases those graphs after computing the scalar metrics, without modifying optimization or model outputs; its provenance is recorded. The controller now invokes that finalizer. No old outputs were overwritten.

## Reproduction and files

```
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/053_run_mnist_update_aware.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_update_aware/NEW_RUN_ID
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_update_aware_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/054_report_mnist_update_aware.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_update_aware/NEW_RUN_ID
```

Use a new raw directory and a separate checkout for report reproduction: this reporting script refuses to overwrite an existing report. Config: `experiments/configs/mnist_update_aware.json`. Core code: `experiments/src/lift_ce/counterfactuals/update_aware.py` and `experiments/src/lift_ce/experiments/mnist_update_aware.py`; runner uses `experiments/run.py`. CSV summaries/manifests live here; large raw tensors and future checkpoints remain outside Git. Figures: `figures/mnist_update_aware/comparison.pdf` and `figures/mnist_update_aware/qualitative.pdf` (first 16 factuals by fixed order, including unsuccessful visual cases).
