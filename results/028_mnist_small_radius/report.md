# MNIST near-zero radius grid

Selected r=0.25, beta=1.02; test FV=100.0000%, CV=100.00%, pixel cost=5.63440.

Base test FV=80.0347%, cost=5.57546. Calibration-selected nominal-only representative: r0_beta1.02, test FV=97.0486%, cost=5.60458. Test scores average36 predeclared scenarios at severities .5/1/1.5/2; they are not comparable directly with task012's severity1-only headline. Selection is frozen before test scoring. No validity correction.

At fixed beta1.02, r0→.05 reduces calibration sensitivity by80.89%, raises FV by0.9549 percentage points, and increases pixel cost by0.01378 (0.231%). This first interval accounts for86.93% of the sensitivity reduction measured from0 to.25. Further radius increases still improve calibration FV; the protocol therefore selects on the whole grid rather than forcing the smallest positive radius. A steep finite-grid change is not proof of a discontinuity. Beta1.02 already permits substantial gains; extra geometry budget does not guarantee monotonically better numerical outputs because these are finite-iteration nonconvex solves.

## Calibration grid requested by the user

Absolute r={0,.05,.10,.25}, beta={1,1.02,1.05,1.10}. These are MNIST radius units, not HELOC r/b_ref. FV_cal averages all36 calibration scenarios; Cost_cal is pixel L2 from factual to deployed CF. For infeasible rows FV is omitted and cost/objective describe raw, non-deployable final iterates. Raw Phi_r values cannot rank different r objectives; the common-radius column fixes r=.25.

```
        method    r  beta   FV_cal  cal_FV_severity1  Cost_cal  nominal  s_Q_dual_norm    Phi_r  current_validity  feasible  converged  iterations  final_objective_gradient_norm  common_r025_surrogate
      r0_beta1 0.00  1.00 0.947049               1.0  5.966140 0.003556       0.351335 0.003556               1.0      True      False         800                       0.003394               0.091390
   r0.05_beta1 0.05  1.00      NaN               NaN  5.991166 0.001684       0.160070 0.009688               1.0     False      False         800                       0.011663               0.041702
    r0.1_beta1 0.10  1.00      NaN               NaN  6.003544 0.001166       0.112429 0.012409               1.0     False      False         800                       0.016542               0.029273
   r0.25_beta1 0.25  1.00      NaN               NaN  6.024409 0.001067       0.092710 0.024245               1.0     False      False         800                       0.040509               0.024245
   r0_beta1.02 0.00  1.02 0.986979               1.0  5.966474 0.000392       0.036140 0.000392               1.0      True       True         333                       0.000223               0.009427
r0.05_beta1.02 0.05  1.02 0.996528               1.0  5.980256 0.000093       0.006905 0.000438               1.0      True       True         350                       0.000225               0.001819
 r0.1_beta1.02 0.10  1.02 0.997830               1.0  5.985026 0.000062       0.004225 0.000485               1.0      True       True         342                       0.000240               0.001119
r0.25_beta1.02 0.25  1.02 0.999566               1.0  5.993081 0.000041       0.002509 0.000668               1.0      True       True         349                       0.000318               0.000668
   r0_beta1.05 0.00  1.05 0.986979               1.0  5.966474 0.000392       0.036140 0.000392               1.0      True       True         333                       0.000223               0.009427
r0.05_beta1.05 0.05  1.05 0.996094               1.0  5.978375 0.000111       0.008444 0.000533               1.0      True       True         275                       0.000281               0.002222
 r0.1_beta1.05 0.10  1.05 0.997830               1.0  5.985344 0.000061       0.004091 0.000470               1.0      True       True         355                       0.000233               0.001083
r0.25_beta1.05 0.25  1.05 0.999566               1.0  5.995694 0.000033       0.001875 0.000501               1.0      True       True         368                       0.000227               0.000501
    r0_beta1.1 0.00  1.10 0.986979               1.0  5.966474 0.000392       0.036140 0.000392               1.0      True       True         333                       0.000223               0.009427
 r0.05_beta1.1 0.05  1.10 0.996528               1.0  5.980256 0.000093       0.006905 0.000438               1.0      True       True         350                       0.000225               0.001819
  r0.1_beta1.1 0.10  1.10 0.997830               1.0  5.985344 0.000061       0.004091 0.000470               1.0      True       True         355                       0.000233               0.001083
 r0.25_beta1.1 0.25  1.10 0.999566               1.0  5.995495 0.000033       0.001912 0.000511               1.0      True       True         359                       0.000232               0.000511
```

Best feasible calibration row per radius (no test-driven selection):

```
   r  beta   cal_FV  pixel_cost  nominal  sensitivity  surrogate
0.00  1.02 0.986979    5.966474 0.000392     0.036140   0.000392
0.05  1.02 0.996528    5.980256 0.000093     0.006905   0.000438
0.10  1.02 0.997830    5.985026 0.000062     0.004225   0.000485
0.25  1.02 0.999566    5.993081 0.000041     0.002509   0.000668
```

Full test results:

```
        method  selected  feasible  converged  FV_mean   FV_min  current_validity  pixel_cost  nominal  sensitivity  common_r025_surrogate
   r0.05_beta1     False     False      False      NaN      NaN               1.0    5.642859 0.001684     0.156224               0.040740
r0.05_beta1.02     False      True       True 0.995660 0.953125               1.0    5.619272 0.000086     0.005882               0.001557
r0.05_beta1.05     False      True       True 0.995660 0.953125               1.0    5.619272 0.000086     0.005882               0.001557
 r0.05_beta1.1     False      True       True 0.995660 0.953125               1.0    5.619272 0.000086     0.005882               0.001557
    r0.1_beta1     False     False      False      NaN      NaN               1.0    5.658880 0.001187     0.109929               0.028669
 r0.1_beta1.02     False      True       True 0.998264 0.968750               1.0    5.624825 0.000056     0.003486               0.000928
 r0.1_beta1.05     False      True       True 0.998264 0.968750               1.0    5.624825 0.000056     0.003486               0.000928
  r0.1_beta1.1     False      True       True 0.998264 0.968750               1.0    5.624825 0.000056     0.003486               0.000928
   r0.25_beta1     False     False      False      NaN      NaN               1.0    5.686510 0.000566     0.052592               0.013714
r0.25_beta1.02      True      True       True 1.000000 1.000000               1.0    5.634401 0.000032     0.001746               0.000469
r0.25_beta1.05     False      True       True 1.000000 1.000000               1.0    5.634401 0.000032     0.001746               0.000469
 r0.25_beta1.1     False      True       True 1.000000 1.000000               1.0    5.634401 0.000032     0.001746               0.000469
      r0_beta1     False     False      False      NaN      NaN               1.0    5.608141 0.003942     0.366873               0.095660
   r0_beta1.02     False      True       True 0.970486 0.671875               1.0    5.604584 0.000345     0.029846               0.007806
   r0_beta1.05     False      True       True 0.970486 0.671875               1.0    5.604584 0.000345     0.029846               0.007806
    r0_beta1.1     False      True       True 0.970486 0.671875               1.0    5.604584 0.000345     0.029846               0.007806
          Base     False      True       True 0.800347 0.015625               1.0    5.575460 0.013868     1.240154               0.323907
```

## Protocol and provenance

Task tasks/014_mnist_small_radius.md; config experiments/configs/mnist_small_radius.json. Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_small_radius/20260924_v1`. Frozen population reused byte-for-byte from `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/20260924_smooth_v2`. Original classifier/AE checkpoint: `/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt`. All64 factuals/64 Q atoms and original latent Base codes per split,64 reference atoms,256 training examples, seed0, and full T=3 SGD lr.01 update remain fixed. No image re-encoding, Base regeneration, finite-bank objective or repaired Q. CF optimization/W2 are latent64-dimensional; data perturbations are pixel-space with empirical-RMS dual sqrt(256) normalization. Smooth probability hinge t=.05 is used consistently, while BCE trains updated models.

Use the same ALM primal step20/rho1/800 iterations and exact final OT. Calibration 13/16 feasible and 12/16 meet successive-objective stopping; test 12/16 feasible and 12/16 meet it. Failures are saved and excluded without earlier-iterate substitution. Small objective change is not stationarity/global convergence. Pure objective gradient is not a KKT residual.

Broader calibration is predeclared because previous severity1-only scores saturated. New seeds31000/41000 generate fresh stochastic shift realizations, with same rotation15/translation2/noise.08 strengths and severities. There are72 new checkpoints, but deterministic duplicate distributions remain: {'cal_unique': 28, 'test_unique': 28, 'cross_split_identical': 4, 'scenarios_per_split': 36}. These are not72 independent training datasets. Historical Base/factuals/checkpoints were already inspected; this remains a retrospective single-checkpoint pilot, not independent generalization evidence. The sensitivity surrogate is local and not a certified worst-case FV guarantee; the short update is not arbitrary retraining. Digit semantics lack independent human-label verification.

## Validation and artifacts

Ten focused tests passed. Actual CNN directional derivatives are in audit.json. Independent reporting checks all32 exact assignments and geometry totals, smooth initial/final objective arithmetic, all input/model hashes and byte-identical Base population, frozen selection and GPUs0–7 in logs. Each worker caps its allocator at6.5%. Overlapping old cells are compared in overlap_reproducibility.csv; prior outputs are never modified. Compare uncertainty/cost curves rather than asserting a mathematical discontinuity from this finite grid.

Required table: calibration_grid.csv. Per-model calibration/test CSVs, severity_summary.csv, selected parameters and checks accompany this report. Figures: figures/mnist_small_radius/comparison.pdf and qualitative.pdf (first16 fixed factuals, unfiltered). Large outputs stay external. No manuscript claims changed.

Reproduce into a fresh raw path:
```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/060_run_mnist_small_radius.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_small_radius/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_small_radius_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/061_report_mnist_small_radius.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_small_radius/NEW_RUN_ID
```
Reporter refuses an existing report; use a separate checkout for report reproduction.

## Repeated-cell numerical audit

Of12 cells shared with task012, 10 reproduce Q bitwise. The remaining two have maximum coordinate differences [{'split': 'cal', 'method': 'r0.25_beta1.05', 'max_Q_difference': 0.0035629272460937}, {'split': 'test', 'method': 'r0.25_beta1.1', 'max_Q_difference': 0.0014128684997558}]. Thus the full grid is not claimed bitwise reproducible across GPU scheduling. Saved final objectives, stopping traces and these differences remain available; no old outputs were replaced. The scientific interpretation is limited to the measured finite-iteration outcomes.

## Practical next step

The near-zero effect is now resolved at this grid scale: at beta1.02, r=.05 captures86.93% of the observed calibration sensitivity reduction between0 and.25. On the fresh test scenarios, FV progresses97.05%,99.57%,99.83%,100% while cost progresses5.6046,5.6193,5.6248,5.6344. A next confirmatory experiment should hold beta1.02 and compare r0/.05/.25 across independently trained classifier/data seeds, with the evaluation rule fixed in advance. Further broad beta expansion is less informative here. This is a recommendation only; no additional unrequested sweep was launched. A practical lower-cost choice requires a predeclared acceptable-FV/cost criterion; it must not replace this run's frozen FV-maximizing choice after viewing test outcomes.
