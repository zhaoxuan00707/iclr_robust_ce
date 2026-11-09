# HELOC smooth-hinge update-aware GRACE

Calibration selects r/b_ref=3 (r=8.18883), beta=1.1. Selected test Future Validity is 100.00%, current validity 100.00%, cost 1.6890; Base FV is 82.99%, current validity 100.00%, cost 1.2278.

The main comparison is against nominal-only optimization as well as Base. At beta1.1, r=0 already achieves99.9826% test FV with cost1.5162, whereas the selected setting achieves100% with cost1.6890 (11.39% higher). Thus the incremental threshold-validity gain from positive radius is only0.0174 percentage points in this run. Its sensitivity is65.32% lower than that nominal-only control, and worst per-scenario smooth loss falls from0.012332 to0.004707. The nominal-only solve remains iteration-limited, so this is not a comparison of proven optima. Smaller positive radii also reach100% test FV; test outcomes do not replace the frozen selection. The37.56% cost increase versus Base must accompany the larger Base-to-selected FV improvement. This retrospective, single-seed, short-update experiment does not establish a broad robustness advantage or certificate.

The inherited severity control is not monotonic in measured difficulty: Base mean FV rises from77.40% at0.5 to82.45% at1 and89.11% at2.5. Refer to these as severity-control values, not empirically ordered worst-case attacks.

Completed the first new-method HELOC seed1 experiment on eight GPUs. This is a new128-support smooth-loss experiment, not the old3336-support finite-bank comparison. No validity repair or two-leg path realization is used. Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/20260924_v1`.

## Selection and results

Calibration uses all45 existing calibration shift specifications, including severities0.5,1,2.5, with new future models trained under the prescribed update. 2 settings tie at maximum calibration FV; lower recourse cost, then lower radius/beta resolve ties. Final future scores never change the selected parameters. All20 test configurations are predeclared diagnostics; do not pick a winner from the test table.

```
            method  selected  r_ratio  beta  feasible  converged  current_validity  FV_mean   FV_std  recourse_cost  base_displacement  nominal  sensitivity  common_r_bref_surrogate
   r0.682403_beta1     False     0.25  1.00     False      False          0.992188      NaN      NaN       1.229376           0.008377 0.031492     0.048835                 0.164792
r0.682403_beta1.05     False     0.25  1.05      True       True          1.000000 0.999653 0.001628       1.457122           0.277689 0.007610     0.011153                 0.038054
 r0.682403_beta1.1     False     0.25  1.10      True       True          1.000000 1.000000 0.000000       1.543898           0.377473 0.004638     0.006591                 0.022630
 r0.682403_beta1.2     False     0.25  1.20      True       True          1.000000 1.000000 0.000000       1.543898           0.377473 0.004638     0.006591                 0.022630
          r0_beta1     False     0.00  1.00     False      False          0.984375      NaN      NaN       1.228664           0.004437 0.031662     0.049095                 0.165674
       r0_beta1.05     False     0.00  1.05      True       True          1.000000 0.999132 0.005823       1.453495           0.274334 0.007493     0.011970                 0.040166
        r0_beta1.1     False     0.00  1.10      True      False          1.000000 0.999826 0.001165       1.516237           0.347109 0.005152     0.008092                 0.027240
        r0_beta1.2     False     0.00  1.20      True      False          1.000000 0.999826 0.001165       1.516237           0.347109 0.005152     0.008092                 0.027240
    r1.36481_beta1     False     0.50  1.00     False      False          1.000000      NaN      NaN       1.230447           0.012173 0.031332     0.048610                 0.164018
 r1.36481_beta1.05     False     0.50  1.05      True       True          1.000000 0.999653 0.001628       1.459747           0.281054 0.007600     0.010883                 0.037307
  r1.36481_beta1.1     False     0.50  1.10      True       True          1.000000 1.000000 0.000000       1.595049           0.435460 0.003590     0.004827                 0.016766
  r1.36481_beta1.2     False     0.50  1.20      True       True          1.000000 1.000000 0.000000       1.595049           0.435460 0.003590     0.004827                 0.016766
    r2.72961_beta1     False     1.00  1.00     False      False          0.992188      NaN      NaN       1.232560           0.019005 0.030837     0.047482                 0.160443
 r2.72961_beta1.05     False     1.00  1.05      True       True          1.000000 1.000000 0.000000       1.558717           0.427200 0.004387     0.006060                 0.020927
  r2.72961_beta1.1     False     1.00  1.10      True       True          1.000000 1.000000 0.000000       1.648343           0.495816 0.002773     0.003550                 0.012462
  r2.72961_beta1.2     False     1.00  1.20      True       True          1.000000 1.000000 0.000000       1.648343           0.495816 0.002773     0.003550                 0.012462
    r8.18883_beta1     False     3.00  1.00     False      False          0.968750      NaN      NaN       1.239102           0.041709 0.029632     0.044617                 0.151420
 r8.18883_beta1.05     False     3.00  1.05      True       True          1.000000 1.000000 0.000000       1.639050           0.569914 0.003252     0.004315                 0.015029
  r8.18883_beta1.1      True     3.00  1.10      True       True          1.000000 1.000000 0.000000       1.689006           0.542601 0.002293     0.002807                 0.009953
  r8.18883_beta1.2     False     3.00  1.20      True       True          1.000000 1.000000 0.000000       1.749022           0.609825 0.001710     0.002039                 0.007276
              Base     False     3.00   NaN      True       True          1.000000 0.829861 0.260211       1.227846           0.000000 0.031866     0.049368                 0.166623
```

Best calibration setting for each radius, then its test diagnostics:

```
           method  r_ratio  beta  test_feasible  current_validity  FV_all_severities  FV_highest_severity  recourse_cost  sensitivity
             Base      NaN   NaN           True               1.0           0.829861             0.891146       1.227846     0.049368
       r0_beta1.1     0.00   1.1           True               1.0           0.999826             1.000000       1.516237     0.008092
r0.682403_beta1.1     0.25   1.1           True               1.0           1.000000             1.000000       1.543898     0.006591
 r1.36481_beta1.1     0.50   1.1           True               1.0           1.000000             1.000000       1.595049     0.004827
 r2.72961_beta1.1     1.00   1.1           True               1.0           1.000000             1.000000       1.648343     0.003550
 r8.18883_beta1.1     3.00   1.1           True               1.0           1.000000             1.000000       1.689006     0.002807
```

FV mean/sample standard deviation are across heterogeneous45 future-model scenarios, not45 independent dataset seeds or a confidence interval. Strong-severity diagnostics are descriptive, not a second parameter-selection step. The raw surrogate differs across r; the common-radius column evaluates nominal+b_ref*sensitivity for fair objective comparison. Threshold validity remains p>=0.5 while optimization and future smooth-loss diagnostics use t*softplus((0.5-p)/t), t=0.05.

## Reused assets and faithful method implementation

Frozen current classifier and full6000-example training distribution: `/home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2/heloc`. Frozen128-factual Base populations per split: `/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc`. Preprocessing is inherited imputation/standardization with22 continuous features; neither the data nor Base is regenerated. P1 contains all3336 favorable training examples. All IDs, Base validity, P1 favorability, hashes, and split disjointness were checked. No old learned Q or attack-bank model is used in optimization.

Each split starts Q at its own128 Base CFs with uniform masses1/128 and uses its own D=W2(Base,P1). Supports are optimized in the inherited normalized22-dimensional space. Full P1 can have a different count; equal Base/Q cardinality enables exact one-to-one deployment. Exact unregularized OT plans define the two-anchor distances and their support gradients. The final assignment preserves Q up to permutation and retains every factual ID.

The update operator is **10 full-batch SGD steps, lr0.01**, binary cross entropy, starting from the same f0 and updating all parameters on all6000 training rows. This complete operator is differentiated through; it is not a truncated approximation to eight-epoch AdamW. All90 future checkpoints (45 calibration,45 test) use this same operator on shifted data and were saved anew. This makes the run comparable internally but its FV is not directly comparable with old AdamW/scratch-task005 metrics. The classifier itself is reused, not retrained from scratch.

Data uncertainty perturbs normalized covariates with labels fixed, using the empirical RMS norm. Sensitivity is sqrt(6000) times the ordinary tensor-gradient norm. Smooth loss is used in both nominal and sensitivity terms; the outer derivative includes the sensitivity gradient. Every initial objective must equal the independently audited Base smooth nominal+r*sensitivity. r/b_ref is predeclared as0,.25,.5,1,3; b_ref=2.729610610. The largest radius is a diagnostic outside any verified small-radius asymptotic regime.

Optimization uses the augmented-Lagrangian primal–dual algorithm with primal step1, rho1, max800 iterations, relative geometry tolerance1e-5 and successive-objective tolerance1e-6. Calibration final iterates: 15/20 feasible, 13/20 meet the stopping criterion. Test: 15/20 feasible, 13/20 meet the stopping criterion. Failed final iterates are retained as failures and excluded from deployment, without earlier-iterate substitution or validity correction. Small objective change is not a stationarity/global-optimality certificate. The beta=1 zero-excess case is especially restrictive with fixed128-support Q and3336-support P1: intermediate measures on a discrete Wasserstein geodesic can require more than128 atoms. This observation does not prove which fixed-cardinality solutions are optimal, and failed final iterates remain failures. Raw traces and final objective-gradient norms are available.

## Audit and limitations

Actual HELOC directional-gradient checks:

```
split  nominal  sensitivity  objective_at_b_ref  gradient_norm  finite_difference  relative_error
  cal 0.031947     0.046941            0.160077       0.063244           0.063255        0.000174
 test 0.031866     0.049368            0.166623       0.068965           0.068963        0.000036
```

Nine focused unit tests pass for smooth/hard hypergradients, empirical-RMS dual scaling, geometry and exact assignment. Independent reporting verifies all40 output permutations/geometry totals, smooth objective consistency, original input hashes, new checkpoint hashes, and frozen selection. Deterministic repeated updates agree bitwise. All8 physical GPU IDs are verified in worker logs; each worker caps its allocator at4% of device memory. Optimization runtimes exclude shared future-model generation and process startup. The report does not equate elapsed-worker seconds with measured GPU utilization.

Model diagnostics: `{'f0_train_accuracy': 0.7479999661445618, 'f0_cal_accuracy': 0.7090000510215759, 'nominal_updated_train_accuracy': 0.7459999918937683, 'nominal_updated_cal_accuracy': 0.7085000276565552}`. These are prediction checks, not evidence of recourse actionability.

Inherited future shifts use generic-population, target-side and mixed environments, with the existing mean/scale/tail operators and original IDs/seeds. The tail operator inherits its original label handling; it is an external stress protocol, not a proof of membership in the fixed-label local uncertainty ball. Some deterministic shifts coincide across repetitions or splits: `{'count': 90, 'cal_distinct_data_distributions': 39, 'test_distinct_data_distributions': 37, 'cross_split_identical_data_distributions': 11}`. Do not count identical distributions as independent replicates. Historical factuals/scenarios were previously inspected; this is retrospective evaluation, though configuration selection in this run uses calibration only.

All features remain mutable under the historical continuous relaxation. Identity projection on R^22 means no integer, immutable-feature or training-range guarantee; outside_training_range_fraction is saved per method, and training_range_by_feature.csv reports per-feature fractions and maximum normalized excursions. The inherited Base already has range excursions in85.94% of calibration CFs and85.16% of test CFs; these are not newly introduced constraints or claims of validity. W2 budget feasibility is a population geometry statement, not individual actionability. ReLU activations remain nonsmooth despite smoothing the margin loss. The first-order surrogate does not certify exact worst-case future validity, and a short prescribed10-step update does not establish robustness to arbitrary full retraining.

## Reproduction and deliverables

Task: tasks/013_heloc_smooth_hinge.md. Config: experiments/configs/heloc_smooth_hinge.json. Core: experiments/src/lift_ce/experiments/heloc_smooth_hinge.py and counterfactuals/update_aware.py. Outputs here include per-model/summary CSVs, calibration choices, per-environment/severity summaries, counts, hashes and checks. Figures are in figures/heloc_smooth_hinge/: comparison.pdf, calibration.pdf, future_vs_cost.pdf, severity.pdf and feature_changes.pdf (first8 fixed factuals). Large model/data outputs stay external. Prior runs are unchanged.

```
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/058_run_heloc_smooth.py --output /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/heloc_smooth_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/059_report_heloc_smooth.py --run /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/NEW_RUN_ID
```

Use a new raw path and a separate checkout to reproduce the repository report; this reporter refuses to overwrite its existing report.

## Same-Q gradient alignment supplement

Direct nominal/sensitivity gradients at Base and saved r/b_ref0/.25/3 Q
(beta1.1, cal/test) are in gradient_alignment/report.md and summary.csv.
Test cosines.87675–.89506: positively aligned, appreciably less parallel than
MNIST. At test Base, r/b_ref=.25 weighted-gradient ratio.97068; at selected
r/b_ref=3 ratio11.6482. Do not confuse HELOC radius ratios with MNIST absolute r.
Eight gradient-decomposition and input-hash checks pass. No experiments rerun.
