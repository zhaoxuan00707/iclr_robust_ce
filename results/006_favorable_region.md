# Task 003: favorable-region geodesic-excess WRO

**Completed: fixed-training-population experiment and retrospective query diagnostics. No robustness superiority demonstrated. Heldout fixed-budget deployment is infeasible in this graph/query approximation.**

## Implementation and frozen protocol

Code: `experiments/src/lift_ce/counterfactuals/favorable_region.py` and `experiments/src/lift_ce/experiments/favorable_region_geodesic_wro_v1.py`. Configuration: `experiments/configs/favorable_region_geodesic_wro_v1.yaml`.

Directly optimized Q with 3,336 equal target-mass particles restricted to current-favorable graph nodes. The objective is maximum over full-update models of the weighted population mean true hinge at Q, subject only to the total intrinsic graph-W2 budget. There is no current-validity penalty, path-average objective, additional anchor or risk-weighted ground cost. The local solver uses neighboring-node proposals, true risk for acceptance and unregularized discrete OT for every feasibility decision. It is not a global distributional solver.

Frozen HELOC seed 1, tau=.5, original f0/CFs/targets/IDs/preprocessing, official Update and uncertainty model. Data radius r/b_ref=3, b_ref=2.7296106100082396, absolute data budget=8.188831830024718. Official Update: frozen-f0 initialization, fresh AdamW state, lr=.0002, weight decay=.0001, 8 epochs, batch256, update seed700001. Adversarial search seeds are separate. All features retain the inherited mutable continuous representation; no new hard training box or categorical/immutable actionability guarantee.

## Connectivity and numerical validity

The inherited repaired graph has 3,108 nodes and 46,692 edges. Float64 ReLU affine-piece enumeration rejects 65 edges missed by the old nine-point check. The remaining 46,627 edges retain one connected component with all source/target mass compatible; no additional bridge or constraint relaxation was necessary. Edge costs are positive symmetric Euclidean lengths. This is a numerical graph approximation, not an interval or continuous actionability certificate.

Every learned control and every evaluated intrinsic path point on train/cal/test is current-valid. Paths use fixed OT plans and graph shortest routes, with the knot at d0/(d0+d1). Cal/test points attach as single admissible leaves to the frozen graph; they cannot change training routes, Q or base distances.

## Direct-Q construction and fresh-bank results

Intrinsic D=6.927199298839. Beta is a diagnostic axis, not selected using outcomes. All four training controls satisfy their exact discrete budgets. Slacks are 0, .003088936, .002394802 and .001570091. True training risk does not increase with beta; lower-budget feasible incumbents are retained.

|   beta_geo |      D |   B_geo |      d0 |       d1 |   total_length |      slack |   lambda_Q |   true_bank_hinge |
|-----------:|-------:|--------:|--------:|---------:|---------------:|-----------:|-----------:|------------------:|
|       1    | 6.9272 | 6.9272  | 6.9272  | 0        |        6.9272  | 0          |   1        |          0.488634 |
|       1.05 | 6.9272 | 7.27356 | 6.93085 | 0.339617 |        7.27047 | 0.00308894 |   0.953288 |          0.48803  |
|       1.1  | 6.9272 | 7.61992 | 6.90158 | 0.71594  |        7.61752 | 0.0023948  |   0.906014 |          0.484452 |
|       1.2  | 6.9272 | 8.31264 | 7.06493 | 1.24614  |        8.31107 | 0.00157009 |   0.850063 |          0.47614  |

| method    |   training_bank_hinge |   fresh_bank_hinge |
|:----------|----------------------:|-------------------:|
| F_beta100 |              0.488634 |           0.488634 |
| F_beta105 |              0.48803  |           0.48803  |
| F_beta110 |              0.484452 |           0.484452 |
| F_beta120 |              0.47614  |           0.47614  |

Three search replications at each of .25/.5/.75 on all four new paths produced 540 full-update models, including 144 radial candidates. The expanded shared bank contains all 5,387 historical attacks plus all 540 new attacks. All 5,927 candidates were evaluated on all 10 methods and 21 lambdas for train/cal/test. Fresh attacks did not increase any column maximum in this run. The old bank already dominates these new discoveries; this is finite-bank evidence, not a solved supremum or inner-gap certificate.

## Beta=1 representation boundary

P1 is retained and satisfies total graph-W2 length=D exactly. This recovers a feasible point on the shortest-geodesic restriction, not the complete geodesic optimizer. The midpoint realization contains 3,463 split atoms, 3,107 unique off-node positions, and 224 positions whose masses are not integer multiples of 1/3336. The equal-target-mass graph-node parameterization cannot represent the full continuous geodesic set. Some realized path distributions have lower training-bank risk than the learned Q: that is evidence of the representation/local solver limitation, not a path objective used in construction.

## Heldout scope: budget failure, not disconnection

Cal D=8.566627304290 and test D=8.785123009929 both exceed the largest training budget, 8.312639158606. By the triangle inequality, **no query-batch distribution can satisfy these unchanged budgets on this graph/query approximation**. All four query controls fail the absolute-budget checks. No budget, Q or query topology was retuned after this finding. This does not establish infeasibility in the unknown exact continuous intrinsic metric.

The following operating points are exact calibration-bank argmins, with smallest lambda on exact ties (all selected tied sets are singletons). They are **exploratory path comparisons, not feasible heldout deployments of the fixed-budget problem**. `deployment_calibrated` in the raw selection file records use of calibration CFs for lambda choice only; `deployment_scope.json` and the annotated metrics record the failed geometry transfer.

## Matched aligned-Update outcomes (retrospective query diagnostic)

| method         |   lambda |   calibration_bank_hinge |   mean_future_validity |   worst_future_validity |   recourse_cost |   current_validity |
|:---------------|---------:|-------------------------:|-----------------------:|------------------------:|----------------:|-------------------:|
| S_straight     |     0.65 |                 0.479562 |               0.673337 |               0.0533761 |         3.37576 |           0.845792 |
| S_old_waypoint |     0.65 |                 0.441986 |               0.995344 |               0.940348  |         5.27231 |           1        |
| S_beta100      |     0.65 |                 0.479562 |               0.673337 |               0.053376  |         3.37576 |           0.845792 |
| S_beta105      |     0.7  |                 0.415438 |               0.833922 |               0.308153  |         4.23864 |           0.971916 |
| S_beta110      |     0.65 |                 0.371731 |               0.871476 |               0.466389  |         4.19715 |           0.975513 |
| S_beta120      |     0.65 |                 0.314166 |               0.92458  |               0.657974  |         4.41802 |           0.998801 |
| F_beta100      |     0.65 |                 0.481075 |               0.749119 |               0.0885229 |         4.08731 |           1        |
| F_beta105      |     0.65 |                 0.481046 |               0.752056 |               0.0913519 |         4.18285 |           1        |
| F_beta110      |     0.75 |                 0.479888 |               0.772803 |               0.0980965 |         4.60809 |           1        |
| F_beta120      |     0.75 |                 0.472935 |               0.780054 |               0.0999326 |         4.79621 |           1        |

Validity is a fraction; cost is weighted factual-to-final-CF standardized Euclidean distance. Controls are the unchanged compatible weighted Euclidean straight/old-waypoint paths and the four previous Euclidean single-budget paths. Graph and Euclidean budgets use different metrics; no matched-cost or matched-geometry superiority is claimed. Scratch-retraining stress is separately available in the metrics and is not covered by the fine-tuning protocol.

## Scientific answers

- Current validity: all intrinsic supports and sampled routes stay favorable without a penalty, within the numerical checks.
- More freedom: direct-Q risk decreases from .488634 to .476140. The realized training paths also improve overall, but single learned path families need not be nested; the nesting check concerns the direct-Q feasible sets/objective.
- Fresh attacks: these improvements persist against the expanded bank, but none of the new candidates beat the already strong old bank.
- Future validity/cost: the largest intrinsic beta improves mean/worst validity from 74.91%/8.85% to 78.01%/9.99%, while cost grows from 4.0873 to 4.7962. These are budget-infeasible query diagnostics, not deployment improvements.
- Old method comparison: the old waypoint reaches 99.53%/94.03% future validity at cost5.2723 and bank hinge.441986. The previous Euclidean beta1.20 reaches92.46%/65.80% at cost4.4180, with current validity99.88%. The intrinsic method provides no demonstrated robustness superiority. Its100% sampled current validity does not erase the other tradeoffs.
- Connectivity: construction remains connected; query budget failure is caused by the represented distances/budgets, not disconnected mass. Coarse graph/node restrictions and local proposals limit what this experiment says about continuous favorable-region WRO.

Paired scenario differences are saved in `results/006_paired_future_differences.csv`. They are descriptive across45 fixed heterogeneous scenarios, not IID confidence intervals. All future benchmark outcomes had been inspected in earlier experiments; this is not untouched confirmation.

## Validation and reproducibility

76 unit tests passed. `experiments/013_review_favorable_task.py` independently checked exact geometry, current validity, endpoint masses, unchanged historical control tensors, full shared-bank maxima, all lambda selections, frozen paths and complete radial/search coverage. Checks: `results/006_favorable_region_checks.json`.

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1/`. Detailed artifacts: `/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1/heloc/`. Graph audit: `/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit/`.

Exact orchestration: `/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/012_run_favorable_task.py`, from `/home/xu.zhao/iclr_robust_ce/`, after the initial preparation command documented in `experiments/README.md`. The controller sets OMP/MKL threads1 and CUDA_VISIBLE_DEVICES=1,2,3,5 (four exposed GPUs), logs all phases under `orchestration/`, and uses the local launcher/source fingerprint. Historical source/results, production defaults and paper claims remain unchanged. No artifacts were committed.

Decision: retain the implementation as a completed negative/limited diagnostic. A future task would need to address graph-node decision coverage and the query-budget transfer problem before any deployment claim; do not silently modify this completed run.
