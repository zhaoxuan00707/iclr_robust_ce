#!/usr/bin/env python3
"""Publish lightweight task-003 results after independent checks; no retuning."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1')
OUT=RUN/'heloc'
checks=json.loads((ROOT/'results/006_favorable_region_checks.json').read_text())
assert checks['all_bank_maxima_and_selection_recomputed']
metrics=pd.read_csv(OUT/'final_metrics.csv')
geometry=pd.read_csv(OUT/'geometry_feasibility.csv')
queries=pd.read_csv(OUT/'geometry_by_split.csv')
risks=pd.read_csv(OUT/'direct_Q_risk.csv')
curves=pd.read_csv(OUT/'lambda_curves.csv')
assert len(metrics)==20 and len(curves)==630
assert abs(curves.fresh_risk_increase).max()<1e-12
flags=queries[queries.split=='test'][['method','budget_feasible','slack']].rename(columns={'budget_feasible':'test_geometry_budget_feasible','slack':'test_geometry_slack'})
annotated=metrics.merge(flags,on='method',how='left')
annotated['interpretation']='retrospective comparison; no untouched confirmation'
annotated.loc[annotated.method.str.startswith('F_'),'interpretation']='fixed training problem feasible; heldout graph budgets infeasible; exploratory operating point only'
annotated.to_csv(OUT/'final_metrics_annotated.csv',index=False)
annotated.to_csv(ROOT/'results/006_favorable_region_metrics.csv',index=False)
risks.to_csv(ROOT/'results/006_direct_Q_risk.csv',index=False)
query_table=queries[['split','method','D','B_geo','total_length','slack','budget_feasible']]
query_table.to_csv(ROOT/'results/006_geometry_by_split.csv',index=False)
raw=pd.read_csv(OUT/'future_per_model.csv')
selected=json.loads((OUT/'selection.json').read_text())['selected']
selected={a['method']:a['lambda'] for a in selected}
pairs=[]
for method,reference in [('F_beta105','F_beta100'),('F_beta110','F_beta100'),('F_beta120','F_beta100'),
                         ('F_beta120','S_straight'),('F_beta120','S_old_waypoint'),('F_beta120','S_beta120')]:
    def series(name):
        return raw[(raw.protocol=='aligned_full_update')&(raw.method==name)&(raw['lambda']==selected[name])].set_index('model_index').validity.sort_index()
    delta=series(method)-series(reference)
    assert len(delta)==45
    pairs.append({'method':method,'reference':reference,'paired_scenarios':len(delta),
                  'mean_validity_difference':delta.mean(),'sd_of_paired_differences':delta.std(),
                  'scenario_wins':int((delta>1e-12).sum()),'scenario_ties':int((abs(delta)<=1e-12).sum()),
                  'scenario_losses':int((delta < -1e-12).sum()),'minimum_difference':delta.min(),'maximum_difference':delta.max(),
                  'uncertainty_interpretation':'descriptive across 45 fixed heterogeneous stresses; not an IID confidence interval'})
pd.DataFrame(pairs).to_csv(ROOT/'results/006_paired_future_differences.csv',index=False)
pd.DataFrame(pairs).to_csv(OUT/'paired_future_differences.csv',index=False)
primary=annotated[annotated.protocol=='aligned_full_update']
summary=primary[['method','lambda','calibration_bank_hinge','mean_future_validity','worst_future_validity','recourse_cost','current_validity']]
config=json.loads((OUT/'resolved_config.json').read_text())
text=f'''# Task 003: favorable-region geodesic-excess WRO

**Completed: fixed-training-population experiment and retrospective query diagnostics. No robustness superiority demonstrated. Heldout fixed-budget deployment is infeasible in this graph/query approximation.**

## Implementation and frozen protocol

Code: `experiments/src/lift_ce/counterfactuals/favorable_region.py` and `experiments/src/lift_ce/experiments/favorable_region_geodesic_wro_v1.py`. Configuration: `experiments/configs/favorable_region_geodesic_wro_v1.yaml`.

Directly optimized Q with 3,336 equal target-mass particles restricted to current-favorable graph nodes. The objective is maximum over full-update models of the weighted population mean true hinge at Q, subject only to the total intrinsic graph-W2 budget. There is no current-validity penalty, path-average objective, additional anchor or risk-weighted ground cost. The local solver uses neighboring-node proposals, true risk for acceptance and unregularized discrete OT for every feasibility decision. It is not a global distributional solver.

Frozen HELOC seed 1, tau=.5, original f0/CFs/targets/IDs/preprocessing, official Update and uncertainty model. Data radius r/b_ref=3, b_ref=2.7296106100082396, absolute data budget=8.188831830024718. Official Update: frozen-f0 initialization, fresh AdamW state, lr=.0002, weight decay=.0001, 8 epochs, batch256, update seed700001. Adversarial search seeds are separate. All features retain the inherited mutable continuous representation; no new hard training box or categorical/immutable actionability guarantee.

## Connectivity and numerical validity

The inherited repaired graph has 3,108 nodes and 46,692 edges. Float64 ReLU affine-piece enumeration rejects 65 edges missed by the old nine-point check. The remaining 46,627 edges retain one connected component with all source/target mass compatible; no additional bridge or constraint relaxation was necessary. Edge costs are positive symmetric Euclidean lengths. This is a numerical graph approximation, not an interval or continuous actionability certificate.

Every learned control and every evaluated intrinsic path point on train/cal/test is current-valid. Paths use fixed OT plans and graph shortest routes, with the knot at d0/(d0+d1). Cal/test points attach as single admissible leaves to the frozen graph; they cannot change training routes, Q or base distances.

## Direct-Q construction and fresh-bank results

Intrinsic D={config['D']:.12f}. Beta is a diagnostic axis, not selected using outcomes. All four training controls satisfy their exact discrete budgets. Slacks are 0, .003088936, .002394802 and .001570091. True training risk does not increase with beta; lower-budget feasible incumbents are retained.

{geometry[['beta_geo','D','B_geo','d0','d1','total_length','slack','lambda_Q','true_bank_hinge']].to_markdown(index=False)}

{risks.to_markdown(index=False)}

Three search replications at each of .25/.5/.75 on all four new paths produced 540 full-update models, including 144 radial candidates. The expanded shared bank contains all 5,387 historical attacks plus all 540 new attacks. All 5,927 candidates were evaluated on all 10 methods and 21 lambdas for train/cal/test. Fresh attacks did not increase any column maximum in this run. The old bank already dominates these new discoveries; this is finite-bank evidence, not a solved supremum or inner-gap certificate.

## Beta=1 representation boundary

P1 is retained and satisfies total graph-W2 length=D exactly. This recovers a feasible point on the shortest-geodesic restriction, not the complete geodesic optimizer. The midpoint realization contains 3,463 split atoms, 3,107 unique off-node positions, and 224 positions whose masses are not integer multiples of 1/3336. The equal-target-mass graph-node parameterization cannot represent the full continuous geodesic set. Some realized path distributions have lower training-bank risk than the learned Q: that is evidence of the representation/local solver limitation, not a path objective used in construction.

## Heldout scope: budget failure, not disconnection

Cal D=8.566627304290 and test D=8.785123009929 both exceed the largest training budget, 8.312639158606. By the triangle inequality, **no query-batch distribution can satisfy these unchanged budgets on this graph/query approximation**. All four query controls fail the absolute-budget checks. No budget, Q or query topology was retuned after this finding. This does not establish infeasibility in the unknown exact continuous intrinsic metric.

The following operating points are exact calibration-bank argmins, with smallest lambda on exact ties (all selected tied sets are singletons). They are **exploratory path comparisons, not feasible heldout deployments of the fixed-budget problem**. `deployment_calibrated` in the raw selection file records use of calibration CFs for lambda choice only; `deployment_scope.json` and the annotated metrics record the failed geometry transfer.

## Matched aligned-Update outcomes (retrospective query diagnostic)

{summary.to_markdown(index=False)}

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

Raw run: `{RUN}/`. Detailed artifacts: `{OUT}/`. Graph audit: `/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit/`.

Exact orchestration: `/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/012_run_favorable_task.py`, from `/home/xu.zhao/iclr_robust_ce/`, after the initial preparation command documented in `experiments/README.md`. The controller sets OMP/MKL threads1 and CUDA_VISIBLE_DEVICES=1,2,3,5 (four exposed GPUs), logs all phases under `orchestration/`, and uses the local launcher/source fingerprint. Historical source/results, production defaults and paper claims remain unchanged. No artifacts were committed.

Decision: retain the implementation as a completed negative/limited diagnostic. A future task would need to address graph-node decision coverage and the query-budget transfer problem before any deployment claim; do not silently modify this completed run.
'''
text=text.replace('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit/','/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit/')
(ROOT/'results/006_favorable_region.md').write_text(text)
# Add interpretation to the run's generated tables without changing numerical results.
(OUT/'interpretation_report.md').write_text(text)
report=OUT/'report.md'
report.write_text('**Scope warning:** Training controls satisfy the intrinsic budget, but calibration and test anchor distances exceed every fixed budget. Heldout metrics are exploratory only. No robustness superiority is demonstrated. See `interpretation_report.md` for the scientific conclusions and independent checks.\n\n'+report.read_text())
progress=ROOT/'results/006_favorable_region_progress.md'
progress.write_text('# Task 003: complete\n\nSee [completed report](006_favorable_region.md), [metrics](006_favorable_region_metrics.csv), and [checks](006_favorable_region_checks.json). Fixed-population controls are feasible; heldout geometry budgets fail. No robustness superiority demonstrated.\n')
state_file=ROOT/'state.yaml';state=yaml.safe_load(state_file.read_text())
state['status']='complete'
state['current_work'].update(phase='complete',report='results/006_favorable_region.md',checks='results/006_favorable_region_checks.json',
        unit_tests_passed=76,independent_completion_checks_passed=True,construction_bank_count=5387,fresh_attack_count=540,
        expanded_bank_count=5927,fresh_search_replications=3,all_intrinsic_current_validity_checks_passed=True,
        training_budget_feasible=True,calibration_budget_feasible=False,test_budget_feasible=False,
        primary_scope='fixed training population; query metrics exploratory',robustness_superiority_demonstrated=False,worker_pids=[],controller_finished=True)
state['completed_work']['favorable_region_wro']={'task':'tasks/003_baseline.md','status':'complete',
        'report':'results/006_favorable_region.md','run':str(RUN),'scope':'fixed-population constrained experiment and retrospective query diagnostics',
        'remaining_required_evaluation_cells':0,'future_deployment_feasibility_established':False}
state['next_actions']=['No required task-003 evaluation cells remain.','Keep production defaults and manuscript claims unchanged.','Await a separate task before changing graph-node coverage or heldout query/budget policy.']
state_file.write_text(yaml.safe_dump(state,sort_keys=False))
with (ROOT/'notes/decisions.md').open('a') as f:
    f.write('\nTask 003 completed: all4 training controls are exactly graph-budget-feasible and all sampled intrinsic outputs are current-valid;5387+540=5927 shared-bank models evaluated. Direct-Q risk falls .488634 to .476140; fresh discovery changes no column maximum. No robustness superiority over the old waypoint/Euclidean controls is demonstrated. Calibration/test graph anchor distances exceed every frozen budget, so query results remain exploratory and deployment unsupported. Detailed evidence: results/006_favorable_region.md. Production defaults and paper claims remain unchanged.\n')
print(summary.to_string(index=False))
