"""Write lightweight results for the completed path construction gate."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/path_level_geodesic_wro_v1/20260916_path_v1')
PARENT=Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc')


def main():
    out=RUN/'heloc'
    completion=json.loads((out/'completion.json').read_text())
    if completion['status']!='COMPLETED_CONSTRUCTION_REJECTION':
        raise ValueError('This report is only for a completed all-rejected construction gate; continue evaluation otherwise.')
    trace=pd.read_csv(out/'optimization_trace_train.csv')
    statuses=pd.read_csv(out/'train_path_feasibility.csv')
    rows=[]
    for (beta,kind),g in trace.groupby(['beta_geo','initialization']):
        if set(g.step)!=set(range(121)):raise ValueError('incomplete numerical schedule')
        feasible=g[g.feasible]
        rows.append({'beta_geo':beta,'initialization':kind,'evaluated_iterates':len(g),
                     'feasible_iterates':len(feasible),'smallest_length_observed':g.total_length.min(),
                     'largest_slack_observed':g.slack.max(),
                     'smallest_invalid_mass_observed':g.maximum_grid_invalid_mass.min(),
                     'lowest_unconstrained_risk_observed':g.true_path_risk.min(),
                     'last_risk_diagnostic_only':g.iloc[-1].true_path_risk,
                     'last_geometry_slack':g.iloc[-1].slack,
                     'last_invalid_mass':g.iloc[-1].maximum_grid_invalid_mass,
                     'seconds':g.seconds.sum()})
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'initialization_ablation.csv',index=False)
    trace[['beta_geo','initialization','step','nu_geo','nu_valid','rho_geo','rho_valid','g_geo','g_valid']].to_csv(out/'dual_trace.csv',index=False)
    trace.to_csv(out/'optimization_trace.csv',index=False)
    statuses.to_csv(out/'path_feasibility.csv',index=False)
    # Explicit missing cells, not fabricated zeros for unavailable evaluations.
    pd.DataFrame([{'beta_geo':b,'split':s,'status':'NOT_RUN_TRAINING_PATH_REJECTED'}
                  for b in [1.,1.05,1.1,1.2] for s in ['cal','test']]).to_csv(out/'heldout_feasibility.csv',index=False)
    pd.DataFrame(columns=['candidate_id','source_method','source_lambda','radius','status']).to_csv(out/'fresh_attack_manifest.csv',index=False)
    pd.DataFrame(columns=['candidate_id','method','split','lambda','hinge']).to_csv(out/'cross_eval_matrix.csv',index=False)
    pd.DataFrame(columns=['method','split','lambda','bank_hinge']).to_csv(out/'lambda_curves.csv',index=False)
    pd.DataFrame([{'method':r.method,'beta_geo':r.beta_geo,'status':'REJECTED_NO_FEASIBLE_TRAINING_PATH',
                   'selected_lambda':None,'bank_hinge':None,'mean_future_validity':None,
                   'worst_future_validity':None,'recourse_cost':None} for r in statuses.itertuples()]).to_csv(out/'final_metrics.csv',index=False)
    plots=out/'plots';plots.mkdir(exist_ok=True)
    for beta,g in trace.groupby('beta_geo'):
        fig,axes=plt.subplots(1,3,figsize=(13,3.6))
        for kind,h in g.groupby('initialization'):
            for ax,column in zip(axes,['true_path_risk','slack','maximum_grid_invalid_mass']):
                ax.plot(h.step,h[column],label=kind)
        for ax,title in zip(axes,['True path risk (infeasible iterates)','Geometry slack','Worst grid invalid mass']):
            ax.set_title(title);ax.set_xlabel('Adam step');ax.grid(alpha=.2)
        axes[1].axhline(0,color='black',lw=.7);axes[-1].legend(fontsize=7)
        fig.suptitle(f'HELOC beta={beta:g}: diagnostic traces, no accepted path')
        fig.tight_layout();fig.savefig(plots/f'beta{round(beta*100):03d}_traces.png',dpi=140);plt.close(fig)
    # Before-fresh controls use exactly this run's frozen construction model set.
    bank=json.loads((out/'old_bank.json').read_text())['candidates']
    control_rows=[]
    for split in ['train','cal','test']:
        replay=np.load(PARENT/f'bank_{split}_replay.npz')
        if set(replay['candidate_ids'])!={r['candidate_id'] for r in bank}:raise ValueError('historical matrix bank mismatch')
        for j,method in enumerate(replay['methods']):
            curve=replay['hinge'][:,j,:].max(0)
            control_rows.append({'split':split,'method':str(method),'construction_path_mean_bank_risk':float(curve.mean()),
                                 'construction_minimum_bank_risk':float(curve.min()),
                                 'lambda_at_minimum':float(replay['lambdas'][np.argmin(curve)]),
                                 'role':'historical matched-bank diagnostic, not a new fresh-attack result'})
    pd.DataFrame(control_rows).to_csv(out/'historical_control_bank_diagnostic.csv',index=False)
    result=REPO/'results'
    for src,name in [(out/'initialization_ablation.csv','005_path_optimization_summary.csv'),
                     (out/'path_feasibility.csv','005_path_feasibility.csv'),
                     (out/'final_metrics.csv','005_path_final_metrics.csv'),
                     (out/'historical_control_bank_diagnostic.csv','005_path_historical_controls.csv'),
                     (out/'completion.json','005_path_completion.json')]:shutil.copyfile(src,result/name)
    report='''# Task 002: path-level constrained solver result

Implemented a K=2 intermediate-distribution path solver with the predeclared
mean-grid bank objective, exact adjacent OT, normalized Wasserstein arclength,
separate geometry/current-validity constraint handling, and rejection of every
infeasible candidate. The same solver implements held-out batch inference with
fixed absolute B_geo and frozen construction information. Historical source,
models, results and manuscript claims remain unchanged.

## Completed scope

All 16 construction jobs completed: four geometry budgets, four initializations,
120 Adam steps plus final checks per job. Every iterate used all 5,387 verified
training-derived full-update models and all 21 lambda values. The fixed instance
has 128 source CFs and 3,336 target/control particles per distribution.
Execution used the four available GPUs, physical IDs 1,2,3,5.

No iterate satisfied all declared constraints. Therefore there is no selected
new lambda, deployment CF, recourse cost or accepted robustness result. Lowest
unconstrained risks in the trace are diagnostics and cannot replace the
lowest-feasible training objective.

## Training and held-out feasibility

Training: all four beta values rejected. Exact OT and reference-CPU f0 checks
were applied throughout; current-valid controls alone were insufficient.
Held-out: not run because there is no retained training path. The new batch
refinement implementation passed synthetic endpoint-budget rejection and
feasible-case tests, but full HELOC transfer remains unevaluated. Endpoint
necessary checks from the earlier preflight passed; they were never a path
certificate.

## Robustness, fresh attacks and cost

Fresh discovery was gated off after the training-feasibility failure. Thus no
new post-construction bank, future-validity gain or matched-cost improvement is
established. Historical straight, old waypoint and previous single-control
curves are preserved and separately summarized on the matching 5,387-model
construction bank; they are not a new fresh-attack experiment. Known historical
validity/budget failures remain disclosed in results/002_baseline_audit.md.

No unrestricted geometric impossibility or general failure of path enrichment
is proved. This is failure of the declared finite-support local solver to find
a feasible iterate. Optional K=3/4 is deferred by the feasibility gate. A revised
constraint-restoration solver would be a separately versioned experiment,
not a relabeling of these outcomes.

## Artifacts and exact missing scope

Run: `RUN_PATH`.
Traces, duals, checkpoint diagnostics, exact feasibility summaries, frozen-bank
provenance and unsmoothed plots are stored there. No large artifacts were
committed. Empty fresh/replay tables explicitly represent unrun evaluations;
missing metrics are null, not zero.

Missing/gated: cal/test constrained path solves; post-construction searches and
full-update replays; expanded-bank lambda selection; future metrics/cost for an
accepted new path; optional larger K. These were not silently substituted by
historical outcomes.

Finite bank, finite support, discrete validity grid and numerical OT remain
limitations. There is no unrestricted global optimum or continuous intrinsic
geodesic guarantee.
'''.replace('RUN_PATH',str(RUN))
    (out/'report.md').write_text(report)
    (result/'005_path_level_task_report.md').write_text(report)
    print(summary.to_string(index=False))

if __name__=='__main__':main()
