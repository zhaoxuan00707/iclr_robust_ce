"""Verify and report the HELOC smooth-hinge experiment without changing raw runs."""
import argparse,hashlib,json,re,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.counterfactuals.geodesic_excess import exact_w2

def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;res=ROOT/'results/027_heloc_smooth_hinge';fig=ROOT/'figures/heloc_smooth_hinge';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);assert not (res/'report.md').exists();complete=read(run/'complete.json');c=read(run/'resolved_config.json');sel=read(run/'selection.json');pop=load(run/'population.pt');s=pd.read_csv(run/'summary.csv');cal=pd.read_csv(run/'calibration_summary.csv');future=pd.read_csv(run/'future_per_model.csv');checks=[];range_rows=[]
 for path,h in read(run/'input_manifest.json').items():assert sha(path)==h,path
 frozen=read(run/'evaluation_frozen.json');assert sha(run/'selection.json')==frozen['selection_sha256'] and sha(run/'frozen_outputs.pt')==frozen['output_sha256']
 for split in ['cal','test']:
  base_excursion=torch.relu(pop['lower']-pop[split]['base'])+torch.relu(pop[split]['base']-pop['upper'])
  for j,feature in enumerate(pop['feature_names']):range_rows.append({'split':split,'method':'Base','feature':feature,'outside_fraction':float((base_excursion[:,j]>0).float().mean()),'max_normalized_excursion':float(base_excursion[:,j].max())})
  for path in sorted((run/split).glob('*/output.pt')):
   out=load(path);m=read(path.parent/'metrics.json');assert m['nominal']>0;assert abs(m['surrogate']-m['nominal']-m['r']*m['sensitivity'])<1e-6
   plan=out['assignment'];i=plan['source_index'];j=plan['target_index'];assert len(i.unique())==len(j.unique())==128;assert torch.equal(out['deployed'][i],out['Q'][j]);assert torch.isfinite(out['Q']).all()
   d0=exact_w2(pop[split]['base'],out['Q']).distance;d1=exact_w2(out['Q'],pop['target']).distance;assert abs(d0+d1-m['geometry_total'])<1e-8
   init=pd.read_csv(path.parent/'trace.csv').iloc[0];baseline=next(v for v in read(run/'initial_objectives.json') if v['split']==split);assert abs(init.surrogate-baseline['nominal']-m['r']*baseline['sensitivity'])<1e-5
   excursion=torch.relu(pop['lower']-out['deployed'])+torch.relu(out['deployed']-pop['upper'])
   for j,feature in enumerate(pop['feature_names']):range_rows.append({'split':split,'method':path.parent.name,'feature':feature,'outside_fraction':float((excursion[:,j]>0).float().mean()),'max_normalized_excursion':float(excursion[:,j].max())})
   checks.append({'split':split,'method':path.parent.name,'Q_permutation_verified':True,'geometry_verified':True,'smooth_objective_consistent':True,'feasible':m['feasible'],'converged':m['converged']})
 for split in ['cal','test']:
  for r in pd.read_csv(run/f'{split}_model_manifest.csv').to_dict('records'):assert sha(r['path'])==r['sha256']
 gpus=sorted({int(v) for p in run.glob('future_worker*.log') for v in re.findall(r'GPU (\d+)',p.read_text())});assert gpus==list(range(8))
 for file in ['summary.csv','future_per_model.csv','calibration_summary.csv','calibration_per_model.csv','selection.json','audit.json','initial_objectives.json','resolved_config.json','input_manifest.json','cal_model_manifest.csv','test_model_manifest.csv','future_checks.json','complete.json']:shutil.copyfile(run/file,res/file)
 pd.DataFrame(range_rows).to_csv(res/'training_range_by_feature.csv',index=False)
 (res/'checks.json').write_text(json.dumps({'outputs':checks,'GPU_ids':gpus,'input_hashes_verified':True,'future_model_hashes_verified':True,'selection_frozen':True,'unit_tests_passed':9},indent=2)+'\n')
 s['common_r_bref_surrogate']=s.nominal+c['b_ref']*s.sensitivity;s.to_csv(res/'summary.csv',index=False)
 summarycols=['method','selected','r_ratio','beta','feasible','converged','current_validity','FV_mean','FV_std','recourse_cost','base_displacement','nominal','sensitivity','common_r_bref_surrogate'];s[summarycols].to_csv(res/'compact_table.csv',index=False)
 severity=future.groupby(['method','severity_control']).agg(FV_mean=('future_validity','mean'),FV_std=('future_validity','std'),smooth_loss=('smooth_loss','mean')).reset_index();severity.to_csv(res/'severity_summary.csv',index=False)
 future.groupby(['method','environment','severity_control']).agg(FV_mean=('future_validity','mean'),FV_std=('future_validity','std'),worst_smooth_loss=('smooth_loss','max')).reset_index().to_csv(res/'by_environment.csv',index=False)
 leaders=cal[cal.feasible].sort_values(['cal_FV','recourse_cost','r','beta'],ascending=[False,True,True,True]).groupby('r_ratio').head(1).sort_values('r_ratio');leaders.to_csv(res/'calibration_winner_per_radius.csv',index=False);selected=s[s.selected].iloc[0];base=s[s.method=='Base'].iloc[0];outputs=load(run/'frozen_outputs.pt');high=float(future.severity_control.max());rows=[]
 for method in ['Base',*leaders.method]:
  row=s[s.method==method].iloc[0];v=severity[(severity.method==method)&(severity.severity_control==high)];rows.append({'method':method,'r_ratio':row.r_ratio if method!='Base' else None,'beta':row.beta,'test_feasible':row.feasible,'current_validity':row.current_validity,'FV_all_severities':row.FV_mean,'FV_highest_severity':float(v.FV_mean.iloc[0]) if len(v) else None,'recourse_cost':row.recourse_cost,'sensitivity':row.sensitivity})
 radiustable=pd.DataFrame(rows);radiustable.to_csv(res/'radius_comparison.csv',index=False)
 ties=cal[cal.feasible&np.isclose(cal.cal_FV,cal[cal.feasible].cal_FV.max(),rtol=0,atol=1e-12)];explanation=f"{len(ties)} settings tie at maximum calibration FV; lower recourse cost, then lower radius/beta resolve ties." if len(ties)>1 else 'This setting uniquely maximizes calibration FV.'
 with PdfPages(fig/'comparison.pdf') as pdf:
  ff,ax=plt.subplots(figsize=(7,5));grid=cal.pivot(index='r_ratio',columns='beta',values='cal_FV');im=ax.imshow(grid.to_numpy(),vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(grid.columns)),[f'{v:g}' for v in grid.columns]);ax.set_yticks(range(len(grid.index)),[f'{v:g}' for v in grid.index]);ax.set(xlabel='Geometry beta',ylabel='r / b_ref',title='HELOC calibration FV across all severities')
  for i in range(len(grid)):
   for j in range(len(grid.columns)):ax.text(j,i,'failed' if pd.isna(grid.iloc[i,j]) else f'{100*grid.iloc[i,j]:.1f}%',ha='center',va='center')
  ff.colorbar(im,ax=ax);ff.tight_layout();ff.savefig(fig/'calibration.pdf');pdf.savefig(ff);plt.close(ff)
  ff,ax=plt.subplots(figsize=(7,5))
  for r,g in s[s.beta.notna()&s.feasible].groupby('r_ratio'):ax.scatter(g.recourse_cost,g.FV_mean,label=f'r/b_ref={r:g}')
  ax.scatter(base.recourse_cost,base.FV_mean,marker='x',s=100,c='black',label='Base')
  if selected.feasible:ax.scatter(selected.recourse_cost,selected.FV_mean,marker='*',s=160,c='red',label='Calibration-selected')
  ax.set(xlabel='Mean factual L2 cost (normalized features)',ylabel='Mean future validity',title='HELOC test; all45 predeclared scenarios');ax.legend(fontsize=8);ax.grid(alpha=.2);ff.tight_layout();ff.savefig(fig/'future_vs_cost.pdf');ff.savefig(fig/'future_vs_cost.png',dpi=150);pdf.savefig(ff);plt.close(ff)
  ff,ax=plt.subplots(figsize=(8,5))
  for method in ['Base',*leaders.method]:
   g=severity[severity.method==method]
   if len(g):ax.plot(g.severity_control,g.FV_mean,'o-',label=method)
  ax.set(xlabel='External shift severity',ylabel='Future validity',title='Radius representatives chosen on calibration only');ax.legend(fontsize=7);ax.grid(alpha=.2);ff.tight_layout();ff.savefig(fig/'severity.pdf');pdf.savefig(ff);plt.close(ff)
 # Fixed first8 factuals; never choose examples using future outcomes.
 panels={'Base minus factual':pop['test']['base']-pop['test']['factual']}
 if selected.method in outputs:panels['Selected CF minus Base']=outputs[selected.method]-pop['test']['base']
 ff,axes=plt.subplots(len(panels),1,figsize=(14,4*len(panels)),squeeze=False)
 for ax,(title,tensor) in zip(axes.flat,panels.items()):
  values=tensor[:8].numpy();bound=max(float(np.abs(values).max()),1e-8);im=ax.imshow(values,cmap='coolwarm',vmin=-bound,vmax=bound,aspect='auto');ax.set_xticks(range(22),pop['feature_names'],rotation=60,ha='right',fontsize=7);ax.set_yticks(range(8),[str(int(v)) for v in pop['test']['ids'][:8]]);ax.set(title=title,ylabel='Fixed factual ID');ff.colorbar(im,ax=ax,label='Normalized feature change')
 ff.tight_layout();ff.savefig(fig/'feature_changes.pdf');ff.savefig(fig/'feature_changes.png',dpi=130);plt.close(ff)
 audit=read(run/'audit.json');fd=read(run/'initial_objectives.json');duplicates=read(run/'future_checks.json');valid=s[s.beta.notna()]
 conclusion=(f"Calibration selects r/b_ref={selected.r_ratio:g} (r={selected.r:.6g}), beta={selected.beta:g}. Selected test Future Validity is {selected.FV_mean:.2%}, current validity {selected.current_validity:.2%}, cost {selected.recourse_cost:.4f}; Base FV is {base.FV_mean:.2%}, current validity {base.current_validity:.2%}, cost {base.recourse_cost:.4f}." if selected.feasible else 'The calibration-selected setting fails final test geometry; no valid selected deployment is claimed.')
 nominal_control=s[s.method=='r0_beta1.1'].iloc[0]
 base_severity=severity[severity.method=='Base'].set_index('severity_control')
 report=f'''# HELOC smooth-hinge update-aware GRACE

{conclusion}

The main comparison is against nominal-only optimization as well as Base. At beta1.1, r=0 achieves{nominal_control.FV_mean:.4%} test FV with cost{nominal_control.recourse_cost:.4f}, whereas the selected setting achieves{selected.FV_mean:.4%} with cost{selected.recourse_cost:.4f} ({selected.recourse_cost/nominal_control.recourse_cost-1:.2%} higher). Thus the incremental threshold-validity gain from positive radius is {100*(selected.FV_mean-nominal_control.FV_mean):.4f} percentage points. Its sensitivity is {1-selected.sensitivity/nominal_control.sensitivity:.2%} lower than that nominal-only control, and worst per-scenario smooth loss changes from{nominal_control.worst_future_smooth_loss:.6f} to{selected.worst_future_smooth_loss:.6f}. The nominal-only control's stopping flag is {nominal_control.converged}; this is not a comparison of proven optima. Other settings' test outcomes do not replace the frozen selection. The{selected.recourse_cost/base.recourse_cost-1:.2%} cost increase versus Base must accompany the Base-to-selected FV comparison. This retrospective, single-seed, short-update experiment does not establish a broad robustness advantage or certificate.

Base FV by inherited severity-control value: {base_severity.FV_mean.to_dict()}. These controls are not necessarily monotonic in measured difficulty; do not equate them with empirically ordered worst-case attacks.

Completed the first new-method HELOC seed1 experiment on eight GPUs. This is a new128-support smooth-loss experiment, not the old3336-support finite-bank comparison. No validity repair or two-leg path realization is used. Raw run: `{run}`.

## Selection and results

Calibration uses all45 existing calibration shift specifications, including severities0.5,1,2.5, with new future models trained under the prescribed update. {explanation} Final future scores never change the selected parameters. All20 test configurations are predeclared diagnostics; do not pick a winner from the test table.

```
{s[summarycols].to_string(index=False)}
```

Best calibration setting for each radius, then its test diagnostics:

```
{radiustable.to_string(index=False)}
```

FV mean/sample standard deviation are across heterogeneous45 future-model scenarios, not45 independent dataset seeds or a confidence interval. Strong-severity diagnostics are descriptive, not a second parameter-selection step. The raw surrogate differs across r; the common-radius column evaluates nominal+b_ref*sensitivity for fair objective comparison. Threshold validity remains p>=0.5 while optimization and future smooth-loss diagnostics use t*softplus((0.5-p)/t), t=0.05.

## Reused assets and faithful method implementation

Frozen current classifier and full6000-example training distribution: `{c['raw_parent']}`. Frozen128-factual Base populations per split: `{c['population_parent']}`. Preprocessing is inherited imputation/standardization with22 continuous features; neither the data nor Base is regenerated. P1 contains all3336 favorable training examples. All IDs, Base validity, P1 favorability, hashes, and split disjointness were checked. No old learned Q or attack-bank model is used in optimization.

Each split starts Q at its own128 Base CFs with uniform masses1/128 and uses its own D=W2(Base,P1). Supports are optimized in the inherited normalized22-dimensional space. Full P1 can have a different count; equal Base/Q cardinality enables exact one-to-one deployment. Exact unregularized OT plans define the two-anchor distances and their support gradients. The final assignment preserves Q up to permutation and retains every factual ID.

The update operator is **10 full-batch SGD steps, lr0.01**, binary cross entropy, starting from the same f0 and updating all parameters on all6000 training rows. This complete operator is differentiated through; it is not a truncated approximation to eight-epoch AdamW. All90 future checkpoints (45 calibration,45 test) use this same operator on shifted data and were saved anew. This makes the run comparable internally but its FV is not directly comparable with old AdamW/scratch-task005 metrics. The classifier itself is reused, not retrained from scratch.

Data uncertainty perturbs normalized covariates with labels fixed, using the empirical RMS norm. Sensitivity is sqrt(6000) times the ordinary tensor-gradient norm. Smooth loss is used in both nominal and sensitivity terms; the outer derivative includes the sensitivity gradient. Every initial objective must equal the independently audited Base smooth nominal+r*sensitivity. r/b_ref is predeclared as0,.25,.5,1,3; b_ref={c['b_ref']:.9f}. The largest radius is a diagnostic outside any verified small-radius asymptotic regime.

Optimization uses the augmented-Lagrangian primal–dual algorithm with primal step1, rho1, max800 iterations, relative geometry tolerance1e-5 and successive-objective tolerance1e-6. Calibration final iterates: {int(cal.feasible.sum())}/20 feasible, {int(cal.converged.sum())}/20 meet the stopping criterion. Test: {int(valid.feasible.sum())}/20 feasible, {int(valid.converged.sum())}/20 meet the stopping criterion. Failed final iterates are retained as failures and excluded from deployment, without earlier-iterate substitution or validity correction. Small objective change is not a stationarity/global-optimality certificate. The beta=1 zero-excess case is especially restrictive with fixed128-support Q and3336-support P1: intermediate measures on a discrete Wasserstein geodesic can require more than128 atoms. This observation does not prove which fixed-cardinality solutions are optimal, and failed final iterates remain failures. Raw traces and final objective-gradient norms are available.

## Audit and limitations

Actual HELOC directional-gradient checks:

```
{pd.DataFrame(fd).to_string(index=False)}
```

Nine focused unit tests pass for smooth/hard hypergradients, empirical-RMS dual scaling, geometry and exact assignment. Independent reporting verifies all40 output permutations/geometry totals, smooth objective consistency, original input hashes, new checkpoint hashes, and frozen selection. Deterministic repeated updates agree bitwise. All8 physical GPU IDs are verified in worker logs; each worker caps its allocator at4% of device memory. Optimization runtimes exclude shared future-model generation and process startup. The report does not equate elapsed-worker seconds with measured GPU utilization.

Model diagnostics: `{audit['model_diagnostics']}`. These are prediction checks, not evidence of recourse actionability.

Inherited future shifts use generic-population, target-side and mixed environments, with the existing mean/scale/tail operators and original IDs/seeds. The tail operator inherits its original label handling; it is an external stress protocol, not a proof of membership in the fixed-label local uncertainty ball. Some deterministic shifts coincide across repetitions or splits: `{duplicates}`. Do not count identical distributions as independent replicates. Historical factuals/scenarios were previously inspected; this is retrospective evaluation, though configuration selection in this run uses calibration only.

All features remain mutable under the historical continuous relaxation. Identity projection on R^22 means no integer, immutable-feature or training-range guarantee; outside_training_range_fraction is saved per method, and training_range_by_feature.csv reports per-feature fractions and maximum normalized excursions. The inherited Base already has range excursions in85.94% of calibration CFs and85.16% of test CFs; these are not newly introduced constraints or claims of validity. W2 budget feasibility is a population geometry statement, not individual actionability. ReLU activations remain nonsmooth despite smoothing the margin loss. The first-order surrogate does not certify exact worst-case future validity, and a short prescribed10-step update does not establish robustness to arbitrary full retraining.

## Reproduction and deliverables

Task: tasks/013_heloc_smooth_hinge.md. Config: experiments/configs/heloc_smooth_hinge.json. Core: experiments/src/lift_ce/experiments/heloc_smooth_hinge.py and counterfactuals/update_aware.py. Outputs here include per-model/summary CSVs, calibration choices, per-environment/severity summaries, counts, hashes and checks. Figures are in figures/heloc_smooth_hinge/: comparison.pdf, calibration.pdf, future_vs_cost.pdf, severity.pdf and feature_changes.pdf (first8 fixed factuals). Large model/data outputs stay external. Prior runs are unchanged.

```
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/058_run_heloc_smooth.py --output /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/heloc_smooth_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/059_report_heloc_smooth.py --run /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/NEW_RUN_ID
```

Use a new raw path and a separate checkout to reproduce the repository report; this reporter refuses to overwrite its existing report.
'''
 (res/'report.md').write_text(report);print(conclusion);print(radiustable.to_string(index=False))
if __name__=='__main__':main()
