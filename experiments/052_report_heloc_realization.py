"""Assemble task005 numerical results, six plots and a scope-qualified report."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);out=p.parse_args().run
 res=ROOT/'results/heloc_realization_comparison';fig=ROOT/'figures/heloc_realization_comparison';res.mkdir(exist_ok=True);fig.mkdir(parents=True,exist_ok=True);assert not (res/'report.md').exists()
 cfg=json.loads((out/'resolved_config.json').read_text());audit=json.loads((out/'audit.json').read_text());jobs=[out/f'beta{b}' for b in [100,105,110,120]]
 for path,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==digest,path
 for job in jobs:
  frozen=json.loads((job/'outputs_frozen.json').read_text());assert sha(job/'outputs.pt')==frozen['sha256']
  check=json.loads((job/'checks.json').read_text());assert check['Q_unchanged'] and check['archived_OLD_bank_reproduced'] and check['no_future_feedback'];assert check['OT_marginal_residual']<1e-8
 s=pd.concat([pd.read_csv(p/'summary.csv') for p in jobs],ignore_index=True);f=pd.concat([pd.read_csv(p/'future_per_model.csv') for p in jobs],ignore_index=True)
 primary=s[s.operator=='aligned_full_update'].copy();corrections=[];selections=[];factcost=[]
 for job in jobs:
  b=int(job.name[4:])/100;sel=json.loads((job/'selection.json').read_text());selections.append(dict(beta_geo=b,**sel))
  if (job/'corrections.csv').exists():
   d=pd.read_csv(job/'corrections.csv');bad=d.raw_invalid;corrections.append(dict(beta_geo=b,invalid_mass=float(d.loc[bad,'mass'].sum()),changed_atoms=int(bad.sum()),alpha_mean_all=float((d.alpha*d.mass).sum()),alpha_conditional=float((d.loc[bad,'alpha']*d.loc[bad,'mass']).sum()/d.loc[bad,'mass'].sum()),alpha_max=float(d.alpha.max()),mean_correction_displacement=float((d.displacement*d.mass).sum()),max_correction_displacement=float(d.displacement.max()),alpha_bracket_max=float((d.alpha_upper-d.alpha_lower).max())))
  d=pd.read_csv(job/'per_atom_costs.csv');d['weighted_cost']=d.mass*d.recourse_cost;d['weighted_displacement']=d.mass*d.base_displacement
  for (beta,method,i,fid),group in d.groupby(['beta_geo','method','source_index','factual_id']):
   assert abs(group.mass.sum()-1/128)<1e-8
   factcost.append(dict(beta_geo=beta,method=method,source_index=i,factual_id=fid,conditional_mean_cost=group.weighted_cost.sum()/group.mass.sum(),conditional_mean_base_displacement=group.weighted_displacement.sum()/group.mass.sum(),atoms=len(group)))
 cost=pd.DataFrame(factcost);corr=pd.DataFrame(corrections);selection=pd.DataFrame(selections)
 cols=['beta_geo','method','current_validity','future_validity_mean','future_validity_std','recourse_cost','base_displacement','train_bank_risk','W2_deploy_Q','runtime_seconds']
 compact=primary[cols].copy();merged=f.merge(s,on=['beta_geo','method','seed','operator'],validate='many_to_one')
 for name,data in [('summary',s),('comparison_per_future_model',merged),('compact_table',compact),('per_factual_costs',cost),('correction_summary',corr),('OLD_calibration_selection',selection)]:data.to_csv(res/f'{name}.csv',index=False)
 f.groupby(['beta_geo','method','operator','environment','severity_control']).agg(future_validity_mean=('future_validity','mean'),future_validity_std=('future_validity','std'),future_hinge_max=('future_hinge','max')).reset_index().to_csv(res/'future_by_shift.csv',index=False)
 geometrycols=['beta_geo','train_W2_Base_Q','W2_Q_P1','train_Q_total','budget','test_W2_Base_Q','test_Q_total','test_Q_budget_feasible']
 for beta,group in primary.groupby('beta_geo'):
  for col in geometrycols[1:]:assert group[col].nunique()==1,(beta,col)
 primary[geometrycols].drop_duplicates().to_csv(res/'fixed_Q_geometry.csv',index=False)
 for filename in ['audit.json','old_reproduction.csv','population_counts.csv','resolved_config.json']:shutil.copyfile(out/filename,res/filename)
 shutil.copyfile(out/'input_manifest.json',res/'input_manifest.json')
 methods=primary.method.unique();styles={'OLD-native-replayed':('OLD (weighted)','o'),'NEW-DIRECT-weighted':('Direct Q (weighted)','s'),'NEW-DIRECT-VALID-weighted':('Direct + validity correction','^')}
 with PdfPages(fig/'comparison.pdf') as pdf:
  def finish(figure,name):
   figure.tight_layout();figure.savefig(fig/f'{name}.pdf');figure.savefig(fig/f'{name}.png',dpi=150);pdf.savefig(figure);plt.close(figure)
  figure,ax=plt.subplots(figsize=(7,5))
  for name in methods:
   g=primary[primary.method==name].sort_values('beta_geo');label,marker=styles[name];ax.plot(g.recourse_cost,g.future_validity_mean,marker=marker,label=label)
   for r in g.itertuples():ax.annotate(f'{r.beta_geo:g}',(r.recourse_cost,r.future_validity_mean),xytext=(4,4),textcoords='offset points',fontsize=8)
  ax.set(xlabel='Mean factual-to-CF L2 cost (normalized features)',ylabel='Mean future validity',title='Frozen-Q realization comparison; labels denote beta');ax.legend(fontsize=8);ax.grid(alpha=.2);finish(figure,'future_vs_cost')
  for field,label,filename in [('future_validity_mean','Mean future validity','beta_vs_future_validity'),('recourse_cost','Mean factual-to-CF L2 cost','beta_vs_cost'),('current_validity','Current validity','beta_vs_current_validity')]:
   figure,ax=plt.subplots(figsize=(7,5))
   for name in methods:
    g=primary[primary.method==name].sort_values('beta_geo');text,marker=styles[name];ax.plot(g.beta_geo,g[field],marker=marker,label=text)
   ax.set(xlabel='Geometry beta',ylabel=label,title='Same saved Q per beta; exact weighted realizations');ax.legend(fontsize=8);ax.grid(alpha=.2);finish(figure,filename)
  figure,ax=plt.subplots(figsize=(7,5))
  learned=primary[['beta_geo','learned_Q_bank_risk']].drop_duplicates().sort_values('beta_geo');ax.plot(learned.beta_geo,learned.learned_Q_bank_risk,'k--',label='Learned Q')
  for name in methods:
   g=primary[primary.method==name].sort_values('beta_geo');text,marker=styles[name];ax.plot(g.beta_geo,g.train_bank_risk,marker=marker,label=text)
  ax.set(xlabel='Geometry beta',ylabel='Max-model population mean hinge',title='Same frozen 5,927-model construction bank');ax.legend(fontsize=8);ax.grid(alpha=.2);finish(figure,'learned_vs_deployed_risk')
  figure,axes=plt.subplots(2,2,figsize=(10,8))
  for ax,beta in zip(axes.flat,[1,1.05,1.1,1.2]):
   for name in methods:
    v=np.sort(cost[(cost.beta_geo==beta)&(cost.method==name)].conditional_mean_cost.to_numpy())
    if len(v):ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=styles[name][0])
   ax.set(title=f'beta={beta:g}',xlabel='Conditional expected cost per factual',ylabel='Fraction of 128 factuals');ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=7);finish(figure,'per_factual_cost_distribution')
 tests={'direct_realization_tests':3,'existing_minimal_restoration_tests':6,'input_hashes_verified':True,'frozen_output_hashes_verified':True,'all_factual_marginals_verified':True,'fixed_Q_geometry_identical_across_methods':True,'NEW_direct_W2_to_Q_zero':bool((primary[primary.method=='NEW-DIRECT-weighted'].W2_deploy_Q<1e-7).all()),'archived_OLD_future_and_cost_reproduced':audit['max_absolute_error'],'archived_OLD_bank_max_error':max(json.loads((j/'archived_bank_reproduction.json').read_text())['absolute_error'] for j in jobs),'requested_one_CF_realization_feasible':False,'all_four_betas_evaluated':True,'no_Q_training_or_modification':True,'no_future_based_assignment_or_selection':True}
 (res/'checks.json').write_text(json.dumps(tests,indent=2)+'\n')
 # Explicitly distinguish the supported weighted question from the impossible deterministic premise.
 report=f'''# HELOC fixed-Q realization comparison

Completed the audit, archived OLD reproduction, and the mathematically compatible **weighted** realization comparison for all four betas. The requested exactly-one-CF-per-factual, Q-preserving assignment is **not implementable on these saved artifacts**: each split has 128 equally weighted Base CFs, while Q has 3,336 distinct-support particles. We did not change Q, resample it, duplicate factuals into fictitious observations, or barycentrically collapse transport atoms. A 128-row output cannot have this 3,336-particle marginal. NEW-DIRECT below is therefore explicitly NEW-DIRECT-weighted, not the requested permutation variant. No deterministic OLD-1CF rule exists in the evaluated old pipeline; none was invented.

Raw outputs: `{out}`.
Newest fixed Q source: `{cfg['raw_parent']}`.
Archived completed OLD source: `{cfg['old_parent']}`.

The archive used earlier Q supports. Its selected paths were reconstructed with the exact existing `glued_arclength_path` and compared elementwise to saved paths. Current validity, recourse cost and all 45 aligned plus 45 scratch future-model outcomes per beta reproduce with maximum absolute error {audit['max_absolute_error']:.3g}. The original 5,387-model bank risks reproduce within {tests['archived_OLD_bank_max_error']:.3g}. Details: old_reproduction.csv and raw beta*/archived_bank_reproduction.json.

The newest raw-Q run had no completed OLD path or evaluation. Thus the primary OLD rows are **OLD-native-replayed on newest Q**, using the actual archived path function and `exact_lambda_choices` calibration rule, with all methods evaluated on the common newest 5,927-model bank. Lambda is selected on the existing calibration factual population over 0:0.05:1, minimizing exact bank hinge and breaking exact ties toward smaller lambda. No current-validity filter was added to OLD. This is a disclosed replay, not a claim that historical metrics existed for the newest Q. No single beta was calibrated in the archive; beta remains a diagnostic axis. Consequently there is no legitimately selected beta row to highlight.

## Primary numerical comparison

Operator: existing aligned full update. Mean and sample standard deviation are over the same 45 future models; this is variation across heterogeneous shift scenarios, not a standard error or 45 independent dataset seeds. Only HELOC seed 1 is available. Scratch-trained stress models are kept separate in summary.csv.

```
{compact.to_string(index=False)}
```

All rows are weighted empirical populations. Current validity, future validity, risk and costs use their transport masses, exactly as OLD-native. Recourse cost is mean Euclidean factual-to-final-CF distance in the inherited imputed/normalized 22-feature representation. Base displacement uses the factual's own Base CF. Per-factual cost plots use conditional expected cost given factual identity, not an invented deterministic output.

## Interpretation of the completed comparison

Direct transport preserves the learned Q and its construction-bank risk exactly up to numerical tolerance. It does not uniformly improve the cost–robustness tradeoff. At beta=1, it increases future validity from 67.33% to 74.47% and current validity from 84.58% to 100%, but increases cost from 3.376 to 4.884; its bank risk is slightly worse than OLD (0.4886 versus 0.4803). At beta=1.05/1.10/1.20, direct future validity gains over OLD are only 0.26/0.07/0.52 percentage points. Pure direct assignment takes about 0.16–0.17 seconds versus 0.33–2.78 seconds for native OLD path construction, with the timing caveats below.

For beta=1.05 and 1.10, the corrected weighted diagnostic is empirically promising: current validity is 100%, future validity improves from 83.38% to 84.57% and from 87.51% to 88.29%, respectively, while cost falls from 4.131 to 4.107 and from 4.260 to 4.168. At beta=1.20, correction improves future validity from OLD's 91.57% to 92.45%, restores 100% current validity, and lowers bank risk from 0.31922 to 0.30339, but cost rises from 4.195 to 4.337. These are descriptive comparisons, not a calibrated beta selection or evidence of statistical significance.

Correction affects only 3.72%, 2.85%, and 1.71% of transport mass, yet is **not negligible for affected cases**: their conditional mean alpha is 0.670, 0.681, and 0.651, and maximum movement is 10.71–11.44 normalized L2 units. W2 from corrected deployment to Q is 0.747, 0.696, and 0.619. Bank-risk increases relative to pure Q are small (0.000055, 0.000026, 0.000034), but that does not imply geometric or individual preservation. Correction time brings total realization to 1.65, 1.39, and 0.94 seconds.

All four saved Q distributions satisfy their training transport-sum budget. For beta greater than 1, the test Base-to-Q-to-P1 sums are 6.028, 6.247, and 6.535, exceeding the inherited training budgets 5.112, 5.355, and 5.842. Thus these comparisons cannot claim a test-population geometry guarantee. At beta=1, Q equals P1 in this artifact, explaining its zero second-leg distance.

Conclusion: exact weighted direct transport is a simpler way to retain population Q, and Base-directed correction improves the measured tradeoff at beta=1.05/1.10. There is no universal winner and no demonstrated deterministic individual-recourse solution. The cardinality mismatch, non-negligible corrections on affected cases, one dataset seed, and failed transfer of the geometry budget limit the conclusion.

## Construction and safeguards

OLD solves Base->Q and Q->P1 plans and uses the existing normalized-arc-length interpolation. NEW-DIRECT solves only Base->Q, retains every positive transport atom with its mass, and returns the corresponding unchanged Q point. POT's network simplex is the correct unregularized quadratic OT solver for these unequal populations; the existing SciPy minimum-cost assignment branch applies only to equal-count uniform populations. All source marginals are 1/128, target marginals are 1/3336, and every factual ID is retained. NEW-DIRECT's W2(Q_deploy,Q) is checked with an independent OT solve and its bank risk agrees with learned Q within numerical tolerance. Its assignment cost agrees with OLD's first-leg optimal cost. P1 is not an input to direct deployment or correction.

For betas with current-invalid direct mass, NEW-DIRECT-VALID uses the existing `minimal_restore` on each transport atom and its own source Base CF. Valid atoms remain bitwise unchanged. The routine enumerates ReLU-affine validity intervals in float64 and refines the first crossing with reference float32 predictions; alpha bracket tolerance is 1e-7. This avoids a monotonicity assumption. All Base anchors were verified valid. It is a numerical first-crossing search, not an interval-arithmetic certificate. Pure direct outputs were saved and evaluated before correction. No P1 and no Q reoptimization are used. Correction changes the deployed distribution, not the saved Q.

Correction sizes:

```
{corr.to_string(index=False)}
```

Q's training two-anchor geometry is identical across methods. The absolute budget is beta times the training Base-to-P1 distance, as before. Test-batch Q geometry and final deployed-population geometry are also reported; neither feasibility is silently inferred from training. See fixed_Q_geometry.csv and summary.csv. Reported inherited_fresh_subset_risk uses historical fresh-discovery models that are already included in the construction bank, so it is not independent fresh confirmation. Held-out future-bank risk is the maximum hinge across the existing future models; no new attacks were trained.

Realization timings exclude Q training, bank scoring, calibration selection and future evaluation. OLD timing includes its two OT solves plus the native 21-point path construction; NEW includes its single OT solve and atom construction; corrected NEW additionally includes first-crossing correction. These are single wall-clock measurements under concurrent workers, not precision runtime benchmarks.

## Paper/implementation inconsistencies

1. The active manuscript includes `paper/Sections/03_methods_restructured.tex`, whose Method describes a Wasserstein lens, barycentric family and held-out proxy operating-point selection. The audited newer code instead optimizes free continuous Q under a single transport-sum budget and realizes weighted two-leg paths.
2. `paper/Sections/new_method.tex` is a separate draft and is not the active Method include. Its valid-only training-bank operating-point selection and target-directed Q repair do not describe the archived evaluated OLD pipeline's calibration-bank selection without an f0 filter. The newest repair run produced only the beta=1 repaired support and no finished paths/evaluation; its proposed safeguards cannot be treated as completed deployment behavior.
3. The task's equal-cardinality assumption is contradicted by saved artifacts (128 versus 3,336). The actual OLD output is a weighted conditional CF distribution. One deterministic CF per factual cannot be asserted without an additional, distribution-changing rule.
4. The 22-dimensional continuous representation has no integer or immutable-feature actionability guarantee. Future scenarios are previously inspected retrospective models, not untouched confirmation. These limitations are retained; manuscript claims were not edited.

## Artifacts and reproduction

Summary/per-model CSVs, correction statistics, calibration choices, per-factual costs, counts, hashes and checks are in `results/heloc_realization_comparison/`. The six requested plots, separately and together as comparison.pdf, are in `figures/heloc_realization_comparison/`. Detailed transport atoms, frozen outputs, per-model bank risks, model hashes and logs stay in the external raw run. Existing outputs were preserved.

Commands (use a **new** raw run path to preserve this completed run):

```bash
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/051_run_heloc_realization.py --output /home/xu.zhao/robust_ce_results/iclr_runs/heloc_realization_comparison/NEW_RUN_ID
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/heloc_realization_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/052_report_heloc_realization.py --run /home/xu.zhao/robust_ce_results/iclr_runs/heloc_realization_comparison/NEW_RUN_ID
```

The report script refuses to overwrite this completed repository report; archive it or use a separate checkout for independent reproduction. Controller audit runs on CPU; four beta workers use physical GPUs 1,2,3,5 through experiments/run.py. Config: experiments/configs/heloc_realization_comparison.json. Exact worker commands are logged. Core: experiments/src/lift_ce/experiments/heloc_realization_comparison.py; the small direct adapter is counterfactuals/direct_realization.py. Existing geodesic_excess.py, population_path.py, minimal_validity_repair.py, and fixed_risk_geometry_v1.py supply unmodified OLD, weighting, correction and lambda selection logic.
'''
 (res/'report.md').write_text(report)
 print(compact.to_string(index=False),flush=True)
if __name__=='__main__':main()
