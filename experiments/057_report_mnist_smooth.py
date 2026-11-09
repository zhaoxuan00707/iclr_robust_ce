"""Report the corrected latent Base + user-requested smooth-hinge pilot."""
import argparse,hashlib,json,re,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.counterfactuals.geodesic_excess import exact_w2

def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
 res=ROOT/'results/026_mnist_smooth_hinge';fig=ROOT/'figures/mnist_smooth_hinge';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);assert not (res/'report.md').exists();assert (run/'complete.json').exists()
 c=json.loads((run/'resolved_config.json').read_text());sel=json.loads((run/'selection.json').read_text());s=pd.read_csv(run/'summary.csv');cal=pd.read_csv(run/'calibration_summary.csv');f=pd.read_csv(run/'future_per_model.csv');pop=load(run/'population.pt');rec=json.loads((run/'base_recovery.json').read_text());images=load(run/'frozen_images.pt');checks=[]
 for path,h in json.loads((run/'input_manifest.json').read_text()).items():assert sha(path)==h
 assert sha(run/'frozen_images.pt')==json.loads((run/'frozen_evaluation.json').read_text())['images_sha256']
 assert sha(run/'selection.json')==json.loads((run/'frozen_evaluation.json').read_text())['selection_sha256']
 for split in ['cal','test']:
  for path in sorted((run/split).glob('*/output.pt')):
   out=load(path);metrics=json.loads((path.parent/'metrics.json').read_text());assert metrics['nominal']>0,'Loss implementation mismatch';assert abs(metrics['surrogate']-metrics['nominal']-metrics['r']*metrics['sensitivity'])<1e-6;plan=out['assignment'];i=plan['source_index'];j=plan['target_index'];assert len(i.unique())==len(j.unique())==64;assert torch.equal(out['latents'][i],out['Q'][j]);assert torch.isfinite(out['Q']).all();assert out['images'].min()>=0 and out['images'].max()<=1
   d0=exact_w2(pop[split]['base_z'],out['Q']).distance;d1=exact_w2(out['Q'],pop['target_z']).distance;assert abs(d0+d1-metrics['geometry_total'])<1e-8
   checks.append({'split':split,'method':path.parent.name,'permutation':True,'Q_preserved':True,'exact_geometry_verified':True,'base_to_Q_W2':d0,'feasible':metrics['feasible'],'converged':metrics['converged'],'stop_reason':metrics['stop_reason']})
 gpus=sorted({int(g) for p in run.glob('cal_r*.log') for g in re.findall(r'CUDA (\d+)',p.read_text())});assert gpus==list(range(8))
 for name in ['summary.csv','future_per_model.csv','calibration_summary.csv','calibration_per_model.csv','selection.json','base_recovery.json','input_manifest.json','resolved_config.json','cal_model_manifest.csv','test_model_manifest.csv','complete.json']:shutil.copyfile(run/name,res/name)
 (res/'checks.json').write_text(json.dumps({'output_checks':checks,'GPU_ids':gpus,'all_input_hashes_preserved':True,'tests_passed':14},indent=2)+'\n')
 means=f.groupby(['method','family','severity']).agg(FV_mean=('future_validity','mean'),FV_std=('future_validity','std'),worst_hinge=('hinge','max')).reset_index();means.to_csv(res/'shift_by_family.csv',index=False)
 s['common_r05_surrogate']=s.nominal+.5*s.sensitivity
 s[['method','r','beta','nominal','sensitivity','common_r05_surrogate']].to_csv(res/'common_radius_risk.csv',index=False)
 selected=s[s.selected].iloc[0];base=s[s.method=='Base'].iloc[0];calgood=cal[cal.feasible].copy();leaders=calgood.sort_values(['cal_FV','pixel_cost','r','beta'],ascending=[False,True,True,True]).groupby('r',sort=True).head(1).sort_values('r');leaders.to_csv(res/'calibration_winner_per_radius.csv',index=False)
 severity_table=f.groupby(['method','severity']).future_validity.mean().unstack();severity_table.to_csv(res/'severity_summary.csv')
 radius_rows=[]
 for method in ['Base',*leaders.method]:
  rr=s[s.method==method].iloc[0];radius_rows.append({'method':method,'r':rr.r,'beta':rr.beta,'current_validity':rr.current_validity,'FV_severity1':severity_table.loc[method,1.],'FV_severity2':severity_table.loc[method,2.],'pixel_cost':rr.pixel_cost,'sensitivity':rr.sensitivity})
 pd.DataFrame(radius_rows).to_csv(res/'radius_comparison.csv',index=False)
 ties=calgood[np.isclose(calgood.cal_FV,calgood.cal_FV.max(),atol=1e-12,rtol=0)];tie_explanation=f"{len(ties)} feasible settings tie at the highest calibration FV; lower pixel cost, then lower r and beta break the tie." if len(ties)>1 else 'One setting uniquely maximizes calibration FV.'
 with PdfPages(fig/'comparison.pdf') as pdf:
  ff,ax=plt.subplots(figsize=(7,4));grid=cal.pivot(index='r',columns='beta',values='cal_FV');im=ax.imshow(grid.to_numpy(),aspect='auto',vmin=0,vmax=1);ax.set_xticks(range(len(grid.columns)),[f'{v:g}' for v in grid.columns]);ax.set_yticks(range(len(grid.index)),[f'{v:g}' for v in grid.index]);ax.set(xlabel='Geometry beta',ylabel='Data uncertainty radius r',title='Calibration FV; blanks are infeasible final iterates')
  for row in range(len(grid)):
   for col in range(len(grid.columns)):
    v=grid.iloc[row,col];ax.text(col,row,'failed' if pd.isna(v) else f'{100*v:.2f}%',ha='center',va='center',color='black')
  ff.colorbar(im,ax=ax);ff.tight_layout();ff.savefig(fig/'calibration.pdf');pdf.savefig(ff);plt.close(ff)
  ff,ax=plt.subplots(figsize=(7,5))
  for r,g in s[s.beta.notna()&s.feasible].groupby('r'):ax.plot(g.pixel_cost,g.FV_mean,'o',label=f'r={r:g}')
  ax.scatter([base.pixel_cost],[base.FV_mean],marker='x',s=90,c='black',label='Base');ax.scatter([selected.pixel_cost],[selected.FV_mean],marker='*',s=160,c='red',label='Calibration-selected');ax.set(xlabel='Mean pixel L2 recourse cost',ylabel='Future validity at severity 1',title='All feasible test configurations; no test selection');ax.legend();ax.grid(alpha=.2);ff.tight_layout();ff.savefig(fig/'future_vs_cost.pdf');ff.savefig(fig/'future_vs_cost.png',dpi=150);pdf.savefig(ff);plt.close(ff)
  ff,ax=plt.subplots(figsize=(8,5));methods=['Base','Historical-Base']+[v for v in leaders.method if v in images]
  for method in dict.fromkeys(methods):
   g=f[f.method==method].groupby('severity').future_validity.mean();ax.plot(g.index,g.values,'o-',label=method)
  ax.set(xlabel='External style severity',ylabel='Future validity',title='Radius representatives selected only on calibration');ax.legend(fontsize=8);ax.grid(alpha=.2);ff.tight_layout();ff.savefig(fig/'severity.pdf');pdf.savefig(ff);plt.close(ff)
 labels=['Factual','Base']+[v for v in leaders.method if v in images];values=[pop['test']['factual'],images['Base']]+[images[v] for v in labels[2:]]
 with PdfPages(fig/'qualitative.pdf') as pdf:
  for offset in [0,8]:
   ff,axes=plt.subplots(8,len(labels),figsize=(2*len(labels),14),squeeze=False)
   for row in range(8):
    for col,(label,im) in enumerate(zip(labels,values)):
     ax=axes[row,col];ax.imshow(im[row+offset,0],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
     if row==0:ax.set_title(label,fontsize=8)
     if col==0:ax.set_ylabel(f"ID {int(pop['test']['ids'][row+offset])}",fontsize=8)
   ff.tight_layout();pdf.savefig(ff)
   if offset==0:ff.savefig(fig/'qualitative_preview.png',dpi=110)
   plt.close(ff)
 hist=f[(f.method=='Historical-Base')&(f.severity==1)].future_validity.mean();compact=s[['method','selected','r','beta','feasible','converged','current_validity','FV_mean','FV_std','pixel_cost','nominal','sensitivity','surrogate','common_r05_surrogate']];compact.to_csv(res/'compact_table.csv',index=False)
 report=f'''# MNIST corrected latent Base and smooth-hinge experiment

Calibration selects **r={sel['r']:g}, beta={sel['beta']:g}**. Selected test Future Validity is **{selected.FV_mean:.2%}**, versus **{base.FV_mean:.2%}** for the replayed Base and **{hist:.2%}** for historical Base. Selected current validity is **{selected.current_validity:.2%}**; pixel cost is **{selected.pixel_cost:.4f}** versus Base **{base.pixel_cost:.4f}**. These are single-checkpoint, 64-factual retrospective results, not a multi-seed benchmark.

## Why r was zero, and what changed

In task011, r=0 won because the positive-radius calibration settings performed worse after the pilot incorrectly used Enc(Base) instead of the generator's actual latent codes. That run also used hard hinge and exhausted its iteration budget. The subsequent hard-hinge preparation on corrected Base showed zero hinge and zero sensitivity, but did not complete optimization because of a retained-graph allocator-cap failure. Neither result establishes the behavior of the clarified smooth-loss method. The first smooth integration attempt also retained hard-hinge defaults in optimizer calls; it is explicitly excluded. This corrected run asserts that every initial optimization objective equals the audited smooth nominal plus r times sensitivity and rejects zero nominal loss.

This run follows the user's explicit smooth-hinge clarification, overriding the hard-hinge numerical reading of the PDF. Use L_t(p)=t*softplus((0.5-p)/t), with **t=0.05 fixed in probability space**. Both nominal loss and update-aware sensitivity use this same smooth loss, and its complete outer gradient is differentiated. Training loss remains binary cross entropy; validity remains probability>=0.5. Temperature is an explicitly chosen implementation default, not selected using future outcomes. r=0 is retained as the nominal-loss ablation; it is not forced to win or excluded to make a robustness claim.

{tie_explanation}

Best feasible calibration row per radius (chosen by calibration FV/cost only):

```
{leaders[['r','beta','cal_FV','pixel_cost','surrogate','converged']].to_string(index=False)}
```

## Base representation fix

The original generator produces images by decoding latent codes, but its CEResult discarded those codes. We now retain them, including the fallback branch, and use them directly as the Base geometry anchor. No re-encoding and no GRACE validity repair are applied. Replaying the same generator recovers calibration images exactly; historical test images differ by at most {rec[1]['replay_vs_historical_image_max_error']:.7f} per pixel, so we explicitly store newly replayed image/code pairs and preserve Historical-Base as a separate scored baseline. The first recovery attempt stopped at its strict exact-match check; no original artifact was changed.

Decoding retained codes reproduces the replayed Base exactly in this run and both calibration/test Base current validity are 100%. Compare this to task011's decoded Base current validity of 6.25%/12.5%. The correction is to the Base representation, not to optimized Q.

```
{json.dumps(rec,indent=2)}
```

## Numerical results

Primary FV is the mean over the same nine severity-1 model scenarios; sample std reflects scenario variability. Rotation/noise use three realizations; translation replicates are deterministic duplicates. Full severity0/0.5/1/1.5/2 outcomes are in future_per_model.csv. External severity is distinct from r. All test grid rows are predeclared diagnostics; only calibration determines the selected point.

```
{compact.to_string(index=False)}
```

Calibration: {int(cal.feasible.sum())}/{len(cal)} final iterates feasible, {int(cal.converged.sum())}/{len(cal)} meet the implemented stopping condition. Test: {int(s[s.beta.notna()].feasible.sum())}/12 feasible, {int(s[s.beta.notna()].converged.sum())}/12 meet the stopping condition. Failed rows remain recorded and are not deployed or repaired. The stopping condition is exact-geometry feasibility plus small successive objective change, as in Algorithm1; it is not a global-optimality or stationarity certificate. Final objective-gradient norms and full traces are saved. A pure objective gradient can be nonzero at a constrained optimum; do not interpret it as a KKT residual.

The raw surrogate values at different radii are different objectives and should not be directly ranked. `common_radius_risk.csv` recomputes nominal + 0.5*sensitivity for every saved output, using a common reference radius, to distinguish reduced local sensitivity from the threshold-FV/cost selection rule. This diagnostic does not alter selection.

## Interpretation and remaining solver limitation

The representation fix and smooth objective resolve the earlier collapse: all feasible test configurations retain100% current validity and reach100% Future Validity at severity1. The selected r=0 result improves severity1 FV from Base96.01% to100%, with a small cost increase from5.5755 to5.6046. This selected configuration is a nominal-only refinement, not evidence that the sensitivity term caused its gain.

Positive radii provide additional robustness in the predeclared stronger-shift diagnostics: at severity2, r=0 obtains94.62%, r=0.25 obtains99.83%, and r=0.5 obtains100% FV (beta1.05, representatives chosen on calibration). Costs are5.6046,5.6344,5.6540. Local sensitivity falls from0.02985 to0.001746 and0.000890. This is evidence of a useful sensitivity penalty in this pilot, with a modest cost tradeoff; it is not a certified robustness guarantee or a new test-selected operating point.

Why does calibration still choose zero? All nine feasible beta>=1.05 configurations tie at100% severity1 calibration FV; lower cost selects r=0, then lower beta selects1.05. The calibration criterion cannot distinguish the extra robustness once its threshold score saturates. For a future experiment, predeclare a broader calibration severity range with independent future realizations before evaluation, while retaining r=0 as the nominal control. Do not retroactively change this run's selected radius using its stronger test outcomes.

Numerical progress is real but incomplete at the tightest budget: all nine beta>1 test configurations meet the declared stopping condition, while all three beta1 test configurations fail exact geometry after800 iterations. Calibration has ten feasible final iterates, nine meeting the stopping condition. These beta1 failures are excluded from deployment. Small objective change is not a KKT or global convergence certificate.

The generator replay's tiny test-image difference leaves primary severity1 FV unchanged but changes Base severity2 FV from historical69.79% to replayed69.62%; both are reported rather than silently treating the images as identical.

## Fixed protocol and validation

The update operator, classifier, AE, training/factual/target IDs and all calibration/test future model checkpoints are unchanged from the prior pilot. Updates are the complete prescribed T=3 full-batch SGD steps (lr0.01) on a fixed balanced256-image empirical distribution, all classifier parameters trainable, frozen BN statistics, disabled dropout. This is not full-epoch MNIST retraining. Future checkpoints are reused read-only and hash-checked; no model retraining occurred. Previous final outcomes were already inspected, so this is explicitly retrospective. The saved future-model `hinge` column is the conventional hard threshold-margin diagnostic inherited from the scorer, not the smooth training objective; nominal/sensitivity/surrogate columns use the smooth loss.

Optimization and W2 are in the64-dimensional decoder latent domain. Perturbations underlying data-side sensitivity are in training-image pixel space. Dual norm is sqrt(N)*||dF/dh||_2 for the empirical RMS norm, not the unscaled tensor norm. Q has64 uniform support points and exact one-to-one Base→Q assignment. The augmented-Lagrangian settings remain primal step20 and rho1; maximum iterations increases to800 to test the previous iteration-limit problem. Exact unregularized OT rechecks each final geometry. Retained derivative graphs are released before final scalar metrics, fixing the earlier memory-cap error without changing objective values.

Fourteen focused tests cover returned-code preservation, smooth and hard objective gradients, dual scaling, geometry, and assignment. Actual CNN+decoder directional finite differences are recorded in base_recovery.json. Independent reporting verifies all24 permutations, unchanged Q under assignment, exact geometry totals and all source hashes. All8 GPUs are verified in worker logs; each worker caps its allocator at6.5% of device memory. Frozen inputs and existing runs remain untouched.

## Artifacts and reproduction

Task: tasks/012_mnist_latent_base_fix.md. Config: experiments/configs/mnist_latent_base_fix.json. Raw run: `{run}`. Code: experiments/src/lift_ce/experiments/mnist_latent_base_fix.py; shared objective counterfactuals/update_aware.py; generator code bookkeeping counterfactuals/vision.py. Summary CSVs and provenance live here; model checkpoints remain at their original external paths. Figures: figures/mnist_smooth_hinge/comparison.pdf and qualitative.pdf (first16 fixed factuals, including poor visual cases).

```
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/056_run_mnist_latent_base_fix.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_fix_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/057_report_mnist_smooth.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/NEW_RUN_ID
```

Use a new raw path and a separate checkout for report reproduction, since the reporter refuses to overwrite its existing report. Remaining limits include a single old classifier/AE,64 factuals, short model updates, nonsmooth CNN activations despite smooth hinge, the first-order/local nature of the uncertainty surrogate, and absence of independent semantic digit labels for CF outputs. No claim of certified worst-case validity or global optimization is justified.
'''
 (res/'report.md').write_text(report);print(compact.to_string(index=False));print('SELECTED',sel['r'],sel['beta'],tie_explanation)
if __name__=='__main__':main()
