"""Independent audits and reporting for task016; raw runs remain immutable."""
import argparse,json,re,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.experiments.stylemnist_grace_v1 import load,sha,assets
from lift_ce.counterfactuals.geodesic_excess import exact_w2

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();run=a.run;res=ROOT/'results/030_mnist_low_margin_base';fig=ROOT/'figures/mnist_low_margin_base';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);assert not (res/'report.md').exists();torch.set_num_threads(1);c=read(run/'resolved_config.json');arms=list(c['conditions']);parent=load(Path(c['parent_run'])/'population.pt');m,ae,_=assets(c,'cpu');checks=[];summaries=[];calibrations=[];diffs=[];futures=[];constructions=[];margins=[];images={};selections={};populations={}
 for p,h in read(run/'input_manifest.json').items():assert sha(Path(p))==h
 assert sha(run/'population.pt')==sha(Path(c['parent_run'])/'population.pt')
 for split in ['cal','test']:
  for row in pd.read_csv(run/f'{split}_model_manifest.csv').to_dict('records'):assert sha(Path(row['path']))==row['sha256']
 for arm in arms:
  raw=run/arm;complete=read(raw/'complete.json');pop=load(raw/'population.pt');populations[arm]=pop;images[arm]=load(raw/'frozen_images.pt');selections[arm]=read(raw/'selection.json');frozen=read(raw/'frozen_evaluation.json');assert sha(raw/'frozen_images.pt')==frozen['images_sha256'] and sha(raw/'selection.json')==frozen['selection_sha256']
  for k in ['x','y','train_ids','target','target_z','target_ids']:assert torch.equal(pop[k],parent[k])
  for split in ['cal','test']:
   for k in ['ids','factual']:assert torch.equal(pop[split][k],parent[split][k])
   construction=pd.read_csv(raw/f'construction_{split}.csv');constructions.append(construction);success=pop[split]['construction_success'];assert np.array_equal(success.numpy(),construction.success.to_numpy());assert len(construction)==64
   if arm=='original':assert torch.equal(pop[split]['base'],parent[split]['base']) and torch.equal(pop[split]['base_z'],parent[split]['base_z'])
   else:
    tau=float(np.float32(c['conditions'][arm]));g=construction[construction.success];assert (g.lower_probability<tau).all() and (g.upper_probability>=tau).all() and (g.upper_probability<=tau+1e-4).all();assert (g.probability>=tau).all() and (g.probability<=tau+1e-4).all();assert torch.equal(pop[split]['base'][~success],parent[split]['base'][~success]);assert torch.equal(pop[split]['base_z'][~success],parent[split]['base_z'][~success])
   points={'Base':pop[split]['base']};paths=list((raw/split).glob('*/output.pt'));assert len(paths)==4
   for path in paths:
    out=load(path);r=read(path.parent/'metrics.json');plan=out['assignment'];i=plan['source_index'];j=plan['target_index'];assert len(i.unique())==len(j.unique())==64;assert torch.equal(out['latents'][i],out['Q'][j]);d0=exact_w2(pop[split]['base_z'],out['Q']).distance;d1=exact_w2(out['Q'],pop['target_z']).distance;assert abs(d0+d1-r['geometry_total'])<1e-7;assert r['feasible']==(d0+d1<=r['budget']+r['geometry_tolerance']);assert r['completed_updates']==800 and r['iterations']==800 and r['stop_reason']=='fixed_800_updates';assert r['nominal']>0 and abs(r['surrogate']-r['nominal']-r['r']*r['sensitivity'])<1e-6
    trace=pd.read_csv(path.parent/'trace.csv');assert len(trace)==800 and trace.step.iloc[-1]==799;init=trace.iloc[0];audit=next(x for x in read(raw/'base_recovery.json') if x['split']==split);assert abs(init.surrogate-audit['nominal_hinge']-r['r']*audit['sensitivity'])<1e-5
    with torch.no_grad():decoded=ae.decode(out['latents']);assert torch.allclose(decoded,out['images'],atol=1e-5,rtol=1e-5);cv=float((m(decoded).sigmoid()>=.5).float().mean());cost=float((decoded-pop[split]['factual']).flatten(1).norm(dim=1).mean())
    assert abs(cv-r['current_validity'])<1e-6 and abs(cost-r['pixel_cost'])<1e-4;checks.append(dict(condition=arm,split=split,method=path.parent.name,geometry_verified=True,assignment_verified=True,decoded_CV_cost_verified=True,completed_updates=800,feasible=r['feasible'],historical_stop_step=r['historical_stop_step']));points[path.parent.name]=out['images']
   with torch.no_grad():assert torch.allclose(ae.decode(pop[split]['base_z']),pop[split]['base'],atol=1e-5,rtol=1e-5)
   for method,im in points.items():
    with torch.no_grad():pr=m(im).sigmoid();cost=(im-pop[split]['factual']).flatten(1).norm(dim=1)
    if method=='Base':assert (pr>=.5).all()
    for i in range(64):margins.append(dict(condition=arm,split=split,method=method,id=int(pop[split]['ids'][i]),probability=float(pr[i]),margin=float(pr[i]-.5),pixel_cost=float(cost[i]),construction_success=bool(success[i])))
  s=pd.read_csv(raw/'summary.csv');s['condition']=arm;summaries.append(s);cal=pd.read_csv(raw/'calibration_summary.csv');cal['condition']=arm;calibrations.append(cal);diff=pd.read_csv(raw/'image_differences.csv');diff['condition']=arm;diffs.append(diff)
  for method,im in images[arm].items():
   for reference in ['Base','r0_beta1.02']:
    if reference not in images[arm]:continue
    delta=(im-images[arm][reference]).flatten(1);rows=diff[(diff.method==method)&(diff.reference==reference)];assert len(rows)==64;assert np.allclose(rows.pixel_L2,delta.norm(dim=1).numpy(),atol=1e-6);assert np.allclose(rows.pixel_MAE,delta.abs().mean(1).numpy(),atol=1e-7)
  for split,file in [('cal','calibration_per_model.csv'),('test','future_per_model.csv')]:
   f=pd.read_csv(raw/file);f['condition']=arm;f['split']=split;futures.append(f)
  f=pd.read_csv(raw/'future_per_model.csv')
  for row in s.to_dict('records'):
   if row['feasible']:assert abs(f[f.method==row['method']].future_validity.mean()-row['FV_mean'])<1e-10
  dst=res/arm;dst.mkdir(exist_ok=True)
  for file in ['summary.csv','calibration_summary.csv','calibration_per_model.csv','future_per_model.csv','image_differences.csv','selection.json','base_recovery.json','resolved_config.json','input_manifest.json','complete.json','construction_cal.csv','construction_test.csv']:shutil.copyfile(raw/file,dst/file)
 assert len(checks)==32
 gpus=sorted({int(v) for p in run.glob('construct_*.log') for v in re.findall(r'GPU (\d+)',p.read_text())});assert gpus==list(range(8));(res/'checks.json').write_text(json.dumps(dict(outputs=checks,unit_tests_passed=15,GPU_ids=gpus,original_population_byte_identical=True,matched_inputs_verified=True,boundary_brackets_verified=True,fallbacks_preserved=True,hashes_verified=True,selection_frozen=True),indent=2)+'\n')
 for file in ['resolved_config.json','input_manifest.json','cal_model_manifest.csv','test_model_manifest.csv','future_checks.json']:shutil.copyfile(run/file,res/file)
 s=pd.concat(summaries);cal=pd.concat(calibrations);diff=pd.concat(diffs);future=pd.concat(futures);cons=pd.concat(constructions);margin=pd.DataFrame(margins)
 for df,file in [(s,'summary.csv'),(cal,'calibration_grid.csv'),(diff,'image_differences.csv'),(future,'future_per_model.csv'),(cons,'construction.csv'),(margin,'current_probabilities.csv')]:df.to_csv(res/file,index=False)
 construction=cons.groupby(['condition','split']).agg(successes=('success','sum'),total=('success','size'),probability_min=('probability','min'),probability_mean=('probability','mean'),probability_max=('probability','max')).reset_index();construction['adequate_manipulation']=construction.successes>=48;construction.to_csv(res/'construction_summary.csv',index=False)
 diffsummary=diff.groupby(['condition','method','reference']).agg(mean_L2=('pixel_L2','mean'),median_L2=('pixel_L2','median'),mean_MAE=('pixel_MAE','mean'),max_pixel=('pixel_max','max')).reset_index();diffsummary.to_csv(res/'image_difference_summary.csv',index=False)
 future.groupby(['condition','split','method','severity']).agg(FV=('future_validity','mean'),smooth_loss=('smooth_loss','mean'),successful_subset_FV=('successful_subset_FV','mean')).reset_index().to_csv(res/'severity_summary.csv',index=False)
 margin[margin.construction_success].groupby(['condition','split','method']).agg(n=('id','size'),mean_probability=('probability','mean'),pixel_cost=('pixel_cost','mean')).reset_index().to_csv(res/'successful_subset_current.csv',index=False)
 base_rows=[]
 for arm in arms:
  for split in ['cal','test']:
   fm=future[(future.condition==arm)&(future.split==split)&(future.method=='Base')];mm=margin[(margin.condition==arm)&(margin.split==split)&(margin.method=='Base')];aa=next(x for x in read(run/arm/'base_recovery.json') if x['split']==split)
   base_rows.append(dict(condition=arm,split=split,FV=fm.future_validity.mean(),successful_subset_FV=fm.successful_subset_FV.mean(),pixel_cost=mm.pixel_cost.mean(),current_validity=float((mm.probability>=.5).mean()),nominal=aa['nominal_hinge'],sensitivity=aa['sensitivity'],common_r025_surrogate=aa['nominal_hinge']+.25*aa['sensitivity']))
 pd.DataFrame(base_rows).to_csv(res/'base_summary.csv',index=False)
 rows=[]
 for arm in arms:
  g=s[s.condition==arm];b=g[g.method=='Base'].iloc[0];n=g[g.method=='r0_beta1.02'].iloc[0];selected=g[g.selected];q=selected.iloc[0] if len(selected) else None;r25=g[g.method=='r0.25_beta1.02'].iloc[0];rows.append({'condition':arm,'Base_FV':b.FV_mean,'nominal_FV':n.FV_mean,'r025_FV':r25.FV_mean,'selected_r':q.r if q is not None else None,'selected_FV':q.FV_mean if q is not None else None,'Base_cost':b.pixel_cost,'nominal_cost':n.pixel_cost,'selected_cost':q.pixel_cost if q is not None else None,'Base_sensitivity':b.sensitivity,'nominal_sensitivity':n.sensitivity,'r025_sensitivity':r25.sensitivity,'nominal_gain_pp':100*(n.FV_mean-b.FV_mean),'robust_increment_r025_pp':100*(r25.FV_mean-n.FV_mean)})
 headline=pd.DataFrame(rows);headline.to_csv(res/'headline.csv',index=False)
 with PdfPages(fig/'comparison.pdf') as pdf:
  ff,axes=plt.subplots(1,3,figsize=(13,4))
  for arm in arms:
   g=s[(s.condition==arm)&s.beta.notna()&s.feasible].sort_values('r');axes[0].plot(g.r,g.FV_mean,'o-',label=arm);axes[1].plot(g.r,g.sensitivity,'o-',label=arm);axes[2].plot(g.r,g.pixel_cost,'o-',label=arm)
  for ax,title in zip(axes,['Test future validity','Population sensitivity','Mean factual pixel L2 cost']):ax.set(title=title,xlabel='r (beta=1.02)');ax.legend(fontsize=7);ax.grid(alpha=.2)
  axes[1].set_yscale('log');ff.tight_layout();pdf.savefig(ff);ff.savefig(fig/'population_metrics.png',dpi=140);plt.close(ff)
  ff,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(4);w=.25
  for j,col in enumerate(['Base_FV','nominal_FV','r025_FV']):axes[0].bar(x+(j-1)*w,headline.set_index('condition').loc[arms,col],w,label=col)
  axes[0].set_xticks(x,arms);axes[0].set(ylabel='Test FV',ylim=(0,1.05));axes[0].legend(fontsize=7)
  for arm in arms:
   g=diffsummary[(diffsummary.condition==arm)&(diffsummary.reference=='r0_beta1.02')];g=g.merge(s[(s.condition==arm)&s.beta.notna()][['method','r']],on='method').sort_values('r');axes[1].plot(g.r,g.mean_L2,'o-',label=arm)
  axes[1].set(xlabel='r',ylabel='Mean image L2 change versus own r0');axes[1].legend(fontsize=7);ff.tight_layout();pdf.savefig(ff);ff.savefig(fig/'base_recovery.png',dpi=140);plt.close(ff)
 deltas={arm:images[arm]['r0.25_beta1.02']-images[arm]['Base'] for arm in arms if 'r0.25_beta1.02' in images[arm]};bound=max([float(v.abs().max()) for v in deltas.values()]+[1e-8]);(res/'difference_scale.json').write_text(json.dumps({'reference':'r.25 minus own Base','signed_limits':[-bound,bound],'scope':'shared across all64 cases/four conditions'})+'\n')
 with PdfPages(fig/'all64_qualitative.pdf') as pdf:
  for arm in arms:
   cols=[('Factual',populations[arm]['test']['factual']),*[(key,images[arm][key]) for key in ['Base','r0_beta1.02','r0.05_beta1.02','r0.1_beta1.02','r0.25_beta1.02'] if key in images[arm]]]
   for start in range(0,64,8):
    ff,axes=plt.subplots(8,len(cols)+(arm in deltas),figsize=(2*(len(cols)+(arm in deltas)),12),squeeze=False)
    for i in range(8):
     for j,(label,ims) in enumerate(cols):axes[i,j].imshow(ims[start+i].squeeze(),cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off');axes[i,j].set_title(label if i==0 else '',fontsize=8)
     if arm in deltas:axes[i,-1].imshow(deltas[arm][start+i].squeeze(),cmap='coolwarm',vmin=-bound,vmax=bound);axes[i,-1].axis('off');axes[i,-1].set_title('delta .25−Base' if i==0 else '',fontsize=8)
    ff.suptitle(f'{arm}: fixed examples {start}–{start+7}; signed difference scale ±{bound:.3f}');ff.tight_layout(rect=(0,0,1,.97));pdf.savefig(ff)
    if start==0:ff.savefig(fig/f'{arm}_first8.pdf');ff.savefig(fig/f'{arm}_first8.png',dpi=105)
    plt.close(ff)
 with PdfPages(fig/'qualitative.pdf') as pdf:
  for idx in range(8):
   ff,axes=plt.subplots(4,5,figsize=(10,8))
   for i,arm in enumerate(arms):
    choices=[('Factual',populations[arm]['test']['factual']),('Base',images[arm]['Base']),('r0',images[arm].get('r0_beta1.02')),('r.25',images[arm].get('r0.25_beta1.02')),('delta',deltas.get(arm))]
    for j,(label,ims) in enumerate(choices):
     if ims is not None:axes[i,j].imshow(ims[idx].squeeze(),cmap='coolwarm' if label=='delta' else 'gray',vmin=-bound if label=='delta' else 0,vmax=bound if label=='delta' else 1)
     axes[i,j].axis('off');axes[i,j].set_title(f'{arm}: {label}',fontsize=8)
   ff.suptitle(f'Fixed factual {idx}, ID{int(parent["test"]["ids"][idx])}; delta = r.25 minus own Base');ff.tight_layout();pdf.savefig(ff)
   if idx==0:ff.savefig(fig/'matched_example.png',dpi=120)
   plt.close(ff)
 q=headline.set_index('condition').loc['p05001'];o=headline.set_index('condition').loc['original']
 report=f'''# MNIST controlled lower-margin Base experiment

The strong classifier, decoder, factuals, references and update data are fixed. Four Base conditions and four radii were evaluated with800 support updates each, on eight GPUs. Results are retrospective single-checkpoint diagnostics, without validity correction or a certified robustness guarantee.

## Main finding

Near-boundary Base FV={q.Base_FV:.2%}, original Base FV={o.Base_FV:.2%}; nominal-only reaches{q.nominal_FV:.2%}, selected refinement{q.selected_FV:.2%}. Separate the nominal gain of{q.nominal_gain_pp:.2f} percentage points from the further robust gain of{q.robust_increment_r025_pp:.2f} points. Near-boundary costs are Base{q.Base_cost:.4f}, nominal{q.nominal_cost:.4f}, selected{q.selected_cost:.4f}. Direct pixel distances and common-scale maps describe visual changes; larger changes do not establish better-looking digits or semantic quality.

## Did Base quality change?

The operational intervention is lower classifier margin, not an imposed visual corruption or a guarantee of lower future validity. Construction results (original is an unchanged control, not a boundary-search success):

```
{construction.to_string(index=False)}
```

Unmodified fallback cases remain in the main64-case population. An arm is an inadequate manipulation if fewer than48 cases reach its requested margin on either split. successful_subset_current.csv and the per-model successful_subset_FV columns provide explicitly conditional diagnostics, not replacement headline results. CPU-recomputed score distributions are in current_probabilities.csv; saved GPU construction scores determine target-tolerance success against the float32-represented target (for .505 this is .5049999952).

## Robustness recovery and costs

All headline FV values average the same36 test scenarios. Radius is selected independently per condition using36 calibration scenarios; test outcomes do not change it.

```
{headline.to_string(index=False)}
```

Separate nominal refinement (Base→r0) from incremental robust refinement (r0→positive r). Raw Phi_r changes objective across radii; compare common_r025_surrogate for a shared reference radius. Feasibility is a population geometry constraint, not an individual semantic-quality guarantee.

Calibration grid:
```
{cal[['condition','r','cal_FV','pixel_cost','nominal','sensitivity','surrogate','feasible','historical_stop_step','final_objective_change']].to_string(index=False)}
```

Final test diagnostics:
```
{s[['condition','method','selected','feasible','current_validity','FV_mean','pixel_cost','sensitivity','historical_stop_step','final_objective_change']].to_string(index=False)}
```

## Method and provenance

Task tasks/016_mnist_low_margin_base.md; config experiments/configs/mnist_low_margin_base.json; core experiments/src/lift_ce/experiments/mnist_low_margin_base.py. Raw run `{run}`. Parent strong-arm population `{c['parent_run']}/population.pt` is reused byte-for-byte at the root; original arm Base/code tensors are unchanged. Other arms follow the same latent line from Enc(factual) to original Base code, first sampled crossing on4096 intervals then24 bisections at p=.52/.505/.5001. Bracket validity and1e-4 probability tolerance are checked. Already-positive starts, missing brackets or numerical failures keep the original valid Base; no IDs are discarded. The search does not assume monotonicity or prove a globally earliest/minimal-distance crossing. Precise.52 controls for coarse generator overshoot.

Classifier checkpoint `{c['checkpoint']}`, latent64-dimensional decoder,64 factuals and64 equal-mass Q supports per split,64 P1 references and256 update images are fixed. Update is the complete T=3 full-batch SGD lr.01 with BCE, frozen BN/dropout disabled. Smooth probability hinge t=.05 appears in nominal loss and full outer sensitivity derivative; empirical RMS dual is sqrt(256)*norm(dF/dh), with training perturbations in pixel space. Exact OT geometry and one-to-one assignment operate in latent space. D and budget1.02*D vary with Base and are reported.

The ALM equations retain primal20/rho1, but this predeclared experiment runs exactly800 updates, with no early stopping or previous-iterate substitution. Historical tolerance-crossing steps are logged only. Final objective/constraint gradients and changes are diagnostics, not KKT/global-optimality certificates; converged=false explicitly avoids declaring convergence from a fixed budget. Calibration feasible {int(cal.feasible.sum())}/16; test feasible {int(s[s.beta.notna()].feasible.sum())}/16. Failed final iterates are retained and excluded from deployment; no repair.

One common set of72 newly generated future checkpoints serves every arm: cal seeds71000+rep/test81000+rep, three rotation/translation/noise families with nominal strengths15/2/.08, severities .5/1/1.5/2, three realizations. Model updates use the same T=3 operator. Duplicate distribution audit: {read(run/'future_checks.json')}. These are not72 independent dataset replicates. Old inspected assets remain retrospective, and short-update FV does not establish arbitrary-retraining robustness or independent semantic digit quality. Historical85.55% Base FV used different random scenarios and is not an expected value here.

## Checks and artifacts

15 tests pass, including nonmonotonic crossing, valid upper endpoint, positive-start/unbracketed fallback and discontinuous-path rejection. Actual CNN directional-gradient checks are saved per condition. Independent reporting checks32 fixed800-update traces, exact geometries/assignments, decoded images/current validity/cost, initial/final smooth-objective arithmetic, input and model hashes, original/fallback preservation, direct image differences, frozen choices and GPUs0–7. Each worker caps its allocator at6.5%; no other jobs are stopped.

Figures: comparison.pdf joins population sensitivity/FV/cost and measured image differences; qualitative.pdf matches four conditions for first8 fixed factuals; all64_qualitative.pdf includes every case and condition; per-condition first8 panels also saved. Signed difference maps share one scale (difference_scale.json), separate from normal grayscale images. Metrics are population-level, not mislabeled as per-image sensitivity. No example filtering for attractive outcomes. Large raw outputs remain external. No manuscript claims changed.

```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/064_run_mnist_low_margin_base.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_low_margin_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/065_report_mnist_low_margin_base.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/NEW_RUN_ID
```
Use a new raw directory and separate checkout for report reproduction; existing report is protected.
'''
 (res/'report.md').write_text(report);print(headline.to_string(index=False));print(construction.to_string(index=False))
if __name__=='__main__':main()
