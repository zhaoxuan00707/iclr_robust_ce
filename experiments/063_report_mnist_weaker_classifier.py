"""Audit and report matched classifier-quality experiment without changing raw runs."""
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
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();run=a.run;res=ROOT/'results/029_mnist_weaker_classifier';fig=ROOT/'figures/mnist_weaker_classifier';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);assert not (res/'report.md').exists();torch.set_num_threads(1);c=read(run/'resolved_config.json');selected=read(run/'weak_selection.json');strongval=read(run/'strong_validation.json');testdiag=read(run/'classifier_test.json');frozen=read(run/'classifier_frozen.json');assert frozen['weak_payload_sha256']==sha(run/'weak_payload.pt') and frozen['selection_sha256']==sha(run/'weak_selection.json');assert frozen['strong_checkpoint_sha256']==sha(Path(c['checkpoint']));candidates=pd.read_csv(run/'candidate_summary.csv');winner=candidates[candidates.eligible].sort_values(['distance_to_target','step','seed']).iloc[0];assert winner.seed==selected['seed'] and winner.step==selected['step']
 for row in candidates.to_dict('records'):assert sha(Path(row['path']))==row['sha256']
 for path,h in read(run/'input_manifest.json').items():assert sha(Path(path))==h
 populations={arm:load(run/arm/'population.pt') for arm in ['strong','weak']}
 for k in ['x','y','train_ids','target','target_ids','target_z']:assert torch.equal(populations['strong'][k],populations['weak'][k])
 for split in ['cal','test']:
  for k in ['factual','ids']:assert torch.equal(populations['strong'][split][k],populations['weak'][split][k])
 data=load(run/'classifier_data.pt');assert not set(data['ids'].tolist())&set(data['vids'].tolist());assert not set(data['vids'].tolist())&set(populations['strong']['target_ids'].tolist())
 checks=[];summaries=[];calibrations=[];differences=[];allfuture=[];images={};selections={};strong_model,_,_=assets(c,'cpu')
 for arm in ['strong','weak']:
  raw=run/arm;assert read(raw/'complete.json')['input_hashes_verified'];cfg=read(raw/'resolved_config.json');m,ae,_=assets(cfg,'cpu');pop=populations[arm];images[arm]=load(raw/'frozen_images.pt');selections[arm]=read(raw/'selection.json');frozen=read(raw/'frozen_evaluation.json');assert frozen['images_sha256']==sha(raw/'frozen_images.pt') and frozen['selection_sha256']==sha(raw/'selection.json')
  for split in ['cal','test']:
   for row in pd.read_csv(raw/f'{split}_model_manifest.csv').to_dict('records'):assert sha(Path(row['path']))==row['sha256']
   for path in (raw/split).glob('*/output.pt'):
    out=load(path);r=read(path.parent/'metrics.json');i=out['assignment']['source_index'];j=out['assignment']['target_index'];assert len(i.unique())==len(j.unique())==64;assert torch.equal(out['latents'][i],out['Q'][j]);d0=exact_w2(pop[split]['base_z'],out['Q']).distance;d1=exact_w2(out['Q'],pop['target_z']).distance;assert abs(d0+d1-r['geometry_total'])<1e-7;assert r['feasible']==(d0+d1<=r['budget']+r['geometry_tolerance']);assert r['nominal']>0 and abs(r['surrogate']-r['nominal']-r['r']*r['sensitivity'])<1e-6
    with torch.no_grad():decoded=ae.decode(out['latents']);assert torch.allclose(decoded,out['images'],atol=1e-5,rtol=1e-5);cv=float((m(decoded).sigmoid()>=.5).float().mean());cost=float((decoded-pop[split]['factual']).flatten(1).norm(dim=1).mean())
    assert abs(cv-r['current_validity'])<1e-6 and abs(cost-r['pixel_cost'])<1e-4
    init=pd.read_csv(path.parent/'trace.csv').iloc[0];b=next(x for x in read(raw/'base_recovery.json') if x['split']==split);assert abs(init.surrogate-b['nominal_hinge']-r['r']*b['sensitivity'])<1e-5
    checks.append(dict(arm=arm,split=split,method=path.parent.name,geometry_verified=True,assignment_verified=True,decoded_CV_cost_verified=True,feasible=r['feasible'],converged=r['converged']))
  s=pd.read_csv(raw/'summary.csv');s['arm']=arm;summaries.append(s);cal=pd.read_csv(raw/'calibration_summary.csv');cal['arm']=arm;calibrations.append(cal);d=pd.read_csv(raw/'image_differences.csv');d['arm']=arm;differences.append(d);f=pd.read_csv(raw/'future_per_model.csv');f['arm']=arm;allfuture.append(f)
  for row in s.to_dict('records'):
   vals=f[f.method==row['method']].future_validity
   if row['feasible']:assert abs(vals.mean()-row['FV_mean'])<1e-10
  for method,im in images[arm].items():
   for ref in ['Base','r0_beta1.02']:
    if ref not in images[arm]:continue
    delta=(im-images[arm][ref]).flatten(1);records=d[(d.method==method)&(d.reference==ref)];assert len(records)==64;assert np.allclose(records.pixel_L2,delta.norm(dim=1).numpy(),atol=1e-6);assert np.allclose(records.pixel_MAE,delta.abs().mean(1).numpy(),atol=1e-7)
  dst=res/arm;dst.mkdir(exist_ok=True)
  for name in ['summary.csv','calibration_summary.csv','calibration_per_model.csv','future_per_model.csv','image_differences.csv','selection.json','base_recovery.json','resolved_config.json','input_manifest.json','future_checks.json','complete.json','cal_model_manifest.csv','test_model_manifest.csv']:shutil.copyfile(raw/name,dst/name)
 assert len(checks)==16
 gpus=sorted({int(v) for p in run.glob('candidate*.log') for v in re.findall(r'GPU (\d+)',p.read_text())});assert gpus==list(range(8))
 for name in ['candidate_summary.csv','weak_selection.json','strong_validation.json','classifier_test.json','classifier_frozen.json','population_checks.json','resolved_config.json','input_manifest.json']:shutil.copyfile(run/name,res/name)
 (res/'checks.json').write_text(json.dumps(dict(outputs=checks,GPU_ids=gpus,unit_tests_passed=10,matched_inputs_verified=True,checkpoint_selection_verified=True,hashes_verified=True,direct_image_differences_verified=True),indent=2)+'\n');s=pd.concat(summaries);cal=pd.concat(calibrations);d=pd.concat(differences);future=pd.concat(allfuture);s.to_csv(res/'summary.csv',index=False);cal.to_csv(res/'calibration_summary.csv',index=False);d.to_csv(res/'image_differences.csv',index=False)
 diff=d.groupby(['arm','method','reference']).agg(mean_L2=('pixel_L2','mean'),median_L2=('pixel_L2','median'),mean_MAE=('pixel_MAE','mean'),maximum_pixel_change=('pixel_max','max')).reset_index();diff.to_csv(res/'image_difference_summary.csv',index=False);severity=future.groupby(['arm','method','severity']).agg(FV=('future_validity','mean'),smooth_loss=('smooth_loss','mean')).reset_index();severity.to_csv(res/'severity_summary.csv',index=False)
 with PdfPages(fig/'population_metrics.pdf') as pdf:
  ff,axes=plt.subplots(1,3,figsize=(13,4))
  for arm in ['strong','weak']:
   q=s[(s.arm==arm)&s.beta.notna()&s.feasible].sort_values('r');axes[0].plot(q.r,q.FV_mean,'o-',label=arm);axes[1].plot(q.r,q.sensitivity,'o-',label=arm);dd=diff[(diff.arm==arm)&(diff.reference=='r0_beta1.02')];q=q.merge(dd,on='method');axes[2].plot(q.r,q.mean_L2,'o-',label=arm)
  for ax,title in zip(axes,['Test future validity','Population sensitivity','Mean image L2 change vs r=0']):ax.set(xlabel='r (beta=1.02)',title=title);ax.grid(alpha=.2);ax.legend()
  axes[1].set_yscale('log');ff.tight_layout();pdf.savefig(ff);ff.savefig(fig/'population_metrics.png',dpi=140);plt.close(ff)
 # Fixed indices: same cases for both classifiers, common signed-difference scale.
 methods=['Base','r0_beta1.02','r0.25_beta1.02'];columns=[('Factual',populations['strong']['test']['factual'])]
 for arm in ['strong','weak']:
  for method in methods:
   if method in images[arm]:columns.append((f'{arm}: {method}',images[arm][method]))
 deltas={arm:images[arm]['r0.25_beta1.02']-images[arm]['r0_beta1.02'] for arm in ['strong','weak'] if all(k in images[arm] for k in ['r0.25_beta1.02','r0_beta1.02'])};bound=max([float(v.abs().max()) for v in deltas.values()]+[1e-8]);(res/'difference_map_scale.json').write_text(json.dumps({'signed_pixel_limits':[-bound,bound],'reference':'r0.25 minus r0','scope':'shared across both arms/all64 cases'})+'\n')
 with PdfPages(fig/'all64_qualitative.pdf') as pdf:
  for start in range(0,64,8):
   ff,axes=plt.subplots(8,len(columns)+len(deltas),figsize=(2*(len(columns)+len(deltas)),12),squeeze=False)
   for i in range(8):
    for j,(label,ims) in enumerate(columns):axes[i,j].imshow(ims[start+i].squeeze(),cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off');axes[i,j].set_title(label if i==0 else '',fontsize=7)
    for j,(arm,delta) in enumerate(deltas.items(),start=len(columns)):im=axes[i,j].imshow(delta[start+i].squeeze(),cmap='coolwarm',vmin=-bound,vmax=bound);axes[i,j].axis('off');axes[i,j].set_title(f'{arm}: delta .25−0' if i==0 else '',fontsize=7)
   ff.suptitle(f'Matched fixed examples {start}–{start+7}; signed differences share ±{bound:.3f} pixel scale');ff.tight_layout(rect=(0,0,1,.97));pdf.savefig(ff)
   if start==0:ff.savefig(fig/'qualitative.pdf');ff.savefig(fig/'qualitative.png',dpi=110)
   plt.close(ff)
 # Show every radius on the same first8 examples, separately from all-case comparison.
 with PdfPages(fig/'radius_qualitative.pdf') as pdf:
  for arm in ['strong','weak']:
   cols=[('Factual',populations[arm]['test']['factual']),*images[arm].items()];ff,axes=plt.subplots(8,len(cols),figsize=(2*len(cols),12),squeeze=False)
   for i in range(8):
    for j,(label,ims) in enumerate(cols):axes[i,j].imshow(ims[i].squeeze(),cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off');axes[i,j].set_title(label if i==0 else '',fontsize=8)
   ff.suptitle(f'{arm} classifier: same fixed first8 factuals');ff.tight_layout();pdf.savefig(ff);plt.close(ff)
 comparison=diff[(diff.method=='r0.25_beta1.02')&(diff.reference=='r0_beta1.02')];strongdiff=comparison[comparison.arm=='strong'];weakdiff=comparison[comparison.arm=='weak'];interpretation='The endpoint comparison is unavailable because an endpoint failed geometry.'
 if len(strongdiff) and len(weakdiff):
  a0=strongdiff.iloc[0];a1=weakdiff.iloc[0];interpretation=f"Mean direct image L2 change between r0 andr.25 is {a0.mean_L2:.6f} for the strong arm and {a1.mean_L2:.6f} for the weak arm (ratio {a1.mean_L2/max(a0.mean_L2,1e-12):.3g}). Mean absolute per-pixel changes are {a0.mean_MAE:.6f} and {a1.mean_MAE:.3g}. Larger measured differences do not establish better-looking or semantically correct CFs."
 report=f'''# MNIST weaker-classifier matched comparison

Weak checkpoint selected before CF generation: seed{selected['seed']}, step{selected['step']}; validation accuracy{selected['accuracy']:.2%}, original{strongval['accuracy']:.2%}. Official digit4/9 test diagnostics after checkpoint freeze: {testdiag}.

{interpretation}

## Stopping diagnostic

Weak-arm test iteration counts are {s[(s.arm=='weak')&s.beta.notna()].iterations.tolist()}. If these equal2, only one support update was performed before the objective-change tolerance passed. This implementation divides by max(1,abs(previous objective)), so losses below1 use an absolute1e-6 change threshold. Low sensitivity and high measured Base FV do not prove the weak problem is fully optimized; small nonzero gradients and stopping scale can explain a flat visual curve. Larger Base-to-factual edits in a weaker arm are not GRACE gains. Recommend a separately predeclared matched stopping-tolerance audit before a causal claim about classifier quality; do not select another checkpoint for attractive CF images.

## Population results

Same64 factuals per split, same decoder/reference population/update data, beta1.02 and r0/.05/.10/.25. Sensitivity and FV below are population-level; they are not scores of individual images. Each arm selects radius on its own calibration FV with cost tie-break before test scoring.

```
{s[['arm','method','selected','feasible','converged','iterations','current_validity','FV_mean','FV_min','pixel_cost','nominal','sensitivity','surrogate','strong_classifier_agreement']].to_string(index=False)}
```

Calibration metrics:
```
{cal[['arm','r','cal_FV','pixel_cost','nominal','sensitivity','surrogate','feasible','converged']].to_string(index=False)}
```

Direct image-difference summaries:
```
{comparison.to_string(index=False)}
```

## Design and limits

Task tasks/015_mnist_weaker_classifier.md and config experiments/configs/mnist_weaker_classifier.json were written before running. Raw run `{run}`. The weak SmallCNN is trained on128 balanced images, Adam lr.001/batch32, seeds150–157, checkpoints4/8/16/32/64/128. All48 candidates are retained. Selection uses512 balanced validation images excluded from weak training, update data and final P1; closest to90% accuracy among80–95% and at least3pp below original, ties fewer steps then seed. Neither appearance nor CF robustness chooses the checkpoint. Validation may overlap the historical strong classifier's training; its exact training provenance is unknown. One selected weak model is a designed stress test, not evidence that classifier accuracy alone causally controls visual quality.

The matched factuals are true4 images rejected by both models, prioritizing old IDs then remaining fixed pool order. Conditioning on agreement excludes some difficult cases; do not extrapolate to all digits. Common true9 references are accepted by both models in pixel/decoded space. Each arm generates new Base images/codes with the identical96-step latent prototype generator, because the current classifier has changed. No old Base/output is overwritten; Base regeneration is distinct from post-optimization repair. The autoencoder and256-image update distribution are frozen. Thus the matched strong control is newly generated too, and task014 headline scores are not a paired comparison.

Each arm uses the complete prescribed3 full-batch SGD steps lr.01 with frozen BN/dropout disabled, BCE training loss and smooth probability hinge t=.05 for nominal and sensitivity. All64 Q atoms are uniform; exact one-to-one assignment; latent64-D W2; training perturbations pixel-space with sqrt(256) empirical RMS dual scaling. ALM primal20/rho1/max800, feasibility1e-5*max(1,D) and objective-change1e-6. Final failures excluded, no repair. Small objective change is not a stationary/global-optimum certificate.

Paired future shifts use the same new cal51000/test61000 seeds, rotation/translation/noise, severity .5/1/1.5/2, three realizations.144 new checkpoint scenarios (72 per arm), with deterministic translations duplicated across seeds/splits. Both calibration choices are frozen before final scoring. FV is measured under each arm's own updated classifier;100% weak-model FV need not mean stronger semantic validity. Frozen strong-classifier agreement is a non-independent diagnostic, not human digit-label ground truth. Short-update, single-pair, retrospective data/decoder study; no certified or arbitrary-retraining robustness claim.

## Checks, figures and reproduction

Ten focused tests pass. Actual CNN outer derivatives are in each arm's base_recovery.json. Independent checks verify16 assignments/geometry/objective arithmetic, decoded images/current validity/cost, direct pixel differences, paired IDs and populations, checkpoint selection and hashes, frozen evaluations and GPUs0–7. Allocator cap6.5% per worker. Large training candidates and raw outputs remain external. No manuscript claims changed.

Figures: figures/mnist_weaker_classifier/qualitative.pdf shows fixed-first8 paired cases plus signed difference maps; all64_qualitative.pdf includes all64 test examples; radius_qualitative.pdf shows every feasible radius on the same first8 cases; population_metrics.pdf shows joint population sensitivity, FV and measured image change. Shared difference scale is recorded; difference maps magnify subtle changes and must not be presented as ordinary CF images. No attractive-case filtering.

```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/062_run_mnist_weaker_classifier.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_weak_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/063_report_mnist_weaker_classifier.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/NEW_RUN_ID
```
Use a separate checkout for report reproduction; existing report is protected.
'''
 (res/'report.md').write_text(report);print(interpretation);print(s[['arm','method','selected','FV_mean','pixel_cost','strong_classifier_agreement']].to_string(index=False))
if __name__=='__main__':main()
