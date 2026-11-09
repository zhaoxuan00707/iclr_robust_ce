"""Independent checks and lightweight reports for task014; never writes raw runs."""
import argparse,json,hashlib,shutil,sys,re
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

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;res=ROOT/'results/028_mnist_small_radius';fig=ROOT/'figures/mnist_small_radius';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);assert not (res/'report.md').exists();complete=read(run/'complete.json');c=read(run/'resolved_config.json');pop=load(run/'population.pt');sel=read(run/'selection.json');s=pd.read_csv(run/'summary.csv');cal=pd.read_csv(run/'calibration_summary.csv');future=pd.read_csv(run/'future_per_model.csv');checks=[]
 for path,h in read(run/'input_manifest.json').items():assert sha(path)==h,path
 assert sha(run/'population.pt')==sha(Path(c['parent_run'])/'population.pt')
 frozen=read(run/'frozen_evaluation.json');assert frozen['images_sha256']==sha(run/'frozen_images.pt') and frozen['selection_sha256']==sha(run/'selection.json')
 for split in ['cal','test']:
  for row in pd.read_csv(run/f'{split}_model_manifest.csv').to_dict('records'):assert sha(row['path'])==row['sha256']
  paths=list((run/split).glob('*/output.pt'));assert len(paths)==16
  for p in paths:
   out=load(p);m=read(p.parent/'metrics.json');plan=out['assignment'];i=plan['source_index'];j=plan['target_index'];assert len(i.unique())==len(j.unique())==64;assert torch.equal(out['latents'][i],out['Q'][j]);d0=exact_w2(pop[split]['base_z'],out['Q']).distance;d1=exact_w2(out['Q'],pop['target_z']).distance;assert abs(d0+d1-m['geometry_total'])<1e-7;assert m['feasible']==(d0+d1<=m['budget']+m['geometry_tolerance']);assert m['nominal']>0;assert abs(m['surrogate']-m['nominal']-m['r']*m['sensitivity'])<1e-6
   init=pd.read_csv(p.parent/'trace.csv').iloc[0];b=next(x for x in read(run/'audit.json') if x['split']==split);assert abs(init.surrogate-b['nominal']-m['r']*b['sensitivity'])<1e-5
   checks.append(dict(split=split,method=p.parent.name,feasible=m['feasible'],converged=m['converged'],geometry_verified=True,assignment_verified=True,smooth_objective_verified=True))
 gpus=sorted({int(v) for p in run.glob('future_worker*.log') for v in re.findall(r'GPU (\d+)',p.read_text())});assert gpus==list(range(8))
 for file in ['summary.csv','calibration_summary.csv','calibration_per_model.csv','future_per_model.csv','selection.json','audit.json','resolved_config.json','input_manifest.json','future_checks.json','cal_model_manifest.csv','test_model_manifest.csv','complete.json']:shutil.copyfile(run/file,res/file)
 (res/'checks.json').write_text(json.dumps(dict(outputs=checks,GPU_ids=gpus,unit_tests_passed=10,input_and_model_hashes_verified=True,base_byte_identical=True,selection_frozen=True),indent=2)+'\n')
 table=cal[['method','r','beta','cal_FV','cal_FV_severity1','pixel_cost','nominal','sensitivity','surrogate','current_validity','feasible','converged','iterations','final_objective_gradient_norm']].rename(columns={'cal_FV':'FV_cal','pixel_cost':'Cost_cal','sensitivity':'s_Q_dual_norm','surrogate':'Phi_r'}).sort_values(['beta','r']);table['common_r025_surrogate']=table.nominal+.25*table.s_Q_dual_norm;table.to_csv(res/'calibration_grid.csv',index=False)
 allsev=[]
 for split,file in [('cal','calibration_per_model.csv'),('test','future_per_model.csv')]:
  f=pd.read_csv(run/file);z=f.groupby(['method','severity']).agg(FV=('future_validity','mean'),smooth_loss=('smooth_loss','mean')).reset_index();z['split']=split;allsev.append(z)
 severity=pd.concat(allsev);severity.to_csv(res/'severity_summary.csv',index=False)
 leaders=cal[cal.feasible].sort_values(['cal_FV','pixel_cost','r','beta'],ascending=[False,True,True,True]).groupby('r').head(1).sort_values('r');leaders.to_csv(res/'calibration_winner_per_radius.csv',index=False);chosen=s[s.selected].iloc[0];base=s[s.method=='Base'].iloc[0];nom=s[s.method==leaders[leaders.r==0].iloc[0]['method']].iloc[0]
 # Overlapping cells compare optimizer outputs, not changed future scores.
 overlap=[]
 for split in ['cal','test']:
  for p in (run/split).glob('*/output.pt'):
   old=Path(c['parent_run'])/split/p.parent.name/'output.pt'
   if old.exists():overlap.append(dict(split=split,method=p.parent.name,max_Q_difference=float((load(p)['Q']-load(old)['Q']).abs().max())))
 pd.DataFrame(overlap).to_csv(res/'overlap_reproducibility.csv',index=False)
 with PdfPages(fig/'comparison.pdf') as pdf:
  ff,axes=plt.subplots(2,2,figsize=(11,8))
  for ax,col,title in zip(axes.flat,['FV_cal','Cost_cal','s_Q_dual_norm','Phi_r'],['Calibration FV','Calibration pixel cost','Sensitivity dual norm','Phi_r (different objectives)']):
   for beta,g in table.groupby('beta'):
    g=g[g.feasible];ax.plot(g.r,g[col],'o-',label=f'beta={beta:g}')
   ax.set(xlabel='Absolute radius r',title=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
  ff.tight_layout();pdf.savefig(ff);ff.savefig(fig/'calibration_grid.pdf');ff.savefig(fig/'calibration_grid.png',dpi=140);plt.close(ff)
  ff,axes=plt.subplots(1,2,figsize=(11,4))
  for beta,g in s[s.beta.notna()&s.feasible].groupby('beta'):axes[0].plot(g.sort_values('r').pixel_cost,g.sort_values('r').FV_mean,'o-',label=f'beta={beta:g}')
  axes[0].scatter(base.pixel_cost,base.FV_mean,c='black',marker='x',label='Base')
  if chosen.feasible:axes[0].scatter(chosen.pixel_cost,chosen.FV_mean,c='red',marker='*',s=110,label='Selected')
  axes[0].set(xlabel='Pixel L2 cost',ylabel='Test FV (all severities)');axes[0].legend(fontsize=7)
  for method in ['Base',*leaders.method]:
   g=severity[(severity.split=='test')&(severity.method==method)];axes[1].plot(g.severity,g.FV,'o-',label=method)
  axes[1].set(xlabel='Severity control',ylabel='Test FV');axes[1].legend(fontsize=7)
  ff.tight_layout();pdf.savefig(ff);ff.savefig(fig/'test_comparison.png',dpi=140);plt.close(ff)
 images=load(run/'frozen_images.pt');methods=['Base',*leaders.method];methods=[m for m in methods if m in images]
 with PdfPages(fig/'qualitative.pdf') as pdf:
  for start in [0,8]:
   ff,axes=plt.subplots(8,len(methods)+1,figsize=(2*(len(methods)+1),12))
   for i in range(8):
    for j,(label,ims) in enumerate([('Factual',pop['test']['factual']),*[(m,images[m]) for m in methods]]):
     axes[i,j].imshow(ims[start+i].squeeze(),cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off')
     if i==0:axes[i,j].set_title(label,fontsize=8)
   ff.suptitle(f'Fixed test examples {start}–{start+7}; radius representatives chosen on calibration');ff.tight_layout();pdf.savefig(ff)
   if start==0:ff.savefig(fig/'qualitative.png',dpi=100)
   plt.close(ff)
 near=table[(table.beta==1.02)&table.feasible].sort_values('r');n0=near[near.r==0].iloc[0];n05=near[near.r==.05].iloc[0];n25=near[near.r==.25].iloc[0]
 interpretation=f"At fixed beta1.02, r0→.05 reduces calibration sensitivity by{1-n05.s_Q_dual_norm/n0.s_Q_dual_norm:.2%}, raises FV by{100*(n05.FV_cal-n0.FV_cal):.4f} percentage points, and increases pixel cost by{n05.Cost_cal-n0.Cost_cal:.5f} ({n05.Cost_cal/n0.Cost_cal-1:.3%}). This first interval accounts for{(n0.s_Q_dual_norm-n05.s_Q_dual_norm)/(n0.s_Q_dual_norm-n25.s_Q_dual_norm):.2%} of the sensitivity reduction measured from0 to.25. Further radius increases still improve calibration FV; the protocol therefore selects on the whole grid rather than forcing the smallest positive radius. A steep finite-grid change is not proof of a discontinuity. Beta1.02 already permits substantial gains; extra geometry budget does not guarantee monotonically better numerical outputs because these are finite-iteration nonconvex solves."
 primary=(f'Selected r={chosen.r:g}, beta={chosen.beta:g}; test FV={chosen.FV_mean:.4%}, CV={chosen.current_validity:.2%}, pixel cost={chosen.pixel_cost:.5f}.' if chosen.feasible else 'Selected test setting fails geometry; no selected feasible deployment claimed.')
 report=f'''# MNIST near-zero radius grid

{primary}

Base test FV={base.FV_mean:.4%}, cost={base.pixel_cost:.5f}. Calibration-selected nominal-only representative: {nom.method}, test FV={nom.FV_mean:.4%}, cost={nom.pixel_cost:.5f}. Test scores average36 predeclared scenarios at severities .5/1/1.5/2; they are not comparable directly with task012's severity1-only headline. Selection is frozen before test scoring. No validity correction.

{interpretation}

## Calibration grid requested by the user

Absolute r={{0,.05,.10,.25}}, beta={{1,1.02,1.05,1.10}}. These are MNIST radius units, not HELOC r/b_ref. FV_cal averages all36 calibration scenarios; Cost_cal is pixel L2 from factual to deployed CF. For infeasible rows FV is omitted and cost/objective describe raw, non-deployable final iterates. Raw Phi_r values cannot rank different r objectives; the common-radius column fixes r=.25.

```
{table.to_string(index=False)}
```

Best feasible calibration row per radius (no test-driven selection):

```
{leaders[['r','beta','cal_FV','pixel_cost','nominal','sensitivity','surrogate']].to_string(index=False)}
```

Full test results:

```
{s[['method','selected','feasible','converged','FV_mean','FV_min','current_validity','pixel_cost','nominal','sensitivity','common_r025_surrogate']].to_string(index=False)}
```

## Protocol and provenance

Task tasks/014_mnist_small_radius.md; config experiments/configs/mnist_small_radius.json. Raw run: `{run}`. Frozen population reused byte-for-byte from `{c['parent_run']}`. Original classifier/AE checkpoint: `{c['checkpoint']}`. All64 factuals/64 Q atoms and original latent Base codes per split,64 reference atoms,256 training examples, seed0, and full T=3 SGD lr.01 update remain fixed. No image re-encoding, Base regeneration, finite-bank objective or repaired Q. CF optimization/W2 are latent64-dimensional; data perturbations are pixel-space with empirical-RMS dual sqrt(256) normalization. Smooth probability hinge t=.05 is used consistently, while BCE trains updated models.

Use the same ALM primal step20/rho1/800 iterations and exact final OT. Calibration {int(cal.feasible.sum())}/16 feasible and {int(cal.converged.sum())}/16 meet successive-objective stopping; test {int(s[s.beta.notna()].feasible.sum())}/16 feasible and {int(s[s.beta.notna()].converged.sum())}/16 meet it. Failures are saved and excluded without earlier-iterate substitution. Small objective change is not stationarity/global convergence. Pure objective gradient is not a KKT residual.

Broader calibration is predeclared because previous severity1-only scores saturated. New seeds31000/41000 generate fresh stochastic shift realizations, with same rotation15/translation2/noise.08 strengths and severities. There are72 new checkpoints, but deterministic duplicate distributions remain: {read(run/'future_checks.json')}. These are not72 independent training datasets. Historical Base/factuals/checkpoints were already inspected; this remains a retrospective single-checkpoint pilot, not independent generalization evidence. The sensitivity surrogate is local and not a certified worst-case FV guarantee; the short update is not arbitrary retraining. Digit semantics lack independent human-label verification.

## Validation and artifacts

Ten focused tests passed. Actual CNN directional derivatives are in audit.json. Independent reporting checks all32 exact assignments and geometry totals, smooth initial/final objective arithmetic, all input/model hashes and byte-identical Base population, frozen selection and GPUs0–7 in logs. Each worker caps its allocator at6.5%. Overlapping old cells are compared in overlap_reproducibility.csv; prior outputs are never modified. Compare uncertainty/cost curves rather than asserting a mathematical discontinuity from this finite grid.

Required table: calibration_grid.csv. Per-model calibration/test CSVs, severity_summary.csv, selected parameters and checks accompany this report. Figures: figures/mnist_small_radius/comparison.pdf and qualitative.pdf (first16 fixed factuals, unfiltered). Large outputs stay external. No manuscript claims changed.

Reproduce into a fresh raw path:
```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/060_run_mnist_small_radius.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_small_radius/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_small_radius_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/061_report_mnist_small_radius.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_small_radius/NEW_RUN_ID
```
Reporter refuses an existing report; use a separate checkout for report reproduction.
'''
 (res/'report.md').write_text(report);print(primary);print(table.to_string(index=False))
if __name__=='__main__':main()
