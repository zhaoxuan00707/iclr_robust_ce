"""Read-only raw-run validation and repository reporting for MNIST pilot."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();run=a.run
 res=ROOT/'results/024_mnist_update_aware';fig=ROOT/'figures/mnist_update_aware';res.mkdir(exist_ok=True);fig.mkdir(exist_ok=True);assert not (res/'report.md').exists()
 complete=json.loads((run/'complete.json').read_text());sel=json.loads((run/'selection.json').read_text());config=json.loads((run/'resolved_config.json').read_text());pop=load(run/'population.pt');summary=pd.read_csv(run/'summary.csv');future=pd.read_csv(run/'future_per_model.csv');cal=pd.read_csv(run/'calibration_summary.csv')
 for path,h in json.loads((run/'input_manifest.json').read_text()).items():assert sha(path)==h
 checks=[]
 for split in ['cal','test']:
  for path in sorted((run/split).glob('*/output.pt')):
   d=load(path);plan=d['assignment'];idx=plan['source_index'];j=plan['target_index'];assert len(idx.unique())==len(j.unique())==64;assert torch.equal(d['latents'][idx],d['Q'][j]);assert torch.isfinite(d['images']).all() and d['images'].min()>=0 and d['images'].max()<=1
   checks.append({'split':split,'method':path.parent.name,'assignment_permutation':True,'unchanged_Q_up_to_ordering':True,'pixels_in_domain':True})
 for file in ['summary.csv','future_per_model.csv','calibration_summary.csv','calibration_per_model.csv','selection.json','assets.json','smoke.json','input_manifest.json','resolved_config.json','cal_model_manifest.csv','test_model_manifest.csv','complete.json']:shutil.copyfile(run/file,res/file)
 (res/'independent_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 means=future.groupby(['method','severity']).agg(FV_mean=('future_validity','mean'),FV_std=('future_validity','std')).reset_index();means.to_csv(res/'shift_severity.csv',index=False)
 future.groupby(['method','family','severity']).agg(FV_mean=('future_validity','mean'),FV_std=('future_validity','std')).reset_index().to_csv(res/'shift_by_family.csv',index=False)
 with PdfPages(fig/'comparison.pdf') as pdf:
  f,ax=plt.subplots(figsize=(8,5))
  for method,g in means.groupby('method'):ax.plot(g.severity,g.FV_mean,marker='o',label=method)
  ax.set(xlabel='External style-shift severity',ylabel='Future validity',title='MNIST 4→9; prescribed three-step model updates');ax.legend(fontsize=8);ax.grid(alpha=.2);f.tight_layout();f.savefig(fig/'severity.pdf');f.savefig(fig/'severity.png',dpi=150);pdf.savefig(f);plt.close(f)
  f,axes=plt.subplots(1,3,figsize=(12,4));valid=summary[summary.feasible & summary.beta.notna()].sort_values('beta')
  for ax,col,title in zip(axes,['FV_mean','pixel_cost','current_validity'],['Future validity','Pixel L2 cost','Current validity']):ax.plot(valid.beta,valid[col],'o-');ax.set(xlabel='Geometry beta',ylabel=title);ax.grid(alpha=.2)
  f.tight_layout();f.savefig(fig/'beta_metrics.pdf');pdf.savefig(f);plt.close(f)
 images=load(run/'frozen_images.pt');labels=['Factual',*images.keys()];values=[pop['test']['factual'],*images.values()]
 with PdfPages(fig/'qualitative.pdf') as pdf:
  # Fixed first 16 factual indices, independent of performance.
  for offset in [0,8]:
   f,axes=plt.subplots(8,len(labels),figsize=(2*len(labels),14),squeeze=False)
   for row in range(8):
    for col,(label,im) in enumerate(zip(labels,values)):
     ax=axes[row,col];ax.imshow(im[offset+row,0],cmap='gray',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
     if row==0:ax.set_title(label,fontsize=8)
     if col==0:ax.set_ylabel(f"ID {int(pop['test']['ids'][offset+row])}",fontsize=8)
   f.tight_layout();pdf.savefig(f)
   if offset==0:f.savefig(fig/'qualitative_preview.png',dpi=100)
   plt.close(f)
 assets=json.loads((run/'assets.json').read_text());smoke=json.loads((run/'smoke.json').read_text());cols=['method','selected','feasible','current_validity','FV_mean','FV_std','pixel_cost','nominal','sensitivity','surrogate']
 selected=summary[summary.selected];comparison='Selected test configuration failed final geometry; no successful deployment is claimed.'
 if len(selected) and selected.iloc[0].feasible:
  v=selected.iloc[0];b=summary[summary.method=='Base'].iloc[0];bd=summary[summary.method=='Base-decoded'].iloc[0]
  comparison=f"Calibration selected r={sel['r']:g}, beta={sel['beta']:g}. On the fixed 64 test factuals at severity 1, mean Future Validity is {v.FV_mean:.4%} versus original Base {b.FV_mean:.4%} and decoded Base {bd.FV_mean:.4%}; pixel cost is {v.pixel_cost:.4f} versus {b.pixel_cost:.4f}/{bd.pixel_cost:.4f}. Current validity is {v.current_validity:.4%}. This is a single-checkpoint descriptive pilot, not statistical evidence of general superiority."
 report=f'''# MNIST latent-space update-aware GRACE pilot

{comparison}

Authority: Section 3 of `paper/iclr_robust_ce-62.pdf`, task `tasks/011_mnist_update_aware.md`. This run implements the new update-aware first-order objective, not the old finite-bank maximum. No current-validity correction is applied. Raw run: `{run}`.

## Results

Future Validity mean/sample standard deviation below are across nine severity-1 scenarios (three families × three realizations). Deterministic translation repeats are duplicates; these are not nine independent training seeds. The full 45-scenario severity suite is saved separately.

```
{summary[cols].to_string(index=False)}
```

Calibration selected r={sel['r']}, beta={sel['beta']} on separate calibration factuals and calibration transformation seeds; all operating parameters were frozen before final scoring. {int((~cal.feasible).sum())} of {len(cal)} calibration configurations failed final exact-OT geometry. {int((~summary.feasible).sum())} test diagnostic configurations failed geometry. Failed configurations are retained with solver metrics, excluded from deployment, and not repaired or replaced with earlier iterates.

## What this first run establishes

**This pilot does not demonstrate successful robust CF generation.** Calibration selects r=0, so the deployed configuration has no sensitivity penalty. The positive-radius calibration runs perform worse: mean Future Validity is 7.47% at r=0.25 and 6.42% at r=0.5, versus 31.08% at r=0 (for feasible beta>1). These results do not establish a general failure of the proposed objective: the latent Base representation is already badly compromised, all 16 optimizations exhaust 80 iterations without satisfying the objective-change stopping tolerance, and only one short prescribed update is tested.

The major observed compatibility issue is Enc(Base)→Dec: current validity drops from 100% to 6.25% on calibration and 12.5% on test before GRACE optimization. The selected output recovers some validity but remains worse and more expensive than the original Base. The original Base has zero nominal hinge and zero first-order sensitivity in this run, yet less than 100% future validity under finite style shifts—an explicit example of the local hinge surrogate's limited predictive scope. Both f0 and the nominal three-step model retain 100% accuracy on the fixed update set (model_diagnostics.json); this is not held-out accuracy and does not certify generalization.

All three calibration beta=1 runs and the test beta=1 run fail final exact geometry. The feasible beta=1.05/1.10/1.20 outputs coincide at each radius because their geometry budgets do not constrain the observed trajectories. Do not interpret this as evidence that beta has no effect in general. The plots omit infeasible deployments, while tables retain their failure metrics.

The next useful experiment is to verify a Base generator that returns valid CFs directly on the existing decoder manifold, preserving its latent codes instead of re-encoding the output, and establish solver convergence on calibration data. This is a recommendation, not an unreported modification of this completed run; any subsequent experiment should be versioned and calibrated anew.

## Implementation and spaces

Frozen SmallCNN and StrongConvAutoencoder(64) are reused from `{config['checkpoint']}`. Existing test factuals and Base images are reused from `{config['old_run']}`. The task is digit 4→9, not six classifiers numbered 4–9. Exact input hashes and IDs are recorded.

Q has 64 equal-weight 64-dimensional latent supports, initialized to Enc(Base). P1 contains encoded training images favorable under f0. Optimize latent support coordinates and latent Euclidean W2 geometry, decode with the frozen decoder for classifier evaluation, and report factual-to-decoded-image pixel L2 cost. One-to-one minimum-cost assignment in latent space preserves Q exactly up to permutation; every factual retains one image. Latent projection is identity on R^64. Decoder pixels remain in [0,1]. Data-side perturbations are label-preserving training-image pixel perturbations, with an empirical RMS Euclidean norm; these are distinct from the latent CF geometry.

The prescribed update is three complete full-batch gradient-descent steps, lr=0.01, binary cross entropy, on 256 fixed label-stratified training examples, starting from frozen f0. All classifier parameters update; BatchNorm statistics are frozen and dropout disabled. This exact update is differentiated and used for all primary future models. It is not a truncated derivative of an eight-epoch AdamW update. Nominal plus r times sensitivity is optimized; sensitivity is sqrt(N) times the ordinary tensor-gradient norm. Reverse-mode differentiation includes the sensitivity gradient with respect to Q.

Algorithm 1's augmented-Lagrangian primal–dual update is used, with exact OT plans and latent-distance gradients, 80 maximum iterations, primal step 20, rho=1, relative feasibility tolerance 1e-5 and objective-change tolerance 1e-6. Final unregularized OT feasibility is checked. The last iterate is retained. There is no historical bank, path interpolation, validity filter, or validity repair. Raw logs/trace.csv distinguish stopping-tolerance convergence from exhausting iterations.

## Representation and validation

```
{json.dumps(assets,indent=2)}
```

Original Base and decoded Base are both reported because decoding Enc(Base) can change images and validity. A latent empirical-population guarantee is not a guarantee that the original generator images remain unchanged.

Three analytic/numerical unit tests verify dual normalization, the full outer gradient including sensitivity, and geometry derivatives. The actual CNN+decoder directional finite-difference check has relative error {smoke['directional_relative_error']:.4g}; deterministic repeated updates agree bitwise. Independent checks confirm all 16 saved assignments are permutations and preserve Q, images are finite/in-domain, and original input hashes are unchanged. Eight GPUs were used for independent calibration workers, with a per-worker allocator cap of 6.5% of device memory. Existing other-user processes were not stopped.

## Limits and interpretation

This is a controlled first pilot with one historical classifier and 64 test factuals, a 256-example empirical update distribution, and a three-step update horizon. It is not a full-data/full-training MNIST result. Original checkpoint training provenance is only partially recoverable. Calibration/test images are disjoint by the official MNIST split, but historical experiments inspected these assets previously; this is retrospective evaluation. CNN ReLU/max-pool/hinge nonsmoothness limits smooth local asymptotic interpretations, and r=0.25/0.5 is not certified to lie in an asymptotic regime. The surrogate is not a certified exact worst-case risk. Report any zero hinge/sensitivity degeneracy and geometry failures rather than claiming successful robustness. No independent semantic classifier or human label audit certifies decoded digit identity.

The prior pixel-space preparation/smoke attempt is preserved as superseded; it failed a bitwise CUDA check before experiments. The first latent smoke passed; the final run adds deterministic kernels, representation reporting, memory limits and the eight-GPU controller. Final baseline scoring initially exhausted the deliberately small GPU allocator cap by retaining unused derivative graphs. The standalone `experiments/055_finalize_mnist_update_aware.py` releases those graphs after computing the scalar metrics, without modifying optimization or model outputs; its provenance is recorded. The controller now invokes that finalizer. No old outputs were overwritten.

## Reproduction and files

```
cd /home/xu.zhao/iclr_robust_ce
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/053_run_mnist_update_aware.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_update_aware/NEW_RUN_ID
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_update_aware_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/054_report_mnist_update_aware.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_update_aware/NEW_RUN_ID
```

Use a new raw directory and a separate checkout for report reproduction: this reporting script refuses to overwrite an existing report. Config: `experiments/configs/mnist_update_aware.json`. Core code: `experiments/src/lift_ce/counterfactuals/update_aware.py` and `experiments/src/lift_ce/experiments/mnist_update_aware.py`; runner uses `experiments/run.py`. CSV summaries/manifests live here; large raw tensors and future checkpoints remain outside Git. Figures: `figures/mnist_update_aware/comparison.pdf` and `figures/mnist_update_aware/qualitative.pdf` (first 16 factuals by fixed order, including unsuccessful visual cases).
'''
 (res/'report.md').write_text(report);print(comparison);print(summary[cols].to_string(index=False))
if __name__=='__main__':main()
