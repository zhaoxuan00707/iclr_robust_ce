"""Saved-image evaluation and all-case figures for the matched capacity/margin experiment."""
import argparse,json,hashlib,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments/src'),'/tmp/ffhq_grace_deps']
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from lift_ce.ffhq.genki_classifier import GenkiSmileClassifier

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);out=p.parse_args().run;c=json.loads((out/'resolved_config.json').read_text());parent=Path(c['parent_run']);inc=Path(c['parent_feasible'])
 result=ROOT/'results/023_genki_expanded_editor';fig=ROOT/'figures/genki_expanded_editor';result.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True);assert not (result/'report.md').exists()
 for path,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==digest,path
 for path,digest in json.loads((out/'outputs_frozen.json').read_text())['image_hashes'].items():assert sha(path)==digest
 jobs=[p for p in (out/'jobs').glob('*/complete.json') if not p.parent.is_symlink()];assert len(jobs)==114,len(jobs)
 controls=json.loads((out/'controls_frozen.json').read_text());assert not controls['test_used'] and controls['training_indices']==list(range(32));assert sha(out/'expanded_controls.pt')==controls['sha256']
 arms=[f'c{k}_t{t}' for t in [5,7] for k in [4,8,16]];methods=['Base','Previous']+arms
 geometry=pd.concat([pd.read_csv(out/f'{arm}_geometry.csv') for arm in arms],ignore_index=True);assert (geometry.total_length<=geometry.B_geo+1e-6).all();assert geometry.D.nunique()==1
 q=pd.read_csv(out/'image_quality.csv');assert len(q)==64
 for field,threshold,minimum in [('arcface_similarity',.8,True),('facenet_similarity',.8,True),('LPIPS',.15,False),('outside_rms',.04,False)]:
  assert q[field].notna().all();assert ((q[field]>=threshold-1e-6) if minimum else (q[field]<=threshold+1e-6)).all()
 for i in range(8):
  for method,old in [('Base','Base'),('Previous','Feasible_search')]:assert sha(out/'qualitative_examples'/f'test_{i}_{method}.png')==sha(inc/'qualitative_examples'/f'test_{i}_{old}.png')
 summaries=[]
 for t in [5,7]:
  for k in [4,8,16]:
   assert {json.loads((out/'jobs'/f'quality_{k}_{t}_{i}'/'complete.json').read_text())['gpu'] for i in range(8)}==set(map(str,range(8)))
   for i in range(8):
    job=out/'jobs'/f'search_{k}_{t}_{i}';assert sha(out/'qualitative_examples'/f'proposal_c{k}_t{t}_{i}_0.png')==sha(inc/'qualitative_examples'/f'test_{i}_Feasible_search.png');summaries.append(json.loads((job/'summary.json').read_text()))
    if (job/'trace.csv').stat().st_size>1:
     trace=pd.read_csv(job/'trace.csv');accepted=trace[trace.accepted];assert accepted.quality_pass.all();assert (accepted.risk<accepted.previous_risk).all()
 torch.set_num_threads(4);clf=GenkiSmileClassifier(out/'checkpoints/f0.pt','cpu');cache={};pred=[];features=[];diff=[]
 for row in q.itertuples():
  path=out/'qualitative_examples'/f'test_{row.factual_index}_{row.method}.png';digest=sha(path)
  if digest not in cache:
   native=TF.to_tensor(Image.open(path).convert('RGB'))[None]*2-1
   with torch.no_grad():cache[digest]=clf.features(clf.canonical(native))
  f=cache[digest];features.append(f)
  with torch.no_grad():prob=float(clf.head(f).softmax(1)[0,0])
  pred.append(dict(method=row.method,factual_index=row.factual_index,sha256=digest,nominal_probability=prob))
  base=out/'qualitative_examples'/f'test_{row.factual_index}_Base.png';before=out/'qualitative_examples'/f'test_{row.factual_index}_Previous.png'
  arr=np.array(Image.open(path)).astype(float)/255;b=np.array(Image.open(base)).astype(float)/255
  diff.append(dict(method=row.method,factual_index=row.factual_index,RGB_RMS_vs_Base=float(np.mean((arr-b)**2)**.5),identical_to_previous=sha(before)==digest))
 d=pd.DataFrame(pred);dif=pd.DataFrame(diff);x=torch.cat(features);future=[];bank=[];modelhash={}
 def probs(path):
  modelhash[str(path)]=sha(path);s=torch.load(path,weights_only=True,map_location='cpu');return (x@(s['weight'][0]-s['weight'][1])+s['bias'][0]-s['bias'][1]).sigmoid()
 for path in sorted((parent/'jobs').glob('future_*/model.json')):
  meta=json.loads(path.read_text());values=probs(meta['checkpoint']);future.extend(dict(method=r.method,factual_index=r.factual_index,**meta['scenario'],probability=float(v),valid=bool(v>=.5)) for r,v in zip(d.itertuples(),values))
 for row in pd.read_csv(parent/'attack_bank_manifest.csv').itertuples():
  if row.paired_W_data_upper_bound>2+1e-6:continue
  values=probs(row.checkpoint);bank.extend(dict(model=row.candidate,method=r.method,factual_index=r.factual_index,probability=float(v),hinge05=max(0,.5-float(v)),hinge07=max(0,.7-float(v))) for r,v in zip(d.itertuples(),values))
 f=pd.DataFrame(future);b=pd.DataFrame(bank);assert len(f)==1920
 rows=[]
 for method in methods:
  quality=q[q.method==method];pr=d[d.method==method];risk=b[b.method==method].groupby('model')[['hinge05','hinge07']].mean().max()
  rows.append(dict(method=method,current_validity=float((pr.nominal_probability>=.5).mean()),future_validity=float(f[f.method==method].valid.mean()),mean_nominal_probability=float(pr.nominal_probability.mean()),nominal_target07_count=int((pr.nominal_probability>=.7).sum()),ArcFace_mean=float(quality.arcface_similarity.mean()),ArcFace_min=float(quality.arcface_similarity.min()),FaceNet_min=float(quality.facenet_similarity.min()),LPIPS_mean=float(quality.LPIPS.mean()),outside_RMS_mean=float(quality.outside_rms.mean()),bank_hinge05=risk.hinge05,bank_hinge07=risk.hinge07,RGB_RMS_vs_Base=float(dif[dif.method==method].RGB_RMS_vs_Base.mean())))
 table=pd.DataFrame(rows)
 for name,data in [('main_results',table),('image_predictions',d),('future_per_case',f),('bank_per_case',b),('image_differences',dif),('geometry',geometry),('search_summary',pd.DataFrame(summaries))]:data.to_csv(result/f'{name}.csv',index=False)
 for name in ['resolved_config.json','input_manifest.json','channel_ranking.csv','controls_frozen.json','geometry_setup.json','image_quality.csv']:shutil.copyfile(out/name,result/name)
 (result/'evaluated_model_hashes.json').write_text(json.dumps(modelhash,indent=2)+'\n')
 scores=d.set_index(['method','factual_index']);fv=f.groupby(['method','factual_index']).valid.mean()
 with PdfPages(fig/'qualitative.pdf') as pdf:
  for t in [5,7]:
   cols=['Factual','Base','Previous']+[f'c{k}_t{t}' for k in [4,8,16]];labels=['Factual','Base','Previous GRACE','4 channels','8 channels','16 channels']
   for start in range(0,8,2):
    figure,axes=plt.subplots(2,6,figsize=(16,8));figure.subplots_adjust(left=.025,right=.99,top=.85,bottom=.13,hspace=.35,wspace=.04)
    figure.suptitle(f'GENKI GRACE: editor capacity at optimization target {t/10:.1f}',fontsize=16,y=.97)
    for r,i in enumerate(range(start,start+2)):
     for col,method in enumerate(cols):
      ax=axes[r,col];im=Image.open(out/'qualitative_examples'/f'test_{i}_{method}.png');im.thumbnail((512,512));ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
      if r==0:ax.set_title(labels[col],fontsize=12,pad=10)
      if col==0:ax.set_ylabel(f'Case {i+1}',fontsize=12)
      else:ax.set_xlabel(f'p={scores.loc[method,i].nominal_probability:.3f} | FV={fv.loc[method,i]:.0%}',fontsize=10)
    figure.text(.03,.035,'Identical starting CFs, bank, search settings and preservation gates across new arms. Classification threshold stays 0.5.\nAll 8 generated faces shown. Future heads were previously seen. Higher confidence does not establish a visible smile.',fontsize=10)
    pdf.savefig(figure)
    if start==4:figure.savefig(fig/f'preview_target_{t}.png',dpi=110)
    plt.close(figure)
 # Predeclared largest editor/high-margin arm; select examples by image difference, not future validity.
 chosen=dif[dif.method=='c16_t7'].nlargest(3,'RGB_RMS_vs_Base').factual_index.tolist()
 figure,axes=plt.subplots(3,4,figsize=(12,11));figure.subplots_adjust(left=.04,right=.98,top=.90,bottom=.08,hspace=.27,wspace=.08)
 figure.suptitle('Base vs 16-channel GRACE, target 0.7: largest differences',fontsize=15,y=.97)
 for r,i in enumerate(chosen):
  for col,(method,crop) in enumerate([('Base',False),('c16_t7',False),('Base',True),('c16_t7',True)]):
   ax=axes[r,col];im=np.array(Image.open(out/'qualitative_examples'/f'test_{i}_{method}.png'));h,w=im.shape[:2]
   if crop:im=im[int(.5*h):int(.87*h),int(.27*w):int(.73*w)]
   ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
   if r==0:ax.set_title(('Base' if method=='Base' else 'Ours')+(' — detail' if crop else ''),fontsize=12)
   if col==0:ax.set_ylabel(f'Case {i+1}',fontsize=12)
   if not crop:ax.set_xlabel(f'p={scores.loc[method,i].nominal_probability:.3f} | FV={fv.loc[method,i]:.0%}',fontsize=10)
 figure.text(.04,.025,'Top three by RGB RMS difference from Base in the predeclared 16-channel, 0.7-target arm. No image edits or contrast amplification.\nSame crop coordinates within each pair. All cases and all six arms are available in qualitative.pdf.',fontsize=9)
 figure.savefig(fig/'largest_differences.pdf',dpi=160);figure.savefig(fig/'largest_differences.png',dpi=130);plt.close(figure)
 checks={'tests_passed':7,'completed_jobs':114,'all_selected_preservation_gates_passed':True,'frozen_input_and_output_hashes_verified':True,'same_base_and_starting_images':True,'all_arms_same_16D_geometry':True,'controls_frozen_from_training_only':True,'eight_GPU_final_screening_each_arm':True,'accepted_steps_quality_and_improvement_verified':True,'primary_evaluation_uniform_saved_PNG_CPU_float32':True,'all_cases_included':True,'classification_threshold':.5,'all_arms_reported_no_future_selection':True,'largest_difference_cases':[i+1 for i in chosen]}
 (result/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 report=f'''# Expanded-channel and margin experiment

Raw run: `{out}`. Fixed bank/original Base parent: `{parent}`. Starting feasible CF parent: `{inc}`.

Matched six-arm design: 4, 8, 16 mutable channels crossed with target 0.5 or 0.7 in max-model mean-population hinge. Classification is always evaluated at 0.5; the 0.7 target is an objective ablation, not the original objective. Every arm starts from the same last feasible four-channel result, uses 48 attempts per face (step 0.05, cap 0.3, growth 1.3, halve on rejection), and permits all eight faces to move. The new four-channel control accounts for the longer search and release of previously fixed faces. Proposals must improve conditional bank risk and pass identity/background and exact geometry gates before acceptance. Greedy coordinate combination uses the global bank objective; no future outcomes select candidates. This is a finite block search, not a global solution or jointly differentiated identity constraint.

The original four channels are retained. Twelve additional channels were selected from only the first 32 existing generated training faces using mean GENKI smile gradient divided by estimated outside-region Jacobian RMS, eligible layers 3–11, sign agreement >=0.75, three random probes per face. Nested 8/16 channel sets are frozen before test preparation. Channel IDs and ranking are recorded. No new model/bank training occurred.

All six arms share exact W2 geometry in the union of 16 standardized style coordinates, radius 2 and beta 3. Original Base/reference images are fixed; inactive coordinates stay factual. This extends the previous 4-D geometry, so direct capacity/margin claims use the new matched controls. Fourier input and every unselected style remain factual. Preservation limits are unchanged: ArcFace>=0.8, FaceNet>=0.8, LPIPS<=0.15, outside RMS<=0.04. The outside mask is an ellipse, not exact background segmentation. Metric thresholds are not proof of unchanged human identity.

Primary saved-image results:

```
{table.to_string(index=False)}
```

Geometry:

```
{geometry.to_string(index=False)}
```

All predicted probabilities and future metrics are recomputed uniformly on saved PNGs using CPU float32, with image-hash caching. Native LPIPS/outside diagnostics retain their original definitions. Bank hinge at both 0.5 and 0.7 is reported on the same saved images; optimization traces retain native cached risks. Larger pixel differences do not establish stronger smiles, and classifier confidence is not a substitute for human expression labels.

This is exploratory reuse of eight generated FFHQ faces and 30 previously inspected future heads (18 distinct states). It does not invert real GENKI photos or provide fresh independent validation. All failures and arms are shown. `figures/genki_expanded_editor/qualitative.pdf` has eight pages: four per target. `largest_differences.pdf` shows the top three image differences in the predeclared 16-channel/0.7 arm, selected by RGB RMS, not success. These selections are illustrations, not representative statistical evidence.

Reproduction: wm_gpu Python, `experiments/049_run_genki_expanded.py --output {out}`, module `genki_expanded_editor` through `experiments/run.py`; seed 20260918. Configuration and input/model hashes accompany this report. Validation: seven relevant unit tests, 114 completed jobs, frozen controls/inputs/images, exact common geometry, all accepted-step and final preservation gates, eight GPU IDs for each final arm screen. Existing raw outputs remain unchanged.
'''
 (result/'report.md').write_text(report)
 statepath=ROOT/'state.yaml';state=yaml.safe_load(statepath.read_text());state['active_task']='tasks/010_genki_expanded_editor.md';state['status']='completed';state['current_work']={'task':state['active_task'],'phase':'completed','run':str(out),'report':str(result/'report.md'),'qualitative_pdf':str(fig/'qualitative.pdf'),'completed_jobs':114};statepath.write_text(yaml.safe_dump(state,sort_keys=False))
 with (ROOT/'notes/decisions.md').open('a') as stream:stream.write('\n\n## GENKI expanded editor and target ablation\nUser approved 8/16 channels and target 0.7. Completed matched 4/8/16 by 0.5/0.7 with training-only nested channel discovery, common 16-D geometry and fixed preservation gates. All eight faces remain in every arm; no future-based arm selection. Report: results/023_genki_expanded_editor/report.md. Figures: figures/genki_expanded_editor/qualitative.pdf.\n')
 print(table.to_string(index=False),flush=True)
if __name__=='__main__':main()
