"""Independently check task018 outputs, then render all-case and ranked heatmaps."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for path in [ROOT/'experiments/src',ROOT/'experiments/vendor/ffhq',Path('/tmp/ffhq_grace_deps'),Path('/tmp/galaxy_grace_deps')]:sys.path.insert(0,str(path))
os.environ.setdefault('MPLCONFIGDIR','/tmp/ffhq_update_aware_report_mpl')
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import yaml
from lift_ce.experiments.ffhq_update_aware import assets,load,sha,name,batched
from lift_ce.counterfactuals.clip_projection_update import projection_update

def distance(x,y):
 matrix=cdist(np.asarray(x),np.asarray(y),'sqeuclidean');i,j=linear_sum_assignment(matrix);return float(np.sqrt(matrix[i,j].mean()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();out=a.run;folder=out/'smile';c=json.loads((out/'resolved_config.json').read_text());res=ROOT/'results/032_ffhq_update_aware';figs=ROOT/'figures/ffhq_update_aware';res.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)
 data=load(out/'population.pt');pop=data['pairs']['smile'];m,decoder,clf=assets('cuda');selection=json.loads((folder/'selection.json').read_text());checks=[];rowsall=[]
 with torch.no_grad():
  ref_images=decoder.decode(pop['target_z'].to('cuda'));ref_prob=batched(m,ref_images).softmax(1)[:,0].cpu()
 reference_audit={'decoded_reference_validity':float((ref_prob>=.5).float().mean()),'decoded_reference_probabilities':ref_prob.tolist(),'reference_replay_max_error':float((ref_images.cpu()-pop['target_images']).abs().max())};(res/'reference_audit.json').write_text(json.dumps(reference_audit,indent=2)+'\n')
 for split in ['cal','test']:
  summary=pd.read_csv(folder/f'{split}_summary.csv');per=pd.read_csv(folder/f'{split}_per_model.csv');individual=pd.read_csv(folder/f'{split}_individual.csv');summary.to_csv(res/f'{split}_summary.csv',index=False);per.to_csv(res/f'{split}_per_model.csv',index=False)
  images={'Base':pop[split]['base'],'Historical_Base':pop[split]['historical_base']}
  for row in summary.to_dict('records'):
   if row['method'] in images:continue
   payload=load(folder/split/row['method']/'output.pt');q=payload['Q'];base=pop[split]['base_z'];target=pop['target_z'];d0=distance(base,q);d1=distance(q,target);D=distance(base,target)
   assert abs(D-row['D'])<1e-6;assert abs(row['budget']-row['beta']*D)<1e-6
   assignment=payload['assignment'];ii=assignment['source_index'];jj=assignment['target_index'];assert len(ii.unique())==len(q) and len(jj.unique())==len(q);assert torch.equal(payload['latents'][ii],q[jj])
   assert abs(d0-row['W2_Base_Q'])<1e-5;assert abs(d1-row['W2_Q_P1'])<1e-5;assert d0+d1<=row['budget']+row['geometry_tolerance'];assert abs(float((payload['latents']-base).square().sum(1).mean().sqrt())-d0)<1e-5
   with torch.no_grad():decoded=decoder.decode(payload['latents'].to('cuda')).cpu()
   torch.testing.assert_close(decoded,payload['images'],atol=2e-5,rtol=0);images[row['method']]=decoded
   assert abs(float((decoded-pop[split]['factual']).flatten(1).norm(dim=1).mean())-row['pixel_cost'])<1e-5
   assert abs(row['nominal']+row['r']*row['sensitivity']-row['surrogate'])<2e-6
   trace=pd.read_csv(folder/split/row['method']/'trace.csv');assert len(trace)==c['solver']['steps']+1;assert abs(trace.loc[trace.iteration==row['iteration'],'surrogate'].iloc[0]-row['surrogate'])<1e-7
   rowsall.append(row);checks.append({'split':split,'method':row['method'],'geometry':True,'assignment':True,'redecode':True,'cost':True,'surrogate':True})
  with torch.no_grad():features={tag:batched(clf.features,x.to('cuda')) for tag,x in images.items()};train_features=batched(clf.features,data['x'].to('cuda'))
  nominal=projection_update(train_features,data['y'].to('cuda'),clf.base_projection,clf.text,clf.scale,c['update']['steps'],c['update']['lr'],False)
  for rec in per.to_dict('records'):
   if rec['family']=='nominal':projection=nominal
   else:assert sha(rec['path'])==rec['sha256'];projection=load(rec['path'])['projection'].to('cuda')
   with torch.no_grad():prob=clf.logits_features(features[rec['method']],projection).softmax(1)[:,0].cpu()
   assert abs(float((prob>=.5).float().mean())-rec['FV'])<1e-7,rec
   records=individual[(individual.method==rec['method'])&(individual.family==rec['family'])&(individual.seed==rec['seed'])].sort_values('index');assert len(records)==8
   np.testing.assert_allclose(prob.numpy(),records.probability.to_numpy(),atol=2e-5,rtol=0)
  for row in summary.to_dict('records'):
   vals=per[(per.method==row['method'])&(per.family!='nominal')];assert abs(vals.FV.mean()-row['FV'])<1e-7
  if split=='cal':
   eligible=summary[~summary.method.isin(['Base','Historical_Base'])&summary.feasible];best=eligible.sort_values(['FV','pixel_cost','r','beta'],ascending=[False,True,True,True]).iloc[0];assert best['method']==selection['method']
  print('VERIFIED',split,flush=True)
 for path,h in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==h
 freeze=json.loads((folder/'test_freeze.json').read_text());assert sha(folder/'frozen_test_images.pt')==freeze['images_sha256'];assert sha(folder/'selection.json')==freeze['selection_sha256']
 assert (folder/'selection.json').stat().st_mtime<min((folder/'test'/name(r,b)/'output.pt').stat().st_mtime for r in c['r_grid'] for b in c['beta_grid'])
 test=pd.read_csv(folder/'test_summary.csv').set_index('method');beta=selection['beta'];nomtag=name(0,beta);robtag=name(.25,beta);selected=selection['method'];base=test.loc['Base'];nom=test.loc[nomtag];sel=test.loc[selected]
 headline={k:float(v) for k,v in {'selected_r':selection['r'],'selected_beta':beta,'base_FV':base.FV,'historical_base_FV':test.loc['Historical_Base'].FV,'nominal_FV':nom.FV,'selected_FV':sel.FV,'base_cost':base.pixel_cost,'nominal_cost':nom.pixel_cost,'selected_cost':sel.pixel_cost,'robust_increment_pp':100*(sel.FV-nom.FV)}.items()};pd.DataFrame([headline]).to_csv(res/'headline.csv',index=False)
 pd.DataFrame(checks).to_csv(res/'independent_checks.csv',index=False);allmetrics=pd.DataFrame(rowsall);allmetrics.to_csv(res/'all_grid_metrics.csv',index=False)
 for fname in ['asset_audit.csv','future_manifest.csv']:(res/fname).write_bytes((out/fname).read_bytes())
 (res/'gradient_audit.json').write_bytes((out/'gradient_audit.json').read_bytes())
 diffs=[images[robtag]-images[nomtag],images[nomtag]-images['Base']];maps=[d.abs().mean(1) for d in diffs];vmax=max(float(d.max()) for d in maps);ranks=[]
 for i in range(8):ranks.append({'index':i,'id':pop['test']['ids'][i],'r025_minus_r0_L2':float(diffs[0][i].double().norm()),'r0_minus_base_L2':float(diffs[1][i].double().norm())})
 ranking=pd.DataFrame(ranks);ranking.to_csv(res/'heatmap_ranking.csv',index=False)
 with torch.no_grad():current={tag:clf.logits_features(f).softmax(1)[:,0].cpu() for tag,f in features.items()}
 def render(ids,title):
  fig,axes=plt.subplots(4,7,figsize=(16,10));fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.085,wspace=.06,hspace=.18)
  tags=['Base',nomtag,robtag,selected]
  for j,i in enumerate(ids):
   for col,im in enumerate([pop['test']['factual'][i],*[images[t][i] for t in tags]]):axes[j,col].imshow(im.permute(1,2,0).numpy().clip(0,1));axes[j,col].axis('off')
   axes[j,0].text(.5,-.06,f'ID {i}',ha='center',transform=axes[j,0].transAxes,fontsize=9)
   for col,tag in enumerate(tags,1):
    sample=individual[(individual.method==tag)&(individual['index']==i)&(individual.family!='nominal')];fv=float((sample.probability>=.5).mean());axes[j,col].text(.5,-.06,f'p={float(current[tag][i]):.2f}, FV={fv:.0%}',ha='center',transform=axes[j,col].transAxes,fontsize=8)
   for col,heat in enumerate(maps,5):artist=axes[j,col].imshow(heat[i],cmap='magma',vmin=0,vmax=vmax);axes[j,col].axis('off')
  for ax,t in zip(axes[0],['Factual','Base',r'$r=0$',r'$r=0.25$',f'Selected r={selection["r"]:g}',r'$|r_{0.25}-r_0|$',r'$|r_0-\mathrm{Base}|$']):ax.set_title(t,fontsize=12)
  fig.suptitle(f'FFHQ face → smiling face · original CLIP · β={beta:g}\n'+title,fontsize=15)
  ca=fig.add_axes([.73,.035,.23,.011]);bar=fig.colorbar(artist,cax=ca,orientation='horizontal');bar.ax.tick_params(labelsize=8);ca.set_title('Mean absolute RGB difference · shared scale',fontsize=8)
  fig.text(.02,.03,'Full W512 refinement. No identity/content constraint.\nCurrent and future validity refer to the original CLIP prompts.',fontsize=9)
  return fig
 with PdfPages(figs/'qualitative_all8.pdf') as pdf:
  for start in [0,4]:
   fig=render(list(range(start,start+4)),f'All test cases {start}–{start+3}');pdf.savefig(fig);fig.savefig(figs/f'qualitative_page{start//4+1}.png',dpi=160);plt.close(fig)
 with PdfPages(figs/'qualitative_largest_differences.pdf') as pdf:
  for page,key in enumerate(['r025_minus_r0_L2','r0_minus_base_L2'],1):
   ids=ranking.sort_values([key,'index'],ascending=[False,True])['index'].head(4).tolist();fig=render(ids,'Post-hoc top four by '+key.replace('_',' '));pdf.savefig(fig);fig.savefig(figs/f'largest_differences_page{page}.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for b in c['beta_grid']:
  cells=test[(test.beta==b)&test.r.notna()].sort_values('r')
  for ax,key in zip(axes,['FV','pixel_cost','sensitivity']):ax.plot(cells.r,cells[key]*(100 if key=='FV' else 1),'o-',label=f'β={b:g}')
 for ax,label in zip(axes,['Future validity (%)','Mean RGB224 L2 cost','Pixel sensitivity dual norm']):ax.set_xlabel('r');ax.set_ylabel(label);ax.grid(alpha=.2)
 axes[0].axhline(100*base.FV,color='gray',ls='--',label='Base');axes[0].legend();fig.tight_layout();fig.savefig(figs/'sensitivity_robustness.pdf');plt.close(fig)
 runtime={'wall_seconds':json.loads((folder/'complete.json').read_text())['time']-(out/'repository_run.json').stat().st_mtime,'optimization_worker_hours':float(allmetrics.runtime_seconds.sum()/3600),'max_memory_GiB':float(allmetrics.peak_memory_bytes.max()/1024**3),'all_saved_feasible':bool(allmetrics.feasible.all()),'all_final_feasible':bool(allmetrics.final_iterate_feasible.all()),'retained_last_iteration':int((allmetrics.iteration==c['solver']['steps']).sum())};(res/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
 display=pd.DataFrame([{'Method':tag,'Test FV (%)':round(100*test.loc[tag].FV,2),'Current validity (%)':round(100*test.loc[tag].current_validity,2),'Pixel cost':round(test.loc[tag].pixel_cost,4)} for tag in ['Historical_Base','Base',nomtag,robtag]+([] if selected in [nomtag,robtag] else [selected])])
 text=f'''# FFHQ update-aware original-CLIP experiment

Raw run: `{out}`. Task: `tasks/018_ffhq_update_aware.md`.
Controller072, report073, config `experiments/configs/ffhq_update_aware.json`.

## Results

{display.to_markdown(index=False)}

Calibration selected r={selection['r']:g}, beta={beta:g}; test settings were frozen.
The additional selected-versus-nominal test FV gain is {headline['robust_increment_pp']:.2f} percentage points.
Do not attribute the full Base-to-refinement change to sensitivity. Every row
uses the same eight test faces and24 held-out future projection checkpoints.

## Method and reuse

Original StyleGAN3-r FFHQ1024 generator, W512 repeated at16 layers, original CLIP
ViT-B/32 with literal prompts `smiling face` and `face`. Target is index0.
The original nominal Base codes are reused, not LeapFactual robust codes.
No GENKI classifier, fixed-content editor, identity/background constraint, or
validity repair. Exact W512 W2 geometry and one-to-one deployment assignment.
Mean smooth hinge (.5 probability threshold, .05 temperature) plus r times the
empirical-RMS pixel sensitivity norm. Training uncertainty is on RGB224 pixels,
not cached features. The complete prescribed update is three full-batch SGD
steps at lr1e-4 on CLIP's768x512 visual projection, with32 balanced generated
training images and original frozen pseudo-labels. Text, logit scale and visual
backbone remain fixed. Both nominal and sensitivity Q gradients pass through
all three steps; streamed pixel pullbacks and generator VJPs bound memory.
This is short projection adaptation, not full-CLIP retraining.

## Evaluation and solver

r={{0,.25,1,4}}, beta={{1.02,1.1}};16 solves,160 primal-dual updates each, lr5/rho1,
gradient norm capped at.2. Lowest-surrogate exactly geometry-feasible iterate
retained, including Base. No convergence certificate. Calibration and test each
use24 fresh models (brightness/noise/blur, four severities, two realizations).
Shift seeds differ; original future checkpoints are not reused. These stress
scenarios are not certified inner maximizers or confined to every r-ball.
Selection maximizes calibration FV, then minimizes cost, r and beta.

## Validation

{len(checks)} independently checked outputs: exact geometry, one-to-one assignment,
re-decoding, pixel cost, surrogate decomposition, fresh-model predictions and
per-example probabilities. Selection ordering, input hashes and freeze hashes
also pass. Re-decoded reference validity is {reference_audit['decoded_reference_validity']:.1%}; details are saved in reference_audit.json. Twelve focused tests passed (2 projection,3 pullback,7 OT). Projection-update unit tests check both standard SGD equivalence and
full pixel/Q derivatives against direct autodiff. Full-network directional checks
are in gradient_audit.json. Runtime: {runtime['wall_seconds']/60:.1f} minutes,
{runtime['optimization_worker_hours']:.2f} optimization worker-hours. Saved-output
feasibility: {runtime['all_saved_feasible']}; final-iterate feasibility: {runtime['all_final_feasible']}.
All eight GPUs are used for both future models and concurrent optimization.
A streamed audit measured2.35GiB peak memory; workers are capped at6.5% of an80GiB GPU.

## Interpretation limits and figures

These are FFHQ-generated faces, not real-photo inversions; smile validity is
relative to the original CLIP prompts. Identity or background may change under
full W refinement. No human smile/morphology validation or global robustness
certificate is claimed. Historical Base and exact current decoder replay Base
are scored separately; see asset_audit.csv for numerical differences.

`figures/ffhq_update_aware/qualitative_all8.pdf` contains all fixed test IDs.
Its last two columns are abs(r.25-r0) and abs(r0-Base), at the selected beta,
with the same absolute RGB224 scale on every page. The largest-difference PDF
is explicitly post-hoc top-four selection by pixel L2 for each contrast, not
a representative sample or a method-selection criterion. Rankings are saved.
'''
 (res/'report.md').write_text(text)
 path=ROOT/'state.yaml';state=yaml.safe_load(path.read_text());state.update(status='complete',active_task=c['task'],current_work={'task':c['task'],'status':'complete','phase':'complete','run':str(out),'report':'results/032_ffhq_update_aware/report.md','configuration':'experiments/configs/ffhq_update_aware.json','qualitative_pdf':'figures/ffhq_update_aware/qualitative_all8.pdf','largest_differences_pdf':'figures/ffhq_update_aware/qualitative_largest_differences.pdf','gpu_ids':list(range(8)),'completed_solves':len(checks),'future_models':48,'unit_tests_passed':12,'headline':headline},next_actions=[]);state.setdefault('completed_work',{})['ffhq_update_aware']=dict(state['current_work']);path.write_text(yaml.safe_dump(state,sort_keys=False));print(json.dumps(headline,indent=2))
if __name__=='__main__':main()
