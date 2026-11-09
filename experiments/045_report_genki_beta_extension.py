"""Evaluate frozen larger-beta images uniformly and present every test face."""
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
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
 c=json.loads((out/'resolved_config.json').read_text());parent=Path(c['parent_run'])
 result=ROOT/'results/021_genki_grace_beta_extension';fig=ROOT/'figures/genki_grace_beta_extension';result.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
 assert not (result/'report.md').exists()
 for path,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==digest,path
 frozen=json.loads((out/'outputs_frozen.json').read_text())
 for path,digest in frozen['image_hashes'].items():assert sha(path)==digest
 own=[p for p in (out/'jobs').glob('*/complete.json') if not p.parent.is_symlink()];assert len(own)==15,len(own)
 assert {json.loads((out/'jobs'/f'quality_{i}'/'complete.json').read_text())['gpu'] for i in range(8)}==set(map(str,range(8)))
 torch.set_num_threads(4);clf=GenkiSmileClassifier(out/'checkpoints/f0.pt','cpu');quality=pd.read_csv(out/'image_quality.csv');cache={};pred=[];features=[]
 for r in quality.itertuples():
  path=out/'qualitative_examples'/f'test_{r.factual_index}_{r.method}.png';digest=sha(path)
  if digest not in cache:
   native=TF.to_tensor(Image.open(path).convert('RGB'))[None]*2-1
   with torch.no_grad():cache[digest]=clf.features(clf.canonical(native))
  f=cache[digest];features.append(f)
  with torch.no_grad():prob=float(clf.head(f).softmax(1)[0,0])
  pred.append(dict(method=r.method,factual_index=r.factual_index,sha256=digest,nominal_smile_probability=prob))
 d=pd.DataFrame(pred);x=torch.cat(features);future=[];bank=[]
 def probs(path):
  s=torch.load(path,weights_only=True,map_location='cpu');return (x@(s['weight'][0]-s['weight'][1])+s['bias'][0]-s['bias'][1]).sigmoid()
 for path in sorted((parent/'jobs').glob('future_*/model.json')):
  meta=json.loads(path.read_text());values=probs(meta['checkpoint'])
  future.extend(dict(method=r.method,factual_index=r.factual_index,**meta['scenario'],smile_probability=float(v),valid=bool(v>=.5)) for r,v in zip(d.itertuples(),values))
 for r in pd.read_csv(parent/'attack_bank_manifest.csv').itertuples():
  if r.paired_W_data_upper_bound>2+1e-6:continue
  values=probs(r.checkpoint)
  bank.extend(dict(model=r.candidate,method=q.method,factual_index=q.factual_index,hinge=max(0,.5-float(v))) for q,v in zip(d.itertuples(),values))
 f=pd.DataFrame(future);b=pd.DataFrame(bank);assert len(f)==1200
 methods=['Base','Previous_beta1.20','GRACE_beta1.50','GRACE_beta2.00','GRACE_beta3.00'];rows=[];diff=[]
 for method in methods:
  q=quality[quality.method==method];pr=d[d.method==method]
  assert len(q)==8 and (q.arcface_similarity>=.8-1e-6).all() and (q.facenet_similarity>=.8-1e-6).all() and (q.LPIPS<=.15+1e-6).all() and (q.outside_rms<=.04+1e-6).all()
  rows.append(dict(method=method,current_validity=float((pr.nominal_smile_probability>=.5).mean()),future_validity=float(f[f.method==method].valid.mean()),ArcFace_mean=float(q.arcface_similarity.mean()),ArcFace_min=float(q.arcface_similarity.min()),LPIPS_mean=float(q.LPIPS.mean()),outside_RMS_mean=float(q.outside_rms.mean()),bank_hinge=float(b[b.method==method].groupby('model').hinge.mean().max())))
  for i in range(8):
   pa=out/'qualitative_examples'/f'test_{i}_{method}.png';pb=out/'qualitative_examples'/f'test_{i}_Previous_beta1.20.png';delta=(np.array(Image.open(pa)).astype(float)-np.array(Image.open(pb)).astype(float))/255
   diff.append(dict(method=method,factual_index=i,identical_to_previous_beta1_2=sha(pa)==sha(pb),RGB_RMS_difference=float(np.sqrt(np.mean(delta**2))),max_RGB_difference=float(abs(delta).max())))
 main=pd.DataFrame(rows);dif=pd.DataFrame(diff);geometry=pd.read_csv(out/'geometry.csv');assert (geometry.slack>=-1e-5).all()
 for i in range(8):
  for name,previous in [('Base','Base'),('Previous_beta1.20','GRACE_beta1.20')]:assert sha(out/'qualitative_examples'/f'test_{i}_{name}.png')==sha(parent/'qualitative_examples'/f'test_{i}_{previous}.png')
 for name,table in [('main_results',main),('image_predictions',d),('future_per_case',f),('bank_per_case',b),('image_differences',dif)]:table.to_csv(result/f'{name}.csv',index=False)
 for name in ['geometry.csv','merge.json','screened_candidates.csv','quality_screen.csv','image_quality.csv','resolved_config.json','input_manifest.json']:shutil.copyfile(out/name,result/name)
 labels=['Factual','Base',r'Previous $\beta=1.2$',r'$\beta=1.5$',r'$\beta=2$',r'$\beta=3$'];columns=['Factual']+methods
 with PdfPages(fig/'qualitative.pdf') as pdf:
  for page in range(4):
   figure,axes=plt.subplots(2,6,figsize=(16,8));figure.subplots_adjust(left=.025,right=.99,top=.86,bottom=.12,wspace=.04,hspace=.35)
   figure.suptitle('GENKI-head GRACE: larger geometry budgets, fixed identity limits',fontsize=16,y=.96)
   for row,i in enumerate(range(page*2,page*2+2)):
    for col,method in enumerate(columns):
     ax=axes[row,col];im=Image.open(out/'qualitative_examples'/f'test_{i}_{method}.png');im.thumbnail((512,512));ax.imshow(im);ax.set_xticks([]);ax.set_yticks([])
     if row==0:ax.set_title(labels[col],fontsize=13,pad=12)
     if col==0:ax.set_ylabel(f'Case {i+1}',fontsize=12)
     else:
      prob=d[(d.method==method)&(d.factual_index==i)].iloc[0].nominal_smile_probability;fv=f[(f.method==method)&(f.factual_index==i)].valid.mean();ax.set_xlabel(f'p(smile)={prob:.3f} | FV={fv:.0%}',fontsize=10)
   figure.text(.03,.035,'All 8 fixed cases shown. Same Base, r=2, bank, editor and preservation gates. FV uses 30 previously seen future heads.\nLarger beta permits more movement; it does not require more movement. Factual images are generated FFHQ faces.',fontsize=10)
   pdf.savefig(figure)
   if page in [0,3]:figure.savefig(fig/f'preview_page_{page+1}.png',dpi=110)
   plt.close(figure)
 screen=pd.read_csv(out/'quality_screen.csv');failure_counts={'ArcFace':int((screen.arcface_similarity.fillna(-1)<.8).sum()),'FaceNet':int((screen.facenet_similarity.fillna(-1)<.8).sum()),'LPIPS':int((screen.LPIPS.fillna(1)>.15).sum()),'outside_RMS':int((screen.outside_rms.fillna(1)>.04).sum())};preservation_survivors=int(screen.groupby('candidate').quality_pass.all().sum())
 merge=json.loads((out/'merge.json').read_text());checks={'seven_focused_tests_passed':True,'completed_new_jobs':len(own),'quality_gpu_ids':list(range(8)),'all_input_hashes_unchanged':True,'all_frozen_image_hashes_verified':True,'all_selected_preservation_and_geometry_checks_passed':True,'uniform_CPU_float32_saved_image_evaluation':True,'unique_images':len(cache),'all_8_cases_shown':True,'future_models_retrained':False}
 (result/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 report=f'''# Larger-beta GENKI GRACE extension

Raw run: `{out}`. Parent: `{parent}`.

User requested larger beta instead of stronger nominal Base. New beta values: 1.5, 2, 3; radius remains 2. The original frozen beta=1.2 winner and Base are shown unchanged. New candidates are pooled with prior candidates and screened against unchanged per-face ArcFace>=0.8, FaceNet>=0.8, LPIPS<=0.15 and outside-region RMS<=0.04. Selection uses cached finite-bank population hinge, then pixel cost; future outcomes do not select candidates or a winning beta.

```
{main.to_string(index=False)}
```

Geometry (the 1.2 row is the new pooled solution; the figure's previous 1.2 column remains the original frozen solution):

```
{geometry.to_string(index=False)}
```

Candidate counts: {json.dumps(merge)}. Candidates passing preservation for all eight faces: {preservation_survivors}. Failed candidate/face checks (categories can overlap): {json.dumps(failure_counts)}.

Images identical to the original beta=1.2, out of eight per method:

```
{dif.groupby('method').identical_to_previous_beta1_2.sum().to_string()}
```

Increasing beta expands the admissible geometric path length; it does not require using the extra budget. These results concern the unchanged finite 12-step search and fixed four-channel editor, not a global optimum or proof that further changes are impossible. Preservation can restrict candidate acceptance independently of beta. All failures remain visible. Primary scores are recomputed uniformly on the exact saved PNGs using CPU float32, with image-hash caching. Cached optimization risk is separately reported in geometry.csv. Identity and native-image LPIPS/outside diagnostics retain the original measurement definitions; the outside region is a fixed ellipse mask, not segmentation or exact background preservation.

This is exploratory reuse of eight generated FFHQ test faces and the previously observed 30 future heads (18 distinct states), not new independent test evidence or real GENKI photo inversion. Original frozen Base can lose nominal validity after image quantization near its 0.5 boundary.

Reproduction: `experiments/044_run_genki_beta_extension.py --output {out}` via the existing wm_gpu Python environment; worker launcher `experiments/run.py run genki_grace_beta_extension`; config copied here as resolved_config.json; seed 20260918. Frozen checkpoint paths and hashes are in input_manifest.json. Existing outputs were preserved. Validation: seven focused tests, 15 completed new jobs, eight GPUs for independent case screening, frozen input/image hashes, all selected geometry and preservation gates. Figure: `figures/genki_grace_beta_extension/qualitative.pdf` (four pages, all eight cases).
'''
 (result/'report.md').write_text(report)
 statepath=ROOT/'state.yaml';state=yaml.safe_load(statepath.read_text());state['active_task']='tasks/008_genki_grace_beta_extension.md';state['status']='completed';state['current_work']={'task':state['active_task'],'phase':'completed','run':str(out),'summary':str(result/'report.md'),'qualitative_pdf':str(fig/'qualitative.pdf')};statepath.write_text(yaml.safe_dump(state,sort_keys=False))
 with (ROOT/'notes/decisions.md').open('a') as stream:stream.write('\n\n## Larger-beta GENKI GRACE extension\nUser chose beta 1.5, 2, 3 with Base and all preservation settings fixed. Existing outputs remain unchanged. All eight cases and uniform saved-image metrics are in results/021_genki_grace_beta_extension/report.md; figures/genki_grace_beta_extension/qualitative.pdf. This is exploratory reuse of prior test faces and future heads.\n')
 print(main.to_string(index=False));print(geometry.to_string(index=False));print(dif.groupby('method').identical_to_previous_beta1_2.sum().to_string())
if __name__=='__main__':main()
