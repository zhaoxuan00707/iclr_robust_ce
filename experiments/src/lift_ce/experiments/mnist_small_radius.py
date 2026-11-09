"""Predeclared near-zero MNIST grid with broader, fresh future scenarios."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
import pandas as pd
import torch
from .stylemnist_grace_v1 import assets,load,sha,dump,shifted_images
from .mnist_update_aware import data,update_model
from .mnist_latent_base_fix import optimize
from ..counterfactuals.update_aware import surrogate
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/mnist_small_radius.json'

def prepare(a,c):
 parent=Path(c['parent_run']);manifest=json.loads((parent/'input_manifest.json').read_text())
 for p,h in manifest.items():assert sha(Path(p))==h
 for file in ['population.pt','base_recovery.json']:
  shutil.copyfile(parent/file,a.output/file);manifest[str(parent/file)]=sha(parent/file)
 for p in [CONFIG,ROOT/c['task']]:manifest[str(p)]=sha(p)
 d=data(a);m,ae,_=assets(c,a.device);checks=[]
 for split in ['cal','test']:
  q=d[split]['base_z'].to(a.device).requires_grad_()
  with torch.no_grad():im=ae.decode(q);assert torch.allclose(im.cpu(),d[split]['base'],atol=1e-5,rtol=0);assert (m(im).sigmoid()>=.5).all()
  f,n,s=surrogate(m,d['x'].to(a.device),d['y'].to(a.device),q,.1,steps=c['update']['steps'],lr=c['update']['lr'],decode=ae.decode,hinge_temperature=.05);g=torch.autograd.grad(f,q)[0];v=g.detach()/g.norm();analytic=float(g.norm());row={'split':split,'nominal':float(n.detach()),'sensitivity':float(s.detach()),'directional_autograd':analytic};q=q.detach();del f,n,s,g
  values=[]
  for sign in [-1,1]:
   f,n,s=surrogate(m,d['x'].to(a.device),d['y'].to(a.device),q+sign*.001*v,.1,steps=c['update']['steps'],lr=c['update']['lr'],decode=ae.decode,hinge_temperature=.05);values.append(float(f.detach()));del f,n,s
  fd=(values[1]-values[0])/.002;row.update(finite_difference=fd,relative_error=abs(fd-analytic)/max(analytic,1e-8));assert row['relative_error']<.15;checks.append(row)
 dump(a.output/'audit.json',checks);dump(a.output/'input_manifest.json',manifest);dump(a.output/'resolved_config.json',c)

def futures(a,c):
 d=data(a);m,_,_=assets(c,a.device);rows=[];specs=[(sp,f,st,sv,rep) for sp in ['cal','test'] for f,st in zip(c['future']['families'],c['future']['strengths']) for sv in c['future']['severities'] for rep in range(3)];folder=a.output/'future_models';folder.mkdir(exist_ok=True)
 for idx,(split,family,strength,severity,rep) in enumerate(specs):
  if idx%8!=a.worker:continue
  seed=c['future'][split+'_seed']+rep;xx=shifted_images(d['x'],family,strength*severity,seed);h=hashlib.sha256(xx.numpy().tobytes()).hexdigest();t=time.perf_counter();u=update_model(m,xx.to(a.device),d['y'].to(a.device),c);p=folder/f'model{idx:03d}.pt';torch.save(u.state_dict(),p);rows.append(dict(split=split,family=family,severity=severity,realization=rep,seed=seed,path=str(p),sha256=sha(p),distribution_sha256=h,seconds=time.perf_counter()-t));del u
 pd.DataFrame(rows).to_csv(a.output/f'future_worker{a.worker}.csv',index=False)

def merge(a,c):
 df=pd.concat([pd.read_csv(a.output/f'future_worker{i}.csv') for i in range(8)]);assert len(df)==72
 for split in ['cal','test']:df[df.split==split].to_csv(a.output/f'{split}_model_manifest.csv',index=False)
 sets=[set(df[df.split==s].distribution_sha256) for s in ['cal','test']];dump(a.output/'future_checks.json',{'cal_unique':len(sets[0]),'test_unique':len(sets[1]),'cross_split_identical':len(sets[0]&sets[1]),'scenarios_per_split':36})

def score(a,c,images,split):
 m,_,_=assets(c,a.device);rows=[]
 for rec in pd.read_csv(a.output/f'{split}_model_manifest.csv').to_dict('records'):
  assert sha(Path(rec['path']))==rec['sha256'];m.load_state_dict(load(Path(rec['path'])))
  with torch.no_grad():
   for method,im in images.items():
    p=m(im.to(a.device)).sigmoid();rows.append({**rec,'method':method,'future_validity':float((p>=.5).float().mean()),'smooth_loss':float((.05*torch.nn.functional.softplus((.5-p)/.05)).mean())})
 return pd.DataFrame(rows)

def collect(a,split):
 d=data(a);rows=[];images={'Base':d[split]['base'],'Historical-Base':d[split]['historical_base']}
 for p in sorted((a.output/split).glob('*/metrics.json')):
  row=json.loads(p.read_text());row['method']=p.parent.name;rows.append(row)
  if row['feasible']:images[row['method']]=load(p.parent/'output.pt')['images']
 assert len(rows)==16
 return rows,images

def select(a,c):
 rows,images=collect(a,'cal');f=score(a,c,images,'cal');f.to_csv(a.output/'calibration_per_model.csv',index=False)
 for row in rows:
  v=f[f.method==row['method']];row.update(cal_FV=float(v.future_validity.mean()) if len(v) else None,cal_FV_severity1=float(v[v.severity==1].future_validity.mean()) if len(v) else None)
 frame=pd.DataFrame(rows);frame.to_csv(a.output/'calibration_summary.csv',index=False);best=frame[frame.feasible].sort_values(['cal_FV','pixel_cost','r','beta'],ascending=[False,True,True,True]).iloc[0].to_dict();dump(a.output/'selection.json',best);print('SELECTED',best,flush=True)

def evaluate(a,c):
 rows,images=collect(a,'test');sel=json.loads((a.output/'selection.json').read_text());d=data(a);m,ae,_=assets(c,a.device);initial=next(x for x in json.loads((a.output/'audit.json').read_text()) if x['split']=='test');rows.append(dict(method='Base',r=sel['r'],beta=None,feasible=True,converged=True,nominal=initial['nominal'],sensitivity=initial['sensitivity'],surrogate=initial['nominal']+sel['r']*initial['sensitivity'],current_validity=1.,pixel_cost=float((d['test']['base']-d['test']['factual']).flatten(1).norm(dim=1).mean())))
 torch.save(images,a.output/'frozen_images.pt');dump(a.output/'frozen_evaluation.json',{'images_sha256':sha(a.output/'frozen_images.pt'),'selection_sha256':sha(a.output/'selection.json')});f=score(a,c,images,'test');f.to_csv(a.output/'future_per_model.csv',index=False)
 for row in rows:
  v=f[f.method==row['method']];row.update(selected=row['method']==sel['method'],FV_mean=float(v.future_validity.mean()) if len(v) else None,FV_std=float(v.future_validity.std()) if len(v) else None,FV_min=float(v.future_validity.min()) if len(v) else None,worst_smooth_loss=float(v.smooth_loss.max()) if len(v) else None,common_r025_surrogate=row['nominal']+.25*row['sensitivity'])
 pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
 for p,h in json.loads((a.output/'input_manifest.json').read_text()).items():assert sha(Path(p))==h
 dump(a.output/'complete.json',{'input_hashes_verified':True,'selected_test_feasible':any(r['selected'] and r['feasible'] for r in rows),'no_validity_repair':True})

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','futures','merge','optimize','select','evaluate'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker',type=int,default=0);p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1);a=p.parse_args();c=json.loads(CONFIG.read_text());torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065);print('RUN',vars(a),'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True);globals()[a.phase](a,c)
if __name__=='__main__':main()
