"""MNIST implementation of the revised PDF's update-aware method."""
import argparse,copy,hashlib,json,time,os,socket
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from torchvision.datasets import MNIST
from ..counterfactuals.update_aware import surrogate,unroll,differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
from ..counterfactuals.vision import generate_near_boundary_ce
from ..config import BaseCEConfig
from .stylemnist_grace_v1 import assets,shifted_images,load,sha,dump
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/mnist_update_aware.json'
def data(a):return load(a.output/'population.pt')
def model(a,c):return assets(c,a.device)[0]
def prepare(a,c):
 out=a.output;old=Path(c['old_run']);p=load(old/'population.pt');m,ae,_=assets(c,a.device)
 originalids=np.load(old/'train_ids.npy');calids=np.load(old/'calibration_ids.npy');testids=np.load(old/'factual_ids.npy')
 chosen=torch.cat([torch.where(p['y']==label)[0][:128] for label in [0,1]])
 x=p['x'][chosen];y=p['y'][chosen]
 raw=MNIST(c['data_dir'],train=True,download=False)
 cx=raw.data[calids,None].float()/255;cy=raw.targets[calids]
 with torch.no_grad():scores=m(cx.to(a.device)).sigmoid().cpu()
 ids=torch.where((cy==4)&(scores<.5))[0][:64];assert len(ids)==64
 cfact=cx[ids];factualids=calids[ids.numpy()]
 with torch.no_grad():good=m(p['x'].to(a.device)).sigmoid().cpu()>=.5
 refs=torch.where((p['y']==1)&good)[0][:64];target=p['x'][refs];assert len(target)==64
 cfg=BaseCEConfig(steps=96,autoencoder_type='strong',latent_dim=64)
 t=time.perf_counter();cb=generate_near_boundary_ce(m,ae,cfact,target,cfg,torch.device(a.device)).counterfactuals.cpu();sec=time.perf_counter()-t
 d={'x':x,'y':y,'target':target,'cal':{'factual':cfact,'base':cb,'ids':torch.tensor(factualids)},'test':{'factual':p['factual'],'base':p['base'],'ids':torch.tensor(testids)},'train_ids':torch.tensor(originalids[chosen.numpy()]),'target_ids':torch.tensor(originalids[refs.numpy()])}
 
 with torch.no_grad():
  d['target_z']=ae.encode(target.to(a.device)).cpu()
  for split in ['cal','test']:d[split]['base_z']=ae.encode(d[split]['base'].to(a.device)).cpu()
 torch.save(d,out/'population.pt');paths=[Path(c['checkpoint']),old/'population.pt',old/'train_ids.npy',old/'calibration_ids.npy',old/'factual_ids.npy',ROOT/'paper/iclr_robust_ce-62.pdf']
 manifest={str(p):sha(p) for p in paths};dump(out/'input_manifest.json',manifest);dump(out/'resolved_config.json',c)
 assert not set(originalids[chosen.numpy()])&set(factualids)
 checks={'train_n':len(x),'cal_n':64,'test_n':64,'target_n':64,'pixel_space':False,'latent_dim':64,'target_current_validity':float((m(target.to(a.device)).sigmoid()>=.5).float().mean()),'cal_base_seconds':sec,'disjoint_cal_train_ids':True,'cal_test_disjoint_by_official_MNIST_split':True,'checkpoint_training_split_provenance':'historical source defaults, exact original invocation unknown'}
 with torch.no_grad():
  for split in ['cal','test']:
   bb=d[split]['base'].to(a.device);dec=ae.decode(d[split]['base_z'].to(a.device));checks[split+'_original_base_validity']=float((m(bb).sigmoid()>=.5).float().mean());checks[split+'_decoded_base_validity']=float((m(dec).sigmoid()>=.5).float().mean());checks[split+'_decode_pixel_gap']=float((dec-bb).flatten(1).norm(dim=1).mean())
 dump(out/'assets.json',checks);print('PREPARED',checks,flush=True)

def update_model(m,x,y,c):
 p,b=unroll(m,x,y,c['update']['steps'],c['update']['lr'],False);new=copy.deepcopy(m);new.load_state_dict({**{k:v.detach() for k,v in p.items()},**b});return new.eval().requires_grad_(False)

def smoke(a,c):
 d=data(a);m,ae,_=assets(c,a.device);x=d['x'].to(a.device);y=d['y'].to(a.device);q=d['cal']['base_z'].to(a.device).requires_grad_();t=time.perf_counter()
 f,n,s=surrogate(m,x,y,q,.5,decode=ae.decode,**{k:c['update'][k] for k in ['steps','lr']});g=torch.autograd.grad(f,q)[0]
 assert torch.isfinite(g).all();metrics={'objective':float(f),'nominal':float(n),'sensitivity':float(s),'outer_gradient_norm':float(g.norm()),'seconds':time.perf_counter()-t,'peak_gpu_GB':torch.cuda.max_memory_allocated()/1e9}
 # Validate a finite-difference direction in the actual nonsmooth network; use a diagnostic tolerance.
 v=g.detach()/g.norm().clamp_min(1e-20);eps=1e-3;vals=[]
 for sign in [-1,1]:vals.append(float(surrogate(m,x,y,q.detach()+sign*eps*v,.5,decode=ae.decode,steps=c['update']['steps'],lr=c['update']['lr'])[0].detach()))
 metrics['outer_directional_finite_difference']=(vals[1]-vals[0])/(2*eps);metrics['outer_directional_autograd']=float((g*v).sum());metrics['directional_relative_error']=abs(metrics['outer_directional_finite_difference']-metrics['outer_directional_autograd'])/max(abs(metrics['outer_directional_autograd']),1e-8)
 if metrics['directional_relative_error']>.1:raise ValueError(metrics)
 u=update_model(m,x,y,c);vmodel=update_model(m,x,y,c);assert all(torch.equal(u.state_dict()[k],vmodel.state_dict()[k]) for k in u.state_dict());metrics['update_reproducible']=True
 dump(a.output/'smoke.json',metrics);print('SMOKE',metrics,flush=True)

def train_futures(a,c):
 d=data(a);m=model(a,c);folder=a.output/f'{a.split}_models';folder.mkdir(exist_ok=True);rows=[];future=c['future'];severities=[1] if a.split=='cal' else future['severities'];seed0=future[f'{a.split}_seed']
 for family,strength in zip(future['families'],future['strengths']):
  for severity in severities:
   for rep in range(future['realizations']):
    seed=seed0+rep;xx=shifted_images(d['x'],family,strength*severity,seed).to(a.device);t=time.perf_counter();u=update_model(m,xx,d['y'].to(a.device),c);file=folder/f'{family}_{severity:g}_{rep}.pt';torch.save(u.state_dict(),file)
    rows.append({'family':family,'severity':severity,'realization':rep,'seed':seed,'path':str(file),'sha256':sha(file),'seconds':time.perf_counter()-t,'duplicate_distribution':family=='translation' or severity==0})
 pd.DataFrame(rows).to_csv(a.output/f'{a.split}_model_manifest.csv',index=False);print('FUTURE MODELS',a.split,len(rows),flush=True)

def name(r,b):return f'r{r:g}_beta{b:g}'
def optimize(a,c):
 d=data(a);pop=d[a.split];m,ae,_=assets(c,a.device);x=d['x'].to(a.device);y=d['y'].to(a.device);base=pop['base_z'].to(a.device);target=d['target_z'].to(a.device);q=base.clone().requires_grad_();D=exact_w2(base.flatten(1).cpu(),target.flatten(1).cpu()).distance;budget=a.beta*D;tol=c['solver']['feas_relative']*max(1,D);dual=0.;prev=None;trace=[];start=time.perf_counter()
 folder=a.output/a.split/name(a.r,a.beta);folder.mkdir(parents=True,exist_ok=False)
 for step in range(c['solver']['steps']):
  f,nom,sens=surrogate(m,x,y,q,a.r,decode=ae.decode,steps=c['update']['steps'],lr=c['update']['lr']);d0,_=differentiable_w2(base,q);d1,_=differentiable_w2(q,target);violation=d0+d1-budget
  fv=float(f.detach());gv=float(violation.detach());rel=None if prev is None else abs(fv-prev)/max(1,abs(prev));row={'step':step,'surrogate':fv,'nominal':float(nom.detach()),'sensitivity':float(sens.detach()),'g':gv,'dual':dual,'relative_objective_change':rel};trace.append(row)
  if step%10==0:print(a.split,name(a.r,a.beta),row,flush=True)
  if prev is not None and gv<=tol and rel<=c['solver']['obj_relative']:break
  loss=f+dual*violation+.5*c['solver']['rho']*torch.relu(violation).square();grad=torch.autograd.grad(loss,q)[0];assert torch.isfinite(grad).all()
  with torch.no_grad():q.copy_(q-c['solver']['primal_lr']*grad)
  newg=exact_w2(base.detach().flatten(1).cpu(),q.detach().flatten(1).cpu()).distance+exact_w2(q.detach().flatten(1).cpu(),target.detach().flatten(1).cpu()).distance-budget
  dual=max(0,dual+c['solver']['rho']*newg);prev=fv
 q=q.detach();p0=exact_w2(base.flatten(1).cpu(),q.flatten(1).cpu());p1=exact_w2(q.flatten(1).cpu(),target.flatten(1).cpu());assert len(p0.source_index.unique())==64 and len(p0.target_index.unique())==64
 deployed=torch.empty_like(q);deployed[p0.source_index.to(q.device)]=q[p0.target_index.to(q.device)];torch.testing.assert_close(deployed[p0.source_index],q[p0.target_index],atol=0,rtol=0)
 f,n,s=surrogate(m,x,y,q,a.r,decode=ae.decode,steps=c['update']['steps'],lr=c['update']['lr']);total=p0.distance+p1.distance
 decoded=ae.decode(deployed).detach()
 row={'split':a.split,'r':a.r,'beta':a.beta,'D':D,'budget':budget,'W2_Base_Q':p0.distance,'W2_Q_P1':p1.distance,'geometry_total':total,'feasible':total<=budget+tol,'geometry_tolerance':tol,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':float((m(decoded).sigmoid()>=.5).float().mean()),'pixel_cost':float((decoded-pop['factual'].to(a.device)).flatten(1).norm(dim=1).mean()),'runtime_seconds':time.perf_counter()-start,'iterations':len(trace),'assignment_preserves_Q':True,'assignment_cost_residual':abs(float((deployed-base).flatten(1).square().sum(1).mean().sqrt())-p0.distance),'converged':prev is not None and trace[-1]['g']<=tol and trace[-1]['relative_objective_change'] is not None and trace[-1]['relative_objective_change']<=c['solver']['obj_relative']}
 torch.save({'Q':q.cpu(),'latents':deployed.cpu(),'images':decoded.cpu(),'assignment':p0.state_dict()},folder/'output.pt');pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);dump(folder/'metrics.json',row);print('OPT DONE',row,flush=True)

def score_images(m,images,manifest,device):
 rows=[]
 for rec in manifest.to_dict('records'):
  assert sha(Path(rec['path']))==rec['sha256'];m.load_state_dict(load(Path(rec['path'])))
  with torch.no_grad():
   for method,im in images.items():
    pred=m(im.to(device)).sigmoid();rows.append({**rec,'method':method,'future_validity':float((pred>=.5).float().mean()),'hinge':float(torch.relu(.5-pred).mean())})
 return pd.DataFrame(rows)

def select(a,c):
 images={};metrics=[]
 for p in sorted((a.output/'cal').glob('*/metrics.json')):
  row=json.loads(p.read_text());metrics.append(row)
  if row['feasible']:images[p.parent.name]=load(p.parent/'output.pt')['images']
 if not images:raise RuntimeError('All calibration final iterates infeasible; no valid selection.')
 scores=score_images(model(a,c),images,pd.read_csv(a.output/'cal_model_manifest.csv'),a.device);scores.to_csv(a.output/'calibration_per_model.csv',index=False)
 for row in metrics:row['method']=name(row['r'],row['beta']);row['cal_FV']=float(scores[scores.method==row['method']].future_validity.mean()) if row['feasible'] else None
 frame=pd.DataFrame(metrics);frame.to_csv(a.output/'calibration_summary.csv',index=False);best=frame[frame.feasible].sort_values(['cal_FV','pixel_cost','r','beta'],ascending=[False,True,True,True]).iloc[0].to_dict();dump(a.output/'selection.json',best);print('SELECTION',best,flush=True)

def evaluate(a,c):
 d=data(a);m,ae,_=assets(c,a.device);images={'Base':d['test']['base'],'Base-decoded':ae.decode(d['test']['base_z'].to(a.device)).detach().cpu()};rows=[];selection=json.loads((a.output/'selection.json').read_text())
 for p in sorted((a.output/'test').glob('*/metrics.json')):
  row=json.loads(p.read_text());row['method']=p.parent.name;row['selected']=row['r']==selection['r'] and row['beta']==selection['beta'];rows.append(row)
  if row['feasible']:images[p.parent.name]=load(p.parent/'output.pt')['images']
 if not any(r['selected'] and r['feasible'] for r in rows):print('WARNING selected configuration infeasible on test; report failure',flush=True)
 for label in ['Base','Base-decoded']:
  base=images[label].to(a.device);f,n,s=surrogate(m,d['x'].to(a.device),d['y'].to(a.device),base,selection['r'],steps=c['update']['steps'],lr=c['update']['lr'])
  rows.append({'method':label,'selected':False,'feasible':True,'r':selection['r'],'beta':None,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':float((m(base).sigmoid()>=.5).float().mean()),'pixel_cost':float((base-d['test']['factual'].to(a.device)).flatten(1).norm(dim=1).mean())})
 torch.save(images,a.output/'frozen_images.pt');dump(a.output/'frozen_evaluation.json',{'images_sha256':sha(a.output/'frozen_images.pt'),'selection_sha256':sha(a.output/'selection.json')})
 scores=score_images(m,images,pd.read_csv(a.output/'test_model_manifest.csv'),a.device);scores.to_csv(a.output/'future_per_model.csv',index=False)
 for row in rows:
  vals=scores[(scores.method==row['method'])&(scores.severity==1)].future_validity;row.update(FV_mean=float(vals.mean()) if len(vals) else None,FV_std=float(vals.std()) if len(vals) else None)
 pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
 for path,h in json.loads((a.output/'input_manifest.json').read_text()).items():assert sha(Path(path))==h
 dump(a.output/'complete.json',{'all_input_hashes_unchanged':True,'selection_frozen_before_final_scoring':True,'no_current_validity_repair':True,'one_CF_per_factual':True,'method':'PDF Section 3 update-aware surrogate','selected_test_feasible':any(r['selected'] and r['feasible'] for r in rows)})
 print(pd.DataFrame(rows).to_string(index=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','smoke','futures','optimize','select','evaluate'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--split',choices=['cal','test'],default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1);a=p.parse_args();c=json.loads(CONFIG.read_text());torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065)
 print('RUN',vars(a),'host',socket.gethostname(),'CUDA',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
 {'prepare':prepare,'smoke':smoke,'futures':train_futures,'optimize':optimize,'select':select,'evaluate':evaluate}[a.phase](a,c)
if __name__=='__main__':main()
