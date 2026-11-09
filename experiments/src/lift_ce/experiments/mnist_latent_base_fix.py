"""Recover decoder-manifold Base codes; re-evaluate PDF Method without repairs."""
import argparse,json,time,os,shutil
from pathlib import Path
import pandas as pd
import torch
from .mnist_update_aware import data,model,update_model,name,score_images,select
from .stylemnist_grace_v1 import assets,load,sha,dump
from ..counterfactuals.vision import generate_near_boundary_ce
from ..counterfactuals.update_aware import surrogate,differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
from ..config import BaseCEConfig
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/mnist_latent_base_fix.json'

def prepare(a,c):
 parent=Path(c['parent_run']);d=load(parent/'population.pt');old=load(Path(c['old_run'])/'population.pt');m,ae,_=assets(c,a.device);checks=[]
 manifest=json.loads((parent/'input_manifest.json').read_text());manifest[str(parent/'population.pt')]=sha(parent/'population.pt')
 for split in ['cal','test']:
  # Replay the exact historical Base generator to recover its discarded codes.
  refs=d['target'] if split=='cal' else old['target'];p=d[split]
  result=generate_near_boundary_ce(m,ae,p['factual'],refs,BaseCEConfig(steps=96,autoencoder_type='strong',latent_dim=64),torch.device(a.device));assert torch.isfinite(result.latents).all()
  error=float((result.counterfactuals-p['base']).abs().max());assert error<1e-3,(split,error)
  d[split]['historical_base']=p['base'].clone();d[split]['base']=result.counterfactuals.clone()
  d[split]['previous_encoded_base_z']=p['base_z'];d[split]['base_z']=result.latents
  with torch.no_grad():
   decoded=ae.decode(result.latents.to(a.device));cv=float((m(decoded).sigmoid()>=.5).float().mean());pixel_error=float((decoded.cpu()-p['base']).abs().max())
  assert cv==1. and pixel_error<1e-5
  q=result.latents.to(a.device).requires_grad_();f,n,s=surrogate(m,d['x'].to(a.device),d['y'].to(a.device),q,.5,steps=c['update']['steps'],lr=c['update']['lr'],decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature']);g=torch.autograd.grad(f,q)[0]
  checks.append({'split':split,'replay_vs_historical_image_max_error':error,'historical_Base_replaced_by_explicit_generator_replay':True,'decode_original_image_max_error':pixel_error,'current_validity':cv,'nominal_hinge':float(n.detach()),'sensitivity':float(s.detach()),'r05_objective':float(f.detach()),'r05_gradient_norm':float(g.norm())})
  analytic=float(g.norm());direction=g.detach()/g.norm().clamp_min(1e-20);q0=q.detach();del f,n,s,g,q
  vals=[]
  for sign in [-1,1]:
   ff,nn,ss=surrogate(m,d['x'].to(a.device),d['y'].to(a.device),q0+sign*1e-3*direction,.5,steps=c['update']['steps'],lr=c['update']['lr'],decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature']);vals.append(float(ff.detach()));del ff,nn,ss
  fd=(vals[1]-vals[0])/.002;relative=abs(fd-analytic)/max(analytic,1e-8);checks[-1].update(finite_difference=fd,analytic_directional=analytic,relative_error=relative)
  assert relative<.15,(split,fd,analytic,relative)
 torch.save(d,a.output/'population.pt');dump(a.output/'base_recovery.json',checks);dump(a.output/'resolved_config.json',c)
 for split in ['cal','test']:
  file=parent/f'{split}_model_manifest.csv';shutil.copyfile(file,a.output/file.name);manifest[str(file)]=sha(file)
  for row in pd.read_csv(file).to_dict('records'):assert sha(Path(row['path']))==row['sha256'];manifest[row['path']]=row['sha256']
 dump(a.output/'input_manifest.json',manifest);print('BASE RECOVERY',checks,flush=True)

def optimize(a,c):
 d=data(a);pop=d[a.split];m,ae,_=assets(c,a.device);x=d['x'].to(a.device);y=d['y'].to(a.device);base=pop['base_z'].to(a.device);target=d['target_z'].to(a.device);q=base.clone().requires_grad_();D=exact_w2(base.flatten(1).cpu(),target.flatten(1).cpu()).distance;budget=a.beta*D;tol=c['solver']['feas_relative']*max(1,D);dual=0.;prev=None;trace=[];start=time.perf_counter();stop_reason='iteration_limit'
 folder=a.output/a.split/name(a.r,a.beta);folder.mkdir(parents=True,exist_ok=False)
 for step in range(c['solver']['steps']):
  f,nom,sens=surrogate(m,x,y,q,a.r,decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature'],steps=c['update']['steps'],lr=c['update']['lr']);d0,_=differentiable_w2(base,q);d1,_=differentiable_w2(q,target);violation=d0+d1-budget
  fv=float(f.detach());gv=float(violation.detach());
  if step==0:
   initial=next(v for v in json.loads((a.output/'base_recovery.json').read_text()) if v['split']==a.split)
   expected=initial['nominal_hinge']+a.r*initial['sensitivity'];assert abs(fv-expected)<1e-5*max(1,abs(expected)),(fv,expected)
   assert fv>0,'Smooth probability hinge is strictly positive; hard-hinge fallback detected'
  rel=None if prev is None else abs(fv-prev)/max(1,abs(prev));row={'step':step,'surrogate':fv,'nominal':float(nom.detach()),'sensitivity':float(sens.detach()),'g':gv,'dual':dual,'relative_objective_change':rel};trace.append(row)
  if fv<=1e-12 and gv<=tol:
   gradient=torch.autograd.grad(f,q,retain_graph=True)[0];row['objective_gradient_norm']=float(gradient.norm())
   if row['objective_gradient_norm']<=1e-10:stop_reason='zero_objective_global_lower_bound';break
  if step%10==0:print(a.split,name(a.r,a.beta),row,flush=True)
  if prev is not None and gv<=tol and rel<=c['solver']['obj_relative']:stop_reason='feasible_objective_tolerance';break
  loss=f+dual*violation+.5*c['solver']['rho']*torch.relu(violation).square();grad=torch.autograd.grad(loss,q)[0];assert torch.isfinite(grad).all()
  with torch.no_grad():q.copy_(q-c['solver']['primal_lr']*grad)
  newg=exact_w2(base.detach().flatten(1).cpu(),q.detach().flatten(1).cpu()).distance+exact_w2(q.detach().flatten(1).cpu(),target.detach().flatten(1).cpu()).distance-budget
  dual=max(0,dual+c['solver']['rho']*newg);prev=fv
 del f,nom,sens,d0,d1,violation
 q=q.detach();p0=exact_w2(base.flatten(1).cpu(),q.flatten(1).cpu());p1=exact_w2(q.flatten(1).cpu(),target.flatten(1).cpu());assert len(p0.source_index.unique())==64 and len(p0.target_index.unique())==64
 deployed=torch.empty_like(q);deployed[p0.source_index.to(q.device)]=q[p0.target_index.to(q.device)];torch.testing.assert_close(deployed[p0.source_index],q[p0.target_index],atol=0,rtol=0)
 q=q.requires_grad_()
 f,n,s=surrogate(m,x,y,q,a.r,decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature'],steps=c['update']['steps'],lr=c['update']['lr']);total=p0.distance+p1.distance
 final_gradient=torch.autograd.grad(f,q)[0];final_gradient_norm=float(final_gradient.norm())
 decoded=ae.decode(deployed).detach()
 row={'split':a.split,'r':a.r,'beta':a.beta,'D':D,'budget':budget,'W2_Base_Q':p0.distance,'W2_Q_P1':p1.distance,'geometry_total':total,'feasible':total<=budget+tol,'geometry_tolerance':tol,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':float((m(decoded).sigmoid()>=.5).float().mean()),'pixel_cost':float((decoded-pop['factual'].to(a.device)).flatten(1).norm(dim=1).mean()),'runtime_seconds':time.perf_counter()-start,'iterations':len(trace),'assignment_preserves_Q':True,'assignment_cost_residual':abs(float((deployed-base).flatten(1).square().sum(1).mean().sqrt())-p0.distance),'final_objective_gradient_norm':final_gradient_norm,'converged':stop_reason!='iteration_limit','stop_reason':stop_reason}
 torch.save({'Q':q.detach().cpu(),'latents':deployed.cpu(),'images':decoded.cpu(),'assignment':p0.state_dict()},folder/'output.pt');pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);dump(folder/'metrics.json',row);print('OPT DONE',row,flush=True)

def evaluate(a,c):
 d=data(a);m,ae,_=assets(c,a.device);selection=json.loads((a.output/'selection.json').read_text());images={'Base':d['test']['base'],'Historical-Base':d['test']['historical_base']};rows=[]
 for p in sorted((a.output/'test').glob('*/metrics.json')):
  row=json.loads(p.read_text());row['method']=p.parent.name;row['selected']=row['r']==selection['r'] and row['beta']==selection['beta'];rows.append(row)
  if row['feasible']:images[p.parent.name]=load(p.parent/'output.pt')['images']
 for r in c['r_grid']:
  q=d['test']['base_z'].to(a.device);f,n,s=surrogate(m,d['x'].to(a.device),d['y'].to(a.device),q,r,steps=c['update']['steps'],lr=c['update']['lr'],decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature'])
  if r==selection['r']:
   with torch.no_grad():decoded=ae.decode(q);cv=float((m(decoded).sigmoid()>=.5).float().mean())
   rows.append({'method':'Base','r':r,'beta':None,'selected':False,'feasible':True,'converged':True,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':cv,'pixel_cost':float((d['test']['base']-d['test']['factual']).flatten(1).norm(dim=1).mean())})
  del f,n,s
 torch.save(images,a.output/'frozen_images.pt');dump(a.output/'frozen_evaluation.json',{'images_sha256':sha(a.output/'frozen_images.pt'),'selection_sha256':sha(a.output/'selection.json')})
 scores=score_images(m,images,pd.read_csv(a.output/'test_model_manifest.csv'),a.device);scores.to_csv(a.output/'future_per_model.csv',index=False)
 for row in rows:
  vals=scores[(scores.method==row['method'])&(scores.severity==1)].future_validity;row.update(FV_mean=float(vals.mean()) if len(vals) else None,FV_std=float(vals.std()) if len(vals) else None)
 pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
 for path,h in json.loads((a.output/'input_manifest.json').read_text()).items():assert sha(Path(path))==h
 dump(a.output/'complete.json',{'input_hashes_verified':True,'no_validity_repair':True,'selection_frozen_before_test':True});print(pd.DataFrame(rows).to_string(index=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','optimize','select','evaluate'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1);a=p.parse_args();c=json.loads(CONFIG.read_text());torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065)
 print('RUN',vars(a),'CUDA',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
 {'prepare':prepare,'optimize':optimize,'select':select,'evaluate':evaluate}[a.phase](a,c)
if __name__=='__main__':main()
