"""Controlled Base margin manipulation with fixed-budget latent GRACE."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
import pandas as pd
import torch
from .stylemnist_grace_v1 import assets,load,sha,dump
from .mnist_update_aware import data,name
from .mnist_small_radius import futures,merge
from .mnist_weaker_classifier import prediction,collect
from ..counterfactuals.latent_boundary import first_sampled_crossing
from ..counterfactuals.update_aware import surrogate,differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
ROOT=Path(__file__).resolve().parents[4];CONFIG=ROOT/'experiments/configs/mnist_low_margin_base.json'

def prepare(a,c):
 parent=Path(c['parent_run']);manifest=json.loads((parent/'input_manifest.json').read_text())
 for p in [CONFIG,ROOT/c['task'],parent/'population.pt',parent/'resolved_config.json']:manifest[str(p)]=sha(p)
 for p,h in manifest.items():assert sha(Path(p))==h
 shutil.copyfile(parent/'population.pt',a.output/'population.pt');dump(a.output/'input_manifest.json',manifest);dump(a.output/'resolved_config.json',c)
 d=data(a);m,ae,_=assets(c,a.device)
 assert not set(d['cal']['ids'].tolist())&set(d['train_ids'].tolist()+d['target_ids'].tolist())
 with torch.no_grad():
  assert (m(d['target'].to(a.device)).sigmoid()>=.5).all()
  for split in ['cal','test']:
   assert (m(d[split]['factual'].to(a.device)).sigmoid()<.5).all();assert (m(d[split]['base'].to(a.device)).sigmoid()>=.5).all();assert torch.allclose(ae.decode(d[split]['base_z'].to(a.device)).cpu(),d[split]['base'],atol=1e-5,rtol=0)
 for arm,tau in c['conditions'].items():
  folder=a.output/arm;folder.mkdir();dump(folder/'resolved_config.json',{**c,'condition':arm,'target_probability':tau});dump(folder/'input_manifest.json',manifest)

def construct(a,c):
 parent=load(a.output.parent/'population.pt');p=copy.deepcopy(parent[a.split]);m,ae,_=assets(c,a.device);end=p['base_z'].to(a.device);tau=c['target_probability']
 with torch.no_grad():start=ae.encode(p['factual'].to(a.device))
 def probability(alpha):
  with torch.no_grad():
   z=start+alpha[...,None]*(end-start);shape=z.shape[:-1];flat=z.reshape(-1,64);return torch.cat([m(ae.decode(x)).sigmoid() for x in flat.split(64)]).reshape(shape)
 if tau is None:
  success=torch.ones(64,dtype=torch.bool);rows=[{'id':int(p['ids'][i]),'success':True,'reason':'original_control','alpha':1.} for i in range(64)]
 else:
  result=first_sampled_crossing(probability,64,tau,device=a.device,**c['boundary']);z=start+result['alpha'][:,None]*(end-start)
  with torch.no_grad():im=ae.decode(z);pr=m(im).sigmoid();success=result['success']&(pr>=tau)&(pr<=tau+1e-4)
  # Preserve the original image/code pair exactly for every fallback.
  p['base_z'][success.cpu()]=z[success].cpu();p['base'][success.cpu()]=im[success].cpu();rows=[]
  for i in range(64):
   row={'id':int(p['ids'][i]),'success':bool(success[i]),'reason':result['reason'][i] if bool(success[i])==bool(result['success'][i]) else 'recompute_tolerance_failure','tau':tau}
   row.update({k:float(v[i]) for k,v in result.items() if isinstance(v,torch.Tensor) and k!='success'});rows.append(row)
 p['construction_success']=success.cpu();p['historical_base']=parent[a.split]['base'];torch.save(p,a.output/f'population_{a.split}.pt')
 with torch.no_grad():pred=m(p['base'].to(a.device)).sigmoid();assert (pred>=.5).all()
 for i,row in enumerate(rows):row.update(probability=float(pred[i]),condition=c['condition'],split=a.split)
 pd.DataFrame(rows).to_csv(a.output/f'construction_{a.split}.csv',index=False)
 q=p['base_z'].to(a.device).requires_grad_();x=parent['x'].to(a.device);y=parent['y'].to(a.device);f,n,s=surrogate(m,x,y,q,.1,steps=3,lr=.01,decode=ae.decode,hinge_temperature=.05);g=torch.autograd.grad(f,q)[0];analytic=float(g.norm());direction=g.detach()/g.norm().clamp_min(1e-20);audit={'split':a.split,'nominal_hinge':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':1.,'directional_autograd':analytic};q=q.detach();del f,n,s,g
 values=[]
 for sign in [-1,1]:
  f,n,s=surrogate(m,x,y,q+sign*.001*direction,.1,steps=3,lr=.01,decode=ae.decode,hinge_temperature=.05);values.append(float(f.detach()));del f,n,s
 fd=(values[1]-values[0])/.002;audit.update(finite_difference=fd,relative_error=abs(fd-analytic)/max(analytic,1e-8),successful_constructions=int(success.sum()),adequate_manipulation=int(success.sum())>=48);assert audit['relative_error']<.15,audit;dump(a.output/f'audit_{a.split}.json',audit);print('CONSTRUCTION',c['condition'],audit,flush=True)

def assemble(a,c):
 d=load(a.output.parent/'population.pt');audits=[]
 for split in ['cal','test']:d[split]=load(a.output/f'population_{split}.pt');audits.append(json.loads((a.output/f'audit_{split}.json').read_text()));shutil.copyfile(a.output.parent/f'{split}_model_manifest.csv',a.output/f'{split}_model_manifest.csv')
 torch.save(d,a.output/'population.pt');dump(a.output/'base_recovery.json',audits)

def score(a,c,images,split):
 m,_,_=assets(c,a.device);pop=data(a);success=pop[split]['construction_success'].to(a.device);rows=[];individual=[]
 for rec in pd.read_csv(a.output/f'{split}_model_manifest.csv').to_dict('records'):
  assert sha(Path(rec['path']))==rec['sha256'];m.load_state_dict(load(Path(rec['path'])))
  with torch.no_grad():
   for method,im in images.items():
    p=m(im.to(a.device)).sigmoid();smooth=.05*torch.nn.functional.softplus((.5-p)/.05);rows.append({**rec,'method':method,'future_validity':float((p>=.5).float().mean()),'smooth_loss':float(smooth.mean()),'successful_subset_FV':float((p[success]>=.5).float().mean()) if success.any() else None,'successful_subset_n':int(success.sum())})
    for i in range(64):individual.append({'method':method,'family':rec['family'],'severity':rec['severity'],'seed':rec['seed'],'id':int(pop[split]['ids'][i]),'probability':float(p[i]),'construction_success':bool(success[i])})
 pd.DataFrame(individual).to_csv(a.output/f'{split}_individual_future.csv',index=False)
 return pd.DataFrame(rows)

def optimize(a,c):
 d=data(a);pop=d[a.split];m,ae,_=assets(c,a.device);x=d['x'].to(a.device);y=d['y'].to(a.device);base=pop['base_z'].to(a.device);target=d['target_z'].to(a.device);q=base.clone().requires_grad_();D=exact_w2(base.flatten(1).cpu(),target.flatten(1).cpu()).distance;budget=a.beta*D;tol=c['solver']['feas_relative']*max(1,D);dual=0.;prev=None;trace=[];start=time.perf_counter();stop_reason='fixed_800_updates';historical_stop=None
 folder=a.output/a.split/name(a.r,a.beta);folder.mkdir(parents=True,exist_ok=False)
 for step in range(c['solver']['steps']):
  f,nom,sens=surrogate(m,x,y,q,a.r,decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature'],steps=c['update']['steps'],lr=c['update']['lr']);d0,_=differentiable_w2(base,q);d1,_=differentiable_w2(q,target);violation=d0+d1-budget
  fv=float(f.detach());gv=float(violation.detach());
  if step==0:
   initial=next(v for v in json.loads((a.output/'base_recovery.json').read_text()) if v['split']==a.split)
   expected=initial['nominal_hinge']+a.r*initial['sensitivity'];assert abs(fv-expected)<1e-5*max(1,abs(expected)),(fv,expected)
   assert fv>0,'Smooth probability hinge is strictly positive; hard-hinge fallback detected'
  rel=None if prev is None else abs(fv-prev)/max(1,abs(prev));row={'step':step,'surrogate':fv,'nominal':float(nom.detach()),'sensitivity':float(sens.detach()),'g':gv,'dual':dual,'relative_objective_change':rel};trace.append(row)
  if prev is not None and gv<=tol and rel<=c['solver']['obj_relative'] and historical_stop is None:historical_stop=step
  if step%100==0:print(a.split,name(a.r,a.beta),row,flush=True)
  loss=f if getattr(a,'disable_geometry',False) else f+dual*violation+.5*c['solver']['rho']*torch.relu(violation).square();grad=torch.autograd.grad(loss,q)[0];assert torch.isfinite(grad).all()
  with torch.no_grad():q.copy_(q-c['solver']['primal_lr']*grad)
  newg=exact_w2(base.detach().flatten(1).cpu(),q.detach().flatten(1).cpu()).distance+exact_w2(q.detach().flatten(1).cpu(),target.detach().flatten(1).cpu()).distance-budget
  dual=0. if getattr(a,'disable_geometry',False) else max(0,dual+c['solver']['rho']*newg);prev=fv
 del f,nom,sens,d0,d1,violation
 q=q.detach();p0=exact_w2(base.flatten(1).cpu(),q.flatten(1).cpu());p1=exact_w2(q.flatten(1).cpu(),target.flatten(1).cpu());assert len(p0.source_index.unique())==64 and len(p0.target_index.unique())==64
 deployed=torch.empty_like(q);deployed[p0.source_index.to(q.device)]=q[p0.target_index.to(q.device)];torch.testing.assert_close(deployed[p0.source_index],q[p0.target_index],atol=0,rtol=0)
 q=q.requires_grad_()
 f,n,s=surrogate(m,x,y,q,a.r,decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature'],steps=c['update']['steps'],lr=c['update']['lr']);total=p0.distance+p1.distance
 final_gradient=torch.autograd.grad(f,q)[0];final_gradient_norm=float(final_gradient.norm());gd0,_=differentiable_w2(base,q);gd1,_=differentiable_w2(q,target);constraint_gradient=torch.autograd.grad(gd0+gd1,q)[0];constraint_gradient_norm=float(constraint_gradient.norm())
 decoded=ae.decode(deployed).detach()
 row={'split':a.split,'r':a.r,'beta':a.beta,'D':D,'budget':budget,'W2_Base_Q':p0.distance,'W2_Q_P1':p1.distance,'geometry_total':total,'feasible':total<=budget+tol,'geometry_tolerance':tol,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':float((m(decoded).sigmoid()>=.5).float().mean()),'pixel_cost':float((decoded-pop['factual'].to(a.device)).flatten(1).norm(dim=1).mean()),'runtime_seconds':time.perf_counter()-start,'iterations':len(trace),'assignment_preserves_Q':True,'assignment_cost_residual':abs(float((deployed-base).flatten(1).square().sum(1).mean().sqrt())-p0.distance),'final_objective_gradient_norm':final_gradient_norm,'converged':False,'historical_stop_step':historical_stop,'completed_updates':c['solver']['steps'],'final_objective_change':abs(float(f.detach())-prev),'final_constraint_gradient_norm':constraint_gradient_norm,'final_gradient_dot_constraint':float((final_gradient*constraint_gradient).sum()),'stop_reason':stop_reason}
 torch.save({'Q':q.detach().cpu(),'latents':deployed.cpu(),'images':decoded.cpu(),'assignment':p0.state_dict()},folder/'output.pt');pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);dump(folder/'metrics.json',row);print('OPT DONE',row,flush=True)

def select(a,c):
 rows,images=collect(a,'cal');f=score(a,c,images,'cal');f.to_csv(a.output/'calibration_per_model.csv',index=False)
 for row in rows:
  v=f[f.method==row['method']];row['cal_FV']=float(v.future_validity.mean()) if len(v) else None
 frame=pd.DataFrame(rows);frame.to_csv(a.output/'calibration_summary.csv',index=False);best=frame[frame.feasible].sort_values(['cal_FV','pixel_cost','r'],ascending=[False,True,True]).iloc[0].to_dict() if frame.feasible.any() else {'method':None,'r':0.,'no_feasible_calibration':True};dump(a.output/'selection.json',best)

def evaluate(a,c):
 rows,images=collect(a,'test');sel=json.loads((a.output/'selection.json').read_text());d=load(a.output/'population.pt');initial=next(x for x in json.loads((a.output/'base_recovery.json').read_text()) if x['split']=='test');rows.append(dict(method='Base',r=sel['r'],beta=None,feasible=True,converged=True,nominal=initial['nominal_hinge'],sensitivity=initial['sensitivity'],surrogate=initial['nominal_hinge']+sel['r']*initial['sensitivity'],current_validity=1.,pixel_cost=float((d['test']['base']-d['test']['factual']).flatten(1).norm(dim=1).mean())))
 torch.save(images,a.output/'frozen_images.pt');dump(a.output/'frozen_evaluation.json',{'images_sha256':sha(a.output/'frozen_images.pt'),'selection_sha256':sha(a.output/'selection.json')});f=score(a,c,images,'test');f.to_csv(a.output/'future_per_model.csv',index=False);strong,_,_=assets(json.loads(CONFIG.read_text()),a.device);dist=[]
 for method,im in images.items():
  for reference in ['Base','r0_beta1.02']:
   if reference not in images:continue
   delta=(im-images[reference]).flatten(1)
   for i in range(64):dist.append({'method':method,'reference':reference,'id':int(d['test']['ids'][i]),'pixel_L2':float(delta[i].norm()),'pixel_MAE':float(delta[i].abs().mean()),'pixel_max':float(delta[i].abs().max())})
 pd.DataFrame(dist).to_csv(a.output/'image_differences.csv',index=False)
 for row in rows:
  v=f[f.method==row['method']];row.update(selected=row['method']==sel['method'],FV_mean=float(v.future_validity.mean()) if len(v) else None,FV_min=float(v.future_validity.min()) if len(v) else None,common_r025_surrogate=row['nominal']+.25*row['sensitivity'],strong_classifier_agreement=float((prediction(strong,images[row['method']],a.device)>=.5).float().mean()) if row['method'] in images else None)
 pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
 for p,h in json.loads((a.output/'input_manifest.json').read_text()).items():assert sha(Path(p))==h
 dump(a.output/'complete.json',{'input_hashes_verified':True,'selected_test_feasible':any(r['selected'] and r['feasible'] for r in rows),'no_validity_repair':True})

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','construct','futures','merge','assemble','optimize','select','evaluate'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker',type=int,default=0);p.add_argument('--arm');p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1.02);a=p.parse_args();c=json.loads(CONFIG.read_text());torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065)
 if a.arm:a.output=a.output/a.arm;c=json.loads((a.output/'resolved_config.json').read_text())
 print('RUN',vars(a),'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True);globals()[a.phase](a,c)
if __name__=='__main__':main()
