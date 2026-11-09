"""HELOC: exact PDF-style support refinement with update-aware smooth hinge."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from ..counterfactuals.update_aware import surrogate,unroll,differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
from .heloc_diabetes_official_lift_v1 import load_model,predictive_features
from .diabetes_original_lift_comparison import shift
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/heloc_smooth_hinge.json'
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def current(c,device):return load_model(Path(c['raw_parent'])/'current.pt',22).to(device).eval()
def data(a):return load(a.output/'population.pt')
def key(r,b):return f'r{r:g}_beta{b:g}'
def objective(m,d,q,r,c):return surrogate(m,d['x'],d['y'],q,r,steps=c['update']['steps'],lr=c['update']['lr'],hinge_temperature=c['smooth_hinge']['temperature'])
def updated(m,x,y,c):
 params,buffers=unroll(m,x,y,steps=c['update']['steps'],lr=c['update']['lr'],differentiable=False);u=copy.deepcopy(m);u.load_state_dict({**{k:v.detach() for k,v in params.items()},**buffers});return u.eval().requires_grad_(False)
def predict(m,x):
 with torch.no_grad():return m(x).sigmoid()

def prepare(a,c):
 raw=Path(c['raw_parent']);old=Path(c['population_parent']);meta=read(raw/'instance_manifest.json');tr=load(raw/'training.pt');audit=load(raw/'data_audit.pt');m=current(c,a.device);manifest={};pop={'x':tr['x'],'y':tr['y'],'train_ids':tr['ids'],'target':tr['target'],'target_ids':tr['target_ids'],'b_ref':meta['b_ref'],'feature_names':meta['feature_names'],'lower':tr['x'].min(0).values,'upper':tr['x'].max(0).values};checks=[]
 assert tr['x'].shape==(6000,22) and tr['target'].shape==(3336,22);assert sha(raw/'current.pt')==sha(old/'current.pt');assert torch.equal(tr['x'],audit['train_x']) and torch.equal(tr['y'],audit['train_y'])
 assert (predict(m,tr['target'].to(a.device))>=.5).all()
 for split in ['cal','test']:
  source=load(old/f'{split}_population.pt');base=source['path']['base'];assert base.shape==(128,22);assert (predict(m,base.to(a.device))>=.5).all();assert len(source['source_ids'].unique())==128;assert not set(source['source_ids'].tolist())&set(tr['ids'].tolist())
  pop[split]={'base':base,'factual':source['source'],'ids':source['source_ids']};checks.append({'split':split,'factuals':128,'current_base_validity':1.,'D':exact_w2(base,tr['target']).distance})
  path=old/f'{split}_population.pt';manifest[str(path)]=sha(path)
 assert not set(pop['cal']['ids'].tolist())&set(pop['test']['ids'].tolist())
 for name in ['current.pt','training.pt','data_audit.pt','instance_manifest.json']:manifest[str(raw/name)]=sha(raw/name)
 for p in [ROOT/'paper/iclr_robust_ce-62.pdf',ROOT/c['task'],CONFIG]:manifest[str(p)]=sha(p)
 for split,oldname in [('cal','calibration'),('test','final')]:
  path=Path(meta['original_instance'])/f'future_{oldname}_manifest.csv';spec=pd.read_csv(path);assert len(spec)==45;spec=spec.drop(columns=['path']).rename(columns={'split':'historical_split'});spec['split']=split;spec.to_csv(a.output/f'{split}_future_spec.csv',index=False);manifest[str(path)]=sha(path)
 torch.save(pop,a.output/'population.pt');c={**c,'b_ref':meta['b_ref'],'r_grid':[r*meta['b_ref'] for r in c['r_ratios']]};dump(a.output/'resolved_config.json',c);dump(a.output/'input_manifest.json',manifest)
 d={k:pop[k].to(a.device) for k in ['x','y']};initial=[]
 for split in ['cal','test']:
  q=pop[split]['base'].to(a.device).requires_grad_();f,n,s=objective(m,d,q,meta['b_ref'],c);g=torch.autograd.grad(f,q)[0];analytic=float(g.norm());v=g.detach()/g.norm().clamp_min(1e-20);row={'split':split,'nominal':float(n.detach()),'sensitivity':float(s.detach()),'objective_at_b_ref':float(f.detach()),'gradient_norm':analytic};q0=q.detach();del f,n,s,g,q
  vals=[]
  for sign in [-1,1]:
   ff,nn,ss=objective(m,d,q0+sign*1e-3*v,meta['b_ref'],c);vals.append(float(ff.detach()));del ff,nn,ss
  fd=(vals[1]-vals[0])/.002;error=abs(fd-analytic)/max(analytic,1e-8);assert error<.15,(split,fd,analytic);row.update(finite_difference=fd,relative_error=error);initial.append(row)
 nominal=updated(m,d['x'],d['y'],c);again=updated(m,d['x'],d['y'],c);assert all(torch.equal(nominal.state_dict()[k],again.state_dict()[k]) for k in nominal.state_dict())
 modeldiag={}
 for label,model in [('f0',m),('nominal_updated',nominal)]:
  for split in ['train','cal']:
   xx=audit[f'{split}_x'].to(a.device);yy=audit[f'{split}_y'].to(a.device);prob=predict(model,xx);modeldiag[f'{label}_{split}_accuracy']=float(((prob>=.5)==yy.bool()).float().mean())
 dump(a.output/'initial_objectives.json',initial);dump(a.output/'audit.json',{'populations':checks,'P1_favorable':True,'cal_test_train_ids_disjoint':True,'full_training_count':6000,'features':22,'Q_count':128,'P1_count':3336,'update_bitwise_reproducible':True,'model_diagnostics':modeldiag,'actionability':'all features mutable continuous relaxation; no box/integer/immutable constraints','source_hashes_preserved':True});print('PREPARED',initial,modeldiag,flush=True)

def futures(a,c):
 d=data(a);m=current(c,a.device);features=predictive_features(d['x'],d['y']);records=[];folder=a.output/'future_models';folder.mkdir(exist_ok=True)
 specs=pd.concat([pd.read_csv(a.output/f'{s}_future_spec.csv') for s in ['cal','test']],ignore_index=True)
 for i,rec in enumerate(specs.to_dict('records')):
  if i%8!=a.worker:continue
  sx,sy,kind=shift(rec['environment'],d['x'],d['y'],rec['severity_control'],int(rec['realization']),int(rec['seed']),d['b_ref'],features);start=time.perf_counter();u=updated(m,sx.to(a.device),sy.to(a.device),c);path=folder/f'model{i:03d}.pt';assert not path.exists();torch.save({k:v.cpu() for k,v in u.state_dict().items()},path)
  records.append({**rec,'kind':kind,'path':str(path),'sha256':sha(path),'seconds':time.perf_counter()-start,'update_rows':len(sx),'shift_tensor_sha256':hashlib.sha256(sx.numpy().tobytes()+sy.numpy().tobytes()).hexdigest()})
 pd.DataFrame(records).to_csv(a.output/f'future_worker{a.worker}.csv',index=False);print('FUTURE WORKER',a.worker,len(records),flush=True)
def merge_futures(a,c):
 frame=pd.concat([pd.read_csv(a.output/f'future_worker{i}.csv') for i in range(8)],ignore_index=True);assert len(frame)==90 and frame.path.nunique()==90
 for split in ['cal','test']:
  f=frame[frame.split==split].sort_values('future_model_id');assert len(f)==45;f.to_csv(a.output/f'{split}_model_manifest.csv',index=False)
 dump(a.output/'future_checks.json',{'count':90,'cal_distinct_data_distributions':int(frame[frame.split=='cal'].shift_tensor_sha256.nunique()),'test_distinct_data_distributions':int(frame[frame.split=='test'].shift_tensor_sha256.nunique()),'cross_split_identical_data_distributions':len(set(frame[frame.split=='cal'].shift_tensor_sha256)&set(frame[frame.split=='test'].shift_tensor_sha256))})

def optimize(a,c):
 pop=data(a);d={k:pop[k].to(a.device) for k in ['x','y']};m=current(c,a.device);base=pop[a.split]['base'].to(a.device);target=pop['target'].to(a.device);q=base.clone().requires_grad_();D=exact_w2(base.cpu(),target.cpu()).distance;budget=a.beta*D;tol=c['solver']['feas_relative']*max(1,D);dual=0.;prev=None;trace=[];reason='iteration_limit';start=time.perf_counter();folder=a.output/a.split/key(a.r,a.beta);folder.mkdir(parents=True,exist_ok=False)
 for step in range(c['solver']['steps']):
  f,nom,sens=objective(m,d,q,a.r,c);d0,_=differentiable_w2(base,q);d1,_=differentiable_w2(q,target);violation=d0+d1-budget;fv=float(f.detach());gv=float(violation.detach())
  if step==0:
   initial=next(v for v in read(a.output/'initial_objectives.json') if v['split']==a.split);expected=initial['nominal']+a.r*initial['sensitivity'];assert fv>0 and abs(fv-expected)<1e-5*max(1,expected),(fv,expected)
  rel=None if prev is None else abs(fv-prev)/max(1,abs(prev));row={'step':step,'surrogate':fv,'nominal':float(nom.detach()),'sensitivity':float(sens.detach()),'g':gv,'dual':dual,'objective_change':rel};trace.append(row)
  if step%50==0:print(a.split,key(a.r,a.beta),row,flush=True)
  if prev is not None and gv<=tol and rel<=c['solver']['obj_relative']:reason='feasible_objective_tolerance';break
  loss=f+dual*violation+.5*c['solver']['rho']*torch.relu(violation).square();grad=torch.autograd.grad(loss,q)[0];assert torch.isfinite(grad).all()
  with torch.no_grad():q.sub_(c['solver']['primal_lr']*grad)
  newg=exact_w2(base.cpu(),q.detach().cpu()).distance+exact_w2(q.detach().cpu(),target.cpu()).distance-budget;dual=max(0,dual+c['solver']['rho']*newg);prev=fv
 del f,nom,sens,d0,d1,violation
 q=q.detach().requires_grad_();p0=exact_w2(base.cpu(),q.detach().cpu());p1=exact_w2(q.detach().cpu(),target.cpu());assert len(p0.source_index.unique())==len(p0.target_index.unique())==len(base)
 deployed=torch.empty_like(q);deployed[p0.source_index]=q[p0.target_index];f,n,s=objective(m,d,q,a.r,c);gradient=torch.autograd.grad(f,q)[0];total=p0.distance+p1.distance;lower=pop['lower'].to(a.device);upper=pop['upper'].to(a.device);source=pop[a.split]['factual'].to(a.device)
 row={'split':a.split,'r':a.r,'r_ratio':a.r/c['b_ref'],'beta':a.beta,'D':D,'budget':budget,'W2_Base_Q':p0.distance,'W2_Q_P1':p1.distance,'geometry_total':total,'feasible':total<=budget+tol,'geometry_tolerance':tol,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':float((predict(m,deployed)>=.5).float().mean()),'recourse_cost':float((deployed-source).norm(dim=1).mean()),'base_displacement':float((deployed-base).norm(dim=1).mean()),'outside_training_range_fraction':float(((deployed<lower)|(deployed>upper)).any(1).float().mean()),'runtime_seconds':time.perf_counter()-start,'iterations':len(trace),'assignment_preserves_Q':True,'assignment_cost_residual':abs(float((deployed-base).square().sum(1).mean().sqrt())-p0.distance),'final_objective_gradient_norm':float(gradient.norm()),'converged':reason!='iteration_limit','stop_reason':reason}
 torch.save({'Q':q.detach().cpu(),'deployed':deployed.detach().cpu(),'assignment':p0.state_dict()},folder/'output.pt');pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);dump(folder/'metrics.json',row);print('OPT DONE',row,flush=True)

def score(a,c,points,split):
 m=current(c,a.device);rows=[]
 for rec in pd.read_csv(a.output/f'{split}_model_manifest.csv').to_dict('records'):
  assert sha(rec['path'])==rec['sha256'];m.load_state_dict(load(rec['path']))
  for method,q in points.items():
   p=predict(m,q.to(a.device));smooth=c['smooth_hinge']['temperature']*torch.nn.functional.softplus((.5-p)/c['smooth_hinge']['temperature']);rows.append({**rec,'method':method,'future_validity':float((p>=.5).float().mean()),'smooth_loss':float(smooth.mean())})
 return pd.DataFrame(rows)
def select(a,c):
 metrics=[];points={}
 for path in sorted((a.output/'cal').glob('*/metrics.json')):
  r=read(path);r['method']=path.parent.name;metrics.append(r)
  if r['feasible']:points[r['method']]=load(path.parent/'output.pt')['deployed']
 assert points,'No feasible final calibration iterates';f=score(a,c,points,'cal');f.to_csv(a.output/'calibration_per_model.csv',index=False)
 for row in metrics:row['cal_FV']=float(f[f.method==row['method']].future_validity.mean()) if row['feasible'] else None
 frame=pd.DataFrame(metrics);frame.to_csv(a.output/'calibration_summary.csv',index=False);best=frame[frame.feasible].sort_values(['cal_FV','recourse_cost','r','beta'],ascending=[False,True,True,True]).iloc[0].to_dict();dump(a.output/'selection.json',best);print('SELECTION',best,flush=True)
def evaluate(a,c):
 pop=data(a);m=current(c,a.device);sel=read(a.output/'selection.json');points={'Base':pop['test']['base']};rows=[]
 for path in sorted((a.output/'test').glob('*/metrics.json')):
  row=read(path);row['method']=path.parent.name;row['selected']=row['r']==sel['r'] and row['beta']==sel['beta'];rows.append(row)
  if row['feasible']:points[row['method']]=load(path.parent/'output.pt')['deployed']
 d={k:pop[k].to(a.device) for k in ['x','y']};q=points['Base'].to(a.device);f,n,s=objective(m,d,q,sel['r'],c);rows.append({'method':'Base','r':sel['r'],'r_ratio':sel['r']/c['b_ref'],'beta':None,'selected':False,'feasible':True,'converged':True,'current_validity':float((predict(m,q)>=.5).float().mean()),'recourse_cost':float((q-pop['test']['factual'].to(a.device)).norm(dim=1).mean()),'base_displacement':0.,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'outside_training_range_fraction':float(((q<pop['lower'].to(a.device))|(q>pop['upper'].to(a.device))).any(1).float().mean())});del f,n,s
 torch.save(points,a.output/'frozen_outputs.pt');dump(a.output/'evaluation_frozen.json',{'output_sha256':sha(a.output/'frozen_outputs.pt'),'selection_sha256':sha(a.output/'selection.json')});f=score(a,c,points,'test');f.to_csv(a.output/'future_per_model.csv',index=False)
 for row in rows:
  group=f[f.method==row['method']];row.update(FV_mean=float(group.future_validity.mean()) if len(group) else None,FV_std=float(group.future_validity.std()) if len(group) else None,FV_min=float(group.future_validity.min()) if len(group) else None,worst_future_smooth_loss=float(group.smooth_loss.max()) if len(group) else None)
 pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
 for path,h in read(a.output/'input_manifest.json').items():assert sha(path)==h
 dump(a.output/'complete.json',{'input_hashes_preserved':True,'Q_equals_128_equal_mass_supports':True,'no_validity_repair':True,'selected_test_feasible':any(r['selected'] and r['feasible'] for r in rows)});print(pd.DataFrame(rows).to_string(index=False),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','futures','merge_futures','optimize','select','evaluate'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker',type=int,default=0);p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1);a=p.parse_args();c=read(CONFIG) if a.phase=='prepare' else read(a.output/'resolved_config.json');torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(c['GPU_memory_fraction']);print('RUN',vars(a),'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True);globals()[a.phase](a,c)
if __name__=='__main__':main()
