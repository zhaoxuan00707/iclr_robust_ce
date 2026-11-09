"""Task017: original Galaxy assets, complete head update, pixel sensitivity."""
import argparse,json,os,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ..galaxy.assets import load_classifier,load_autoencoder,CLASSIFIER,AUTOENCODER
from ..counterfactuals.frozen_feature_sensitivity import head_update,pixel_sensitivity,smooth_target
from ..counterfactuals.update_aware import differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/galaxy_update_aware.json'
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def name(r,beta):return f'r{r:g}_beta{beta:g}'
def assets(device):
 m=load_classifier(device).requires_grad_(False);ae=load_autoencoder(device)
 def feature(x):
  z=m.net.features(m.transform(x));z=m.net.avgpool(z).flatten(1)
  return m.net.classifier[:-1](z)
 return m,ae,feature

def batched(fn,x,batch=2):return torch.cat([fn(v) for v in x.split(batch)])

def prepare(a,c):
 out=a.output;manifest={str(p):sha(p) for p in [CONFIG,ROOT/c['task'],CLASSIFIER,AUTOENCODER]};refs=set();factids=set();pops={}
 for pair,parent in c['parents'].items():
  td=Path(parent)/'tasks'/f"target{c['targets'][pair]:02d}";p={}
  for split,old in [('cal','calibration'),('test','test')]:
   path=td/f'{old}_population.pt';d=load(path);manifest[str(path)]=sha(path)
   ids=np.load(td/f'{old}_factual_ids.npy').tolist();manifest[str(td/f'{old}_factual_ids.npy')]=sha(td/f'{old}_factual_ids.npy')
   assert not factids.intersection(ids),'factual cohort overlap';factids.update(ids)
   p[split]={'factual':d['factual'],'base':d['base_images'],'base_z':d['base_raw_generator_latents'],'ids':ids}
  p['target_z']=torch.from_numpy(np.load(td/'target_latents.npy')).float();p['target_images']=load(td/'target_images.pt');p['target_ids']=np.load(td/'target_ids.npy').tolist();refs.update(p['target_ids'])
  for fname in ['target_latents.npy','target_images.pt','target_ids.npy']:manifest[str(td/fname)]=sha(td/fname)
  pops[pair]=p
 parent=Path(c['parents']['merging_round']);d=load(parent/'training.pt');files=pd.read_csv(parent/'dataset_files.csv');train=files[files.split=='train'].sort_values('index');ids=train.path.tolist();assert len(ids)==len(d['x']);assert not factids.intersection(ids),'training/factual overlap'
 generator=torch.Generator().manual_seed(c['seed']);selected=[]
 for cls in range(10):
  valid=torch.tensor([i for i in range(len(ids)) if int(d['y'][i])==cls and ids[i] not in refs]);assert len(valid)>=c['train_per_class'];selected.extend(valid[torch.randperm(len(valid),generator=generator)[:c['train_per_class']]].tolist())
 data={'x':d['x'][selected],'y':d['y'][selected],'train_ids':[ids[i] for i in selected],'pairs':pops}
 for fname in ['training.pt','dataset_files.csv']:manifest[str(parent/fname)]=sha(parent/fname)
 torch.save(data,out/'population.pt');dump(out/'input_manifest.json',manifest);dump(out/'resolved_config.json',c)
 m,ae,feature=assets(a.device);rows=[]
 with torch.no_grad():
  features=batched(feature,data['x'].to(a.device)).cpu();torch.save(features,out/'training_features.pt')
  for pair,p in pops.items():
   folder=out/pair;folder.mkdir()
   refscores=batched(m,p['target_images'].to(a.device));target=c['targets'][pair]
   for split in ['cal','test']:
    q=p[split];decoded=batched(ae.decode,q['base_z'].to(a.device)).cpu();error=float((decoded-q['base']).abs().max());assert error<.005,(pair,split,error);q['historical_base']=q['base'];q['base']=decoded
    torch.backends.cudnn.allow_tf32=True;legacy=batched(ae.decode,q['base_z'].to(a.device),4).cpu();torch.backends.cudnn.allow_tf32=False;legacy_error=float((legacy-q['historical_base']).abs().max());assert legacy_error<2e-5,(pair,split,legacy_error)
    logits=batched(m,q['base'].to(a.device));factual=batched(m,q['factual'].to(a.device));pred=logits.softmax(1)[:,target].cpu()
    torch.save({'base_probability':pred,'factual_logits':factual.cpu(),'base_logits':logits.cpu(),'reference_logits':refscores.cpu()},folder/f'{split}_audit.pt')
    rows.append({'pair':pair,'split':split,'n':len(pred),'decode_max_error':error,'legacy_tf32_replay_max_error':legacy_error,'decode_pixel_L2':float((decoded-q['historical_base']).flatten(1).norm(dim=1).mean()),'current_validity':float((pred>=.5).float().mean()),'base_argmax':float((logits.argmax(1)==target).float().mean()),'factual_source_argmax':float((factual.argmax(1)==c['sources'][pair]).float().mean()),'factual_target_validity':float((factual.softmax(1)[:,target]>=.5).float().mean()),'reference_target_validity':float((refscores.softmax(1)[:,target]>=.5).float().mean()),'pixel_cost':float((q['base']-q['factual']).flatten(1).norm(dim=1).mean())})
 torch.save(data,out/'population.pt');pd.DataFrame(rows).to_csv(out/'asset_audit.csv',index=False);print(rows,flush=True)

def objective_context(a,c):
 d=load(a.output/'population.pt');p=d['pairs'][a.pair];m,ae,feature=assets(a.device);x=d['x'].to(a.device);y=d['y'].to(a.device);features=load(a.output/'training_features.pt').to(a.device).requires_grad_();head=m.net.classifier[-1]
 # All three steps are differentiated; feature extractor parameters are fixed.
 w,b=head_update(features,y,head.weight,head.bias,**{k:c['update'][k] for k in ['steps','lr']})
 def objective(q,r):
  logits=F.linear(batched(lambda z:feature(ae.decode(z)),q),w,b)
  nominal=smooth_target(logits,c['targets'][a.pair],c['temperature'])
  adjoint=torch.autograd.grad(nominal,features,create_graph=True,retain_graph=True)[0]
  sensitivity=pixel_sensitivity(adjoint,x,feature,c['microbatch'])
  return nominal+r*sensitivity,nominal,sensitivity
 return d,p,m,ae,objective

def audit(a,c):
 start=time.perf_counter();d,p,m,ae,obj=objective_context(a,c);q=p['cal']['base_z'].to(a.device).requires_grad_();f,n,s=obj(q,1.);g=torch.autograd.grad(f,q,retain_graph=True)[0];direction=g/g.norm().clamp_min(1e-20);row={'pair':a.pair,'nominal':float(n.detach()),'sensitivity':float(s.detach()),'gradient_norm':float(g.norm())};del f,n,s,g
 vals=[]
 for sign in [-1,1]:
  f,n,s=obj(q.detach()+sign*.002*direction.detach(),1.);vals.append(float(f.detach()));del f,n,s
 fd=(vals[1]-vals[0])/.004;row.update(finite_difference=fd,relative_error=abs(fd-row['gradient_norm'])/max(1e-8,row['gradient_norm']),seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
 dump(a.output/a.pair/'gradient_audit.json',row);print(row,flush=True);assert row['relative_error']<.2,row

def scenarios(c):
 rows=[]
 for split in ['cal','test']:
  for family in ['brightness','noise']:
   for severity in c['future']['severities']:
    for rep in range(c['future']['realizations']):
     rows.append({'split':split,'family':family,'severity':severity,'rep':rep,'seed':c['future'][split+'_seed']+len([r for r in rows if r['split']==split])})
 return rows

def futures(a,c):
 d=load(a.output/'population.pt');m,ae,feature=assets(a.device);head=m.net.classifier[-1];folder=a.output/'futures';folder.mkdir(exist_ok=True);rows=[]
 for i,rec in enumerate(scenarios(c)):
  if i%8!=a.worker:continue
  rng=torch.Generator().manual_seed(rec['seed']);x=d['x'];sev=rec['severity']
  if rec['family']=='brightness':strength=1-.2*sev*float(.8+.4*torch.rand((),generator=rng));shifted=(x*strength).clamp(0,1)
  else:strength=.04*sev;shifted=(x+strength*torch.randn(x.shape,generator=rng)).clamp(0,1)
  with torch.no_grad():feat=batched(feature,shifted.to(a.device))
  w,b=head_update(feat,d['y'].to(a.device),head.weight,head.bias,c['update']['steps'],c['update']['lr'],False)
  path=folder/f'model{i:03d}.pt';torch.save({'weight':w.detach().cpu(),'bias':b.detach().cpu()},path)
  rec={**rec,'strength':strength,'paired_RMS_pixel_displacement':float((shifted-x).flatten(1).square().sum(1).mean().sqrt()),'path':str(path),'sha256':sha(path),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')};rows.append(rec);print(rec,flush=True)
 pd.DataFrame(rows).to_csv(a.output/f'futures_worker{a.worker}.csv',index=False)

def optimize(a,c):
 start=time.perf_counter();d,p,m,ae,obj=objective_context(a,c);base=p[a.split]['base_z'].to(a.device);target=p['target_z'].to(a.device);q=base.clone().requires_grad_();D=exact_w2(base.cpu(),target.cpu()).distance;budget=a.beta*D;tol=c['solver']['feas_relative']*max(1,D);dual=0.;best=None;trace=[]
 folder=a.output/a.pair/a.split/name(a.r,a.beta);folder.mkdir(parents=True,exist_ok=False)
 for step in range(c['solver']['steps']+1):
  f,n,s=obj(q,a.r);d0,_=differentiable_w2(base,q);d1,_=differentiable_w2(q,target);violation=d0+d1-budget
  row={'iteration':step,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'geometry_total':float((d0+d1).detach()),'violation':float(violation.detach()),'dual':dual};trace.append(row)
  if row['violation']<=tol and (best is None or row['surrogate']<best['surrogate']):best={**row,'Q':q.detach().cpu().clone()}
  if step%20==0:print(a.pair,a.split,name(a.r,a.beta),row,flush=True)
  if step==c['solver']['steps']:break
  loss=f+dual*violation+.5*c['solver']['rho']*torch.relu(violation).square();grad=torch.autograd.grad(loss,q,retain_graph=True)[0];assert torch.isfinite(grad).all()
  with torch.no_grad():q.add_(grad*min(1.,c['solver']['gradient_clip']/max(float(grad.norm()),1e-20)),alpha=-c['solver']['primal_lr'])
  newg=exact_w2(base.cpu(),q.cpu()).distance+exact_w2(q.cpu(),target.cpu()).distance-budget;dual=max(0,dual+c['solver']['rho']*newg)
  del f,n,s,d0,d1,violation,loss,grad
 assert best is not None
 selected=best.pop('Q');plan=exact_w2(base.cpu(),selected);plan1=exact_w2(selected,target.cpu());latents=torch.empty_like(selected);latents[plan.source_index]=selected[plan.target_index]
 with torch.no_grad():
  images=batched(ae.decode,latents.to(a.device)).cpu();logits=batched(m,images.to(a.device)).cpu()
 metrics={**best,'pair':a.pair,'split':a.split,'r':a.r,'beta':a.beta,'D':D,'budget':budget,'W2_Base_Q':plan.distance,'W2_Q_P1':plan1.distance,'geometry_total':plan.distance+plan1.distance,'geometry_tolerance':tol,'feasible':plan.distance+plan1.distance<=budget+tol,'final_iterate_feasible':row['violation']<=tol,'final_iterate_surrogate':row['surrogate'],'current_validity':float((logits.softmax(1)[:,c['targets'][a.pair]]>=.5).float().mean()),'pixel_cost':float((images-p[a.split]['factual']).flatten(1).norm(dim=1).mean()),'completed_updates':c['solver']['steps'],'runtime_seconds':time.perf_counter()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'convergence_certified':False}
 torch.save({'Q':selected,'latents':latents,'images':images,'logits':logits,'assignment':plan.state_dict()},folder/'output.pt');dump(folder/'metrics.json',metrics);pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);print('DONE',metrics,flush=True)

def evaluate(a,c):
 split=a.split;d=load(a.output/'population.pt');p=d['pairs'][a.pair];folder=a.output/a.pair;target=c['targets'][a.pair];m,ae,feature=assets(a.device)
 manifest=pd.concat([pd.read_csv(a.output/f'futures_worker{i}.csv') for i in range(8)],ignore_index=True);manifest.to_csv(a.output/'future_manifest.csv',index=False);manifest=manifest[manifest.split==split]
 images={'Base':p[split]['base'],'Historical_Base':p[split]['historical_base']};metrics=[]
 for r in c['r_grid']:
  for beta in c['beta_grid']:
   tag=name(r,beta);path=folder/split/tag;images[tag]=load(path/'output.pt')['images'];metrics.append({**json.loads((path/'metrics.json').read_text()),'method':tag,'output_sha256':sha(path/'output.pt')})
 if split=='test':
  selection=json.loads((folder/'selection.json').read_text());assert (folder/'selection.json').stat().st_mtime<min((folder/split/name(r,b)/'output.pt').stat().st_mtime for r in c['r_grid'] for b in c['beta_grid'])
  torch.save(images,folder/'frozen_test_images.pt');dump(folder/'test_freeze.json',{'time':time.time(),'images_sha256':sha(folder/'frozen_test_images.pt'),'selection_sha256':sha(folder/'selection.json')})
 with torch.no_grad():features={tag:batched(feature,im.to(a.device)) for tag,im in images.items()}
 initial=load(folder/f'{split}_audit.pt')['base_probability']>=.5;rows=[];individual=[]
 head=m.net.classifier[-1];nomw,nomb=head_update(load(a.output/'training_features.pt').to(a.device),d['y'].to(a.device),head.weight,head.bias,c['update']['steps'],c['update']['lr'],False)
 records=manifest.to_dict('records')+[{'family':'nominal','severity':0,'seed':0,'split':split,'path':None}]
 for rec in records:
  if rec['path'] is not None:
   assert sha(Path(rec['path']))==rec['sha256'];state=load(rec['path']);w,b=state['weight'].to(a.device),state['bias'].to(a.device)
  else:w,b=nomw,nomb
  with torch.no_grad():
   for tag,feat in features.items():
    logits=F.linear(feat,w,b).cpu();prob=logits.softmax(1)[:,target];valid=prob>=.5
    rows.append({**rec,'method':tag,'FV':float(valid.float().mean()),'argmax_validity':float((logits.argmax(1)==target).float().mean()),'initial_valid_subset_FV':float(valid[initial].float().mean()) if initial.any() else None,'initial_valid_subset_n':int(initial.sum()),'smooth_loss':float(smooth_target(logits,target))})
    for i in range(len(prob)):individual.append({'method':tag,'family':rec['family'],'severity':rec['severity'],'seed':rec['seed'],'index':i,'id':p[split]['ids'][i],'probability':float(prob[i]),'argmax':int(logits.argmax(1)[i]),'initial_valid':bool(initial[i])})
 frame=pd.DataFrame(rows);frame.to_csv(folder/f'{split}_per_model.csv',index=False);pd.DataFrame(individual).to_csv(folder/f'{split}_individual.csv',index=False)
 audit=pd.read_csv(a.output/'asset_audit.csv');base=audit[(audit.pair==a.pair)&(audit.split==split)].iloc[0].to_dict();initial=pd.read_csv(folder/split/name(0,c['beta_grid'][0])/'trace.csv').iloc[0].to_dict();metrics.append({**base,'method':'Base','feasible':True,'nominal':initial['nominal'],'sensitivity':initial['sensitivity']});metrics.append({'method':'Historical_Base','feasible':True,'pixel_cost':float((p[split]['historical_base']-p[split]['factual']).flatten(1).norm(dim=1).mean()),'current_validity':float((F.linear(features['Historical_Base'],head.weight,head.bias).softmax(1)[:,target]>=.5).float().mean())})
 for row in metrics:
  f=frame[(frame.method==row['method'])&(frame.family!='nominal')];row.update(FV=float(f.FV.mean()),FV_min=float(f.FV.min()),argmax_FV=float(f.argmax_validity.mean()),initial_valid_subset_FV=float(f.initial_valid_subset_FV.mean()),nominal_FV=float(frame[(frame.method==row['method'])&(frame.family=='nominal')].FV.iloc[0]))
 summary=pd.DataFrame(metrics);summary.to_csv(folder/f'{split}_summary.csv',index=False)
 if split=='cal':
  selected=summary[(~summary.method.isin(['Base','Historical_Base']))&summary.feasible].sort_values(['FV','pixel_cost','r','beta'],ascending=[False,True,True,True]).iloc[0].to_dict();dump(folder/'selection.json',selected)
 else:
  for path,h in json.loads((a.output/'input_manifest.json').read_text()).items():assert sha(path)==h
  dump(folder/'complete.json',{'time':time.time(),'selected_method':selection['method'],'no_validity_repair':True,'input_hashes_verified':True})
 print(summary.to_string(index=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--phase',required=True,choices=['prepare','audit','futures','optimize','evaluate']);p.add_argument('--pair',default='merging_round');p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1.02);p.add_argument('--worker',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args();c=json.loads(CONFIG.read_text())
 torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(.20)
 print('ARGS',vars(a),'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True);globals()[a.phase](a,c)
if __name__=='__main__':main()
