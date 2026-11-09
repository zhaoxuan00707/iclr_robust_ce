"""Task018: original CLIP/StyleGAN, full W512, smooth update-aware objective."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from ..ffhq.assets import Generator,SmileClassifier,GAN,CLIP
from ..counterfactuals.clip_projection_update import projection_update
from ..counterfactuals.frozen_feature_sensitivity import pixel_sensitivity,smooth_target
from ..counterfactuals.update_aware import differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
from .galaxy_update_aware import load,dump,sha,name,batched
ROOT=Path(__file__).resolve().parents[4];CONFIG=ROOT/'experiments/configs/ffhq_update_aware.json'

class CanonicalClassifier:
 def __init__(self,model):self.model=model
 def __call__(self,x):return self.model.logits_features(self.model.features(x))
class Decoder:
 def __init__(self,g,c):self.g=g;self.c=c
 def decode(self,q):return torch.cat([self.c.canonical(self.g.decode(w)) for w in q.split(1)])
class StreamedFeatures(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,decoder,clf):
  ctx.decoder=decoder;ctx.clf=clf;ctx.save_for_backward(q.detach())
  with torch.no_grad():return torch.cat([clf.features(decoder.decode(w)) for w in q.split(1)])
 @staticmethod
 def backward(ctx,adjoint):
  q,=ctx.saved_tensors;grad=[]
  for w,u in zip(q,adjoint):
   with torch.enable_grad():
    v=w.detach()[None].requires_grad_();feature=ctx.clf.features(ctx.decoder.decode(v));grad.append(torch.autograd.grad(feature,v,u[None])[0])
  return torch.cat(grad),None,None

def assets(device):
 c=SmileClassifier(device).eval().requires_grad_(False);g=Generator(device);return CanonicalClassifier(c),Decoder(g,c),c

def prepare(a,c):
 out=a.output;parent=Path(c['parent']);inputs=[CONFIG,ROOT/c['task'],GAN,CLIP,parent/'training.pt',parent/'references.pt',parent/'target_ids.npy'];d=load(parent/'training.pt');refs=load(parent/'references.pt');refids=np.load(parent/'target_ids.npy').tolist();rng=torch.Generator().manual_seed(c['seed']);idx=[]
 for label in range(2):
  eligible=torch.tensor([i for i in range(len(d['y'])) if int(d['y'][i])==label and d['ids'][i] not in refids]);assert len(eligible)>=c['train_per_class'];idx.extend(eligible[torch.randperm(len(eligible),generator=rng)[:c['train_per_class']]].tolist())
 p={'target_z':refs['w'],'target_images':refs['x'],'target_ids':refids};seen=set(d['ids'])
 for split,old in [('cal','calibration'),('test','test')]:
  path=parent/f'{old}_population.pt';ids_path=parent/f'{old}_factual_ids.npy';inputs+=[path,ids_path];v=load(path);ids=np.load(ids_path).tolist();assert not seen.intersection(ids);seen.update(ids)
  p[split]={'factual':v['factual_x'],'factual_z':v['factual_w'],'base_z':v['base_w'],'historical_base':v['base_x'],'ids':ids}
 data={'x':d['x'][idx],'y':d['y'][idx],'train_ids':[d['ids'][i] for i in idx],'pairs':{'smile':p}}
 dump(out/'input_manifest.json',{str(path):sha(path) for path in inputs});dump(out/'resolved_config.json',c);folder=out/'smile';folder.mkdir();m,decoder,clf=assets(a.device);rows=[]
 with torch.no_grad():
  torch.save(batched(clf.features,data['x'].to(a.device)),out/'training_features.pt')
  for split in ['cal','test']:
   v=p[split];base=decoder.decode(v['base_z'].to(a.device)).cpu();v['base']=base;factual=decoder.decode(v['factual_z'].to(a.device)).cpu();logits=batched(m,base.to(a.device)).cpu();flogits=batched(m,factual.to(a.device)).cpu();historical=batched(m,v['historical_base'].to(a.device)).cpu();prob=logits.softmax(1)[:,0]
   torch.save({'base_probability':prob,'base_logits':logits,'factual_logits':flogits},folder/f'{split}_audit.pt')
   rows.append({'pair':'smile','split':split,'n':len(base),'decode_max_error':float((base-v['historical_base']).abs().max()),'decode_pixel_L2':float((base-v['historical_base']).flatten(1).norm(dim=1).mean()),'factual_replay_max_error':float((factual-v['factual']).abs().max()),'current_validity':float((prob>=.5).float().mean()),'historical_current_validity':float((historical.softmax(1)[:,0]>=.5).float().mean()),'factual_target_validity':float((flogits.softmax(1)[:,0]>=.5).float().mean()),'pixel_cost':float((base-v['factual']).flatten(1).norm(dim=1).mean())})
 torch.save(data,out/'population.pt');pd.DataFrame(rows).to_csv(out/'asset_audit.csv',index=False);print(rows,flush=True)

def objective_context(a,c):
 d=load(a.output/'population.pt');p=d['pairs']['smile'];m,decoder,clf=assets(a.device);x=d['x'].to(a.device);features=load(a.output/'training_features.pt').to(a.device).requires_grad_()
 projection=projection_update(features,d['y'].to(a.device),clf.base_projection,clf.text,clf.scale,c['update']['steps'],c['update']['lr'])
 def objective(q,r):
  query=StreamedFeatures.apply(q,decoder,clf);nominal=smooth_target(clf.logits_features(query,projection),0,c['temperature']);adj=torch.autograd.grad(nominal,features,create_graph=True,retain_graph=True)[0];sens=pixel_sensitivity(adj,x,clf.features,c['microbatch']);return nominal+r*sens,nominal,sens
 return d,p,m,decoder,objective

def audit(a,c):
 start=time.perf_counter();d,p,m,decoder,obj=objective_context(a,c);q=p['cal']['base_z'].to(a.device).requires_grad_();f,n,s=obj(q,.25);g=torch.autograd.grad(f,q,retain_graph=True)[0];direction=g.detach()/g.norm().clamp_min(1e-20);row={'nominal':float(n.detach()),'sensitivity':float(s.detach()),'gradient_norm':float(g.norm())};del f,n,s,g
 checks=[]
 for eps in [.02,.01,.005,.002]:
  vals=[]
  for sign in [-1,1]:
   f,n,s=obj(q.detach()+sign*eps*direction,.25);vals.append(float(f.detach()));del f,n,s
  fd=(vals[1]-vals[0])/(2*eps);checks.append({'epsilon':eps,'finite_difference':fd,'relative_error':abs(fd-row['gradient_norm'])/max(row['gradient_norm'],1e-8)})
 row.update(checks=checks,seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated());dump(a.output/'gradient_audit.json',row);print(row,flush=True);assert min(v['relative_error'] for v in checks)<.2,row

def scenarios(c):
 rows=[]
 for split in ['cal','test']:
  for family in ['brightness','noise','blur']:
   for severity in c['future']['severities']:
    for rep in range(c['future']['realizations']):rows.append({'split':split,'family':family,'severity':severity,'rep':rep,'seed':c['future'][split+'_seed']+sum(v['split']==split for v in rows)})
 return rows

def futures(a,c):
 d=load(a.output/'population.pt');clf=SmileClassifier(a.device).eval().requires_grad_(False);folder=a.output/'futures';folder.mkdir(exist_ok=True);rows=[]
 for i,rec in enumerate(scenarios(c)):
  if i%8!=a.worker:continue
  rng=torch.Generator().manual_seed(rec['seed']);x=d['x'];sev=rec['severity']
  if rec['family']=='brightness':strength=1-.2*sev*float(.8+.4*torch.rand((),generator=rng));changed=(x*strength).clamp(0,1)
  elif rec['family']=='noise':strength=.04*sev;changed=(x+strength*torch.randn(x.shape,generator=rng)).clamp(0,1)
  else:strength=.8*sev*float(.8+.4*torch.rand((),generator=rng));changed=gaussian_blur(x,[9,9],[strength,strength])
  with torch.no_grad():features=batched(clf.features,changed.to(a.device))
  projection=projection_update(features,d['y'].to(a.device),clf.base_projection,clf.text,clf.scale,c['update']['steps'],c['update']['lr'],False)
  path=folder/f'model{i:03d}.pt';torch.save({'projection':projection.detach().cpu()},path);row={**rec,'strength':strength,'paired_RMS_pixel_displacement':float((changed-x).flatten(1).square().sum(1).mean().sqrt()),'path':str(path),'sha256':sha(path),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')};rows.append(row);print(row,flush=True)
 pd.DataFrame(rows).to_csv(a.output/f'futures_worker{a.worker}.csv',index=False)

def optimize(a,c):
 start=time.perf_counter();d,p,m,ae,obj=objective_context(a,c);base=p[a.split]['base_z'].to(a.device);target=p['target_z'].to(a.device);q=base.clone().requires_grad_();D=exact_w2(base.cpu(),target.cpu()).distance;budget=a.beta*D;tol=c['solver']['feas_relative']*max(1,D);dual=0.;best=None;trace=[]
 folder=a.output/a.pair/a.split/name(a.r,a.beta);folder.mkdir(parents=True,exist_ok=False)
 for step in range(c['solver']['steps']+1):
  f,n,s=obj(q,a.r);d0,_=differentiable_w2(base,q);d1,_=differentiable_w2(q,target);violation=d0+d1-budget
  row={'iteration':step,'surrogate':float(f.detach()),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'geometry_total':float((d0+d1).detach()),'violation':float(violation.detach()),'dual':dual};trace.append(row)
  exact_total=exact_w2(base.cpu(),q.cpu()).distance+exact_w2(q.cpu(),target.cpu()).distance
  if exact_total<=budget+tol and (best is None or row['surrogate']<best['surrogate']):best={**row,'Q':q.detach().cpu().clone()}
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
 split=a.split;d=load(a.output/'population.pt');p=d['pairs'][a.pair];folder=a.output/a.pair;target=c['targets'][a.pair];m,ae,clf=assets(a.device);feature=clf.features
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
 nominal_projection=projection_update(load(a.output/'training_features.pt').to(a.device),d['y'].to(a.device),clf.base_projection,clf.text,clf.scale,c['update']['steps'],c['update']['lr'],False)
 records=manifest.to_dict('records')+[{'family':'nominal','severity':0,'seed':0,'split':split,'path':None}]
 for rec in records:
  if rec['path'] is not None:
   assert sha(Path(rec['path']))==rec['sha256'];state=load(rec['path']);projection=state['projection'].to(a.device)
  else:projection=nominal_projection
  with torch.no_grad():
   for tag,feat in features.items():
    logits=clf.logits_features(feat,projection).cpu();prob=logits.softmax(1)[:,target];valid=prob>=.5
    rows.append({**rec,'method':tag,'FV':float(valid.float().mean()),'argmax_validity':float((logits.argmax(1)==target).float().mean()),'initial_valid_subset_FV':float(valid[initial].float().mean()) if initial.any() else None,'initial_valid_subset_n':int(initial.sum()),'smooth_loss':float(smooth_target(logits,target))})
    for i in range(len(prob)):individual.append({'method':tag,'family':rec['family'],'severity':rec['severity'],'seed':rec['seed'],'index':i,'id':p[split]['ids'][i],'probability':float(prob[i]),'argmax':int(logits.argmax(1)[i]),'initial_valid':bool(initial[i])})
 frame=pd.DataFrame(rows);frame.to_csv(folder/f'{split}_per_model.csv',index=False);pd.DataFrame(individual).to_csv(folder/f'{split}_individual.csv',index=False)
 audit=pd.read_csv(a.output/'asset_audit.csv');base=audit[(audit.pair==a.pair)&(audit.split==split)].iloc[0].to_dict();initial=pd.read_csv(folder/split/name(0,c['beta_grid'][0])/'trace.csv').iloc[0].to_dict();metrics.append({**base,'method':'Base','feasible':True,'nominal':initial['nominal'],'sensitivity':initial['sensitivity']});metrics.append({'method':'Historical_Base','feasible':True,'pixel_cost':float((p[split]['historical_base']-p[split]['factual']).flatten(1).norm(dim=1).mean()),'current_validity':float((clf.logits_features(features['Historical_Base']).softmax(1)[:,target]>=.5).float().mean())})
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
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--phase',required=True,choices=['prepare','audit','futures','optimize','evaluate']);p.add_argument('--pair',default='smile');p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1.02);p.add_argument('--worker',type=int,default=0);p.add_argument('--device',default='cuda');a=p.parse_args();c=json.loads(CONFIG.read_text())
 torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(False)
 torch.cuda.set_per_process_memory_fraction(.065);print('ARGS',vars(a),'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True);globals()[a.phase](a,c)
if __name__=='__main__':main()
