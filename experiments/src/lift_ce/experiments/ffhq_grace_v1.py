"""FFHQ-generated face GRACE with original StyleGAN3/CLIP/conditional flow."""
import os,sys,json,time,socket,hashlib,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from ..ffhq.assets import Generator,SmileClassifier,load_flow,GAN,FLOW,CLIP
from ..ffhq.flow import progressive_leap
from .galaxy_grace_v1 import sha,dump,load,stamp,seed,project_delta,paired_distance,exact_data_w2
from ..counterfactuals.geodesic_excess import exact_w2
from ..counterfactuals.minimal_validity_repair import solve_raw
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/ffhq_grace_v1.json'

def decode(g,w,device):
 with torch.no_grad():return torch.cat([g.decode(q.to(device)).cpu() for q in w.split(1)])
def features(c,x,device):
 with torch.no_grad():return torch.cat([c.features(q.to(device)).cpu() for q in x.split(8)])
def logits(c,f,projections):
 v=torch.einsum('nf,kfd->knd',f,projections)
 return c.scale*F.normalize(v,dim=-1)@c.text.T

def update(c,x,y,config,device):
 f=features(c,x,device).to(device);y=y.to(device);u=config['update'];seed(u['seed']);p=c.base_projection.detach().clone().requires_grad_(True)
 opt=torch.optim.AdamW([p],lr=u['lr'],weight_decay=u['weight_decay']);losses=[]
 for epoch in range(u['epochs']):
  order=torch.randperm(len(x),generator=torch.Generator().manual_seed(u['seed']+epoch)).to(device);total=0.
  for idx in order.split(u['batch_size']):
   opt.zero_grad();loss=F.cross_entropy(c.logits_features(f[idx],p),y[idx]);loss.backward();opt.step();total+=float(loss.detach())*len(idx)
  losses.append(total/len(x))
 return p.detach().cpu(),losses

def surrogate(c,x,y,d,qfeatures,lr):
 p=c.base_projection.detach().clone().requires_grad_(True)
 loss=F.cross_entropy(c.logits_features(c.features(x+d),p),y)
 grad=torch.autograd.grad(loss,p,create_graph=True)[0]
 new=p*(1-lr*.01)-lr*grad/(torch.sqrt(grad.square()+1e-12)+1e-8)
 return c.logits_features(qfeatures,new).softmax(-1)[:,0]

class FeatureGradient(torch.autograd.Function):
 @staticmethod
 def forward(ctx,w,bank):
  ctx.bank=bank;ctx.save_for_backward(w.detach());return bank.cached_features(w)
 @staticmethod
 def backward(ctx,upstream):
  w,=ctx.saved_tensors;result=[]
  for row,derivative in zip(w,upstream):
   with torch.enable_grad():
    q=row.detach()[None].requires_grad_(True);f=ctx.bank.c.features(ctx.bank.c.canonical(ctx.bank.g.decode(q)))
    result.append(torch.autograd.grad((f*derivative).sum(),q)[0])
  return torch.cat(result),None

class FaceBank:
 def __init__(self,records,device):
  self.device=device;self.g=Generator(device);self.c=SmileClassifier(device);self.states=torch.stack([load(Path(r['checkpoint']))['projection'] for r in records]).to(device);self.count=len(records);self.cache={}
 def cached_features(self,w):
  key=hashlib.sha256(w.detach().cpu().numpy().tobytes()).hexdigest()
  if key not in self.cache:
   with torch.no_grad():self.cache[key]=torch.cat([self.c.features(self.c.canonical(self.g.decode(q.to(self.device)))).cpu() for q in w.detach().split(1)])
  return self.cache[key].to(self.device)
 def logits(self,w,start=0,end=None):
  f=FeatureGradient.apply(w,self) if torch.is_grad_enabled() and w.requires_grad else self.cached_features(w)
  scores=logits(self.c,f,self.states[start:end]);return scores[:,:,0]-scores[:,:,1]
 def risk(self,w):
  with torch.no_grad():return float(torch.relu(.5-self.logits(w).sigmoid()).double().mean(1).max())
 def canonical(self,w):
  with torch.no_grad():return torch.cat([self.c.canonical(self.g.decode(q.to(self.device))).cpu() for q in w.split(1)])

def records(out,r):
 rows=[]
 for p in sorted((out/'jobs').glob('attack_*/models.json')):rows+=json.loads(p.read_text())
 return [x for x in rows if x['W_data']<=r+1e-6]

def initialize(a,c):
 out=a.output
 for d in ['jobs','logs','checkpoints','figures','qualitative_examples']:(out/d).mkdir(exist_ok=True)
 dump(out/'resolved_config.json',c);shutil.copyfile(ROOT/'results/013_ffhq_asset_inventory.md',out/'ffhq_asset_inventory.md');shutil.copyfile(ROOT/'results/013_ffhq_asset_checks.json',out/'asset_checks.json')
 dump(out/'model_manifest.json',{'hashes':{str(p):sha(p) for p in [GAN,FLOW,CLIP]},'representation':'W512','class0':'smiling face','class1':'face','smile_target':0,'original_encoder':'none; generated W retained directly','float_dtype':'float32 CLIP/flow, original StyleGAN mixed precision'})
 dump(out/'hardware_manifest.json',{'hostname':socket.gethostname(),'gpu_ids':c['gpu_ids'],'hardware':subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv'],text=True),'inherited_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'scheduler':'sinfo probe timed out during preceding session; local GPU-aware queue','memory_cap_fraction':.16})
 dump(out/'construction_clock.json',{'start_unix':time.time(),'start_utc':stamp()})

def sample(a,c):
 _,split,shard=a.job.split('_');shard=int(shard);g=Generator(a.device);clf=SmileClassifier(a.device);ws=[];xs=[];ys=[];ids=[]
 offsets={'train':0,'calibration':10000,'test':20000}
 for i in range(shard*32,(shard+1)*32):
  s=c['seed']+offsets[split]+i;z=torch.randn(1,512,generator=torch.Generator().manual_seed(s)).to(a.device)
  with torch.no_grad():w=g.mapping(z);x=clf.canonical(g.decode(w));y=clf.logits_features(clf.features(x)).argmax(1)
  ws.append(w.cpu());xs.append(x.cpu());ys.append(y.cpu());ids.append(f'{split}:seed{s}')
 torch.save({'w':torch.cat(ws),'x':torch.cat(xs),'y':torch.cat(ys),'ids':ids},a.output/'jobs'/a.job/'samples.pt')

def prepare(a,c):
 out=a.output;pools={};manifest={}
 for split in ['train','calibration','test']:
  files=sorted((out/'jobs').glob(f'sample_{split}_*/samples.pt'));items=[load(p) for p in files]
  pools[split]={k:torch.cat([d[k] for d in items]) for k in ['w','x','y']};pools[split]['ids']=sum([d['ids'] for d in items],[])
  manifest[split]={'size':len(pools[split]['w']),'class_counts':torch.bincount(pools[split]['y'],minlength=2).tolist(),'files':[str(p) for p in files]}
 train=pools['train'];torch.save(train,out/'training.pt');refs=torch.nonzero(train['y']==0).flatten()[:c['references']];assert len(refs)==c['references']
 torch.save({'w':train['w'][refs],'x':train['x'][refs]},out/'references.pt');np.save(out/'target_ids.npy',np.array([train['ids'][i] for i in refs]));np.save(out/'target_latents.npy',train['w'][refs].numpy())
 for split in ['calibration','test']:
  d=pools[split];idx=torch.nonzero(d['y']==1).flatten()[:c['factuals_per_split']];assert len(idx)==c['factuals_per_split']
  torch.save({k:d[k][idx] for k in ['w','x','y']},out/f'{split}_factuals.pt');np.save(out/f'{split}_factual_ids.npy',np.array([d['ids'][i] for i in idx]))
 assert not set(pools['train']['ids'])&set(pools['test']['ids']);assert not set(pools['calibration']['ids'])&set(pools['test']['ids'])
 shutil.copyfile(out/'test_factual_ids.npy',out/'factual_ids.npy');np.save(out/'factual_images.npy',load(out/'test_factuals.pt')['x'].numpy())
 dump(out/'dataset_manifest.json',{'pools':manifest,'factuals_per_split':c['factuals_per_split'],'references':c['references'],'training_sha256':sha(out/'training.pt'),'generated_not_real_ffhq':True,'ground_truth_smile_labels':False,'selection':'first qualifying generated IDs in seeded order; original CLIP class1 factuals, train class0 references'})

def base(a,c):
 split=a.job.split('_')[1];out=a.output;facts=load(out/f'{split}_factuals.pt');g=Generator(a.device);clf=SmileClassifier(a.device);flow=load_flow(a.device)
 def classify(w):
  with torch.no_grad():return torch.cat([clf(g.decode(q)).argmax(1) for q in w.split(1)])
 target=torch.zeros(len(facts['w']),dtype=torch.long,device=a.device);start=time.time()
 base_w=progressive_leap(flow,facts['w'].to(a.device),classify,target,**c['base_generator'])[-1].cpu();base_seconds=time.time()-start;start=time.time()
 robust_w=progressive_leap(flow,base_w.to(a.device),classify,target,**c['leapfactual_baseline'])[-1].cpu();extra_seconds=time.time()-start
 with torch.no_grad():
  canonical=lambda w:torch.cat([clf.canonical(g.decode(q.to(a.device))).cpu() for q in w.split(1)])
  x=canonical(base_w);robust_x=canonical(robust_w);prob=clf.logits_features(features(clf,x,a.device).to(a.device)).softmax(1)[:,0]
 torch.save({'factual_w':facts['w'],'factual_x':facts['x'],'base_w':base_w,'base_x':x,'robust_w':robust_w,'robust_x':robust_x},out/f'{split}_population.pt')
 dump(out/'jobs'/a.job/'quality.json',{'base_seconds':base_seconds,'robust_extra_seconds':extra_seconds,'base_current_validity':float((prob>=.5).double().mean()),'failed_base_retained':True})
 if split=='test':np.save(out/'base_cf_images.npy',x.numpy());np.save(out/'base_cf_latents.npy',base_w.numpy())

def attack(a,c):
 out=a.output;_,ri,restart=a.job.split('_');ri=int(ri);restart=int(restart);r=c['r_grid'][ri];s=c['seed']+1000+100*ri+restart;seed(s);d=load(out/'training.pt');settings=c['attack'];n=len(d['x']);idx=torch.randperm(n,generator=torch.Generator().manual_seed(s))[:settings['mutable_train_examples']]
 clf=SmileClassifier(a.device);g=Generator(a.device);pop=load(out/'calibration_population.pt');refs=load(out/'references.pt')['w'];plan=exact_w2(pop['base_w'],refs);perm=torch.empty(len(refs),dtype=torch.long);perm[plan.source_index]=plan.target_index
 with torch.no_grad():qf=torch.cat([clf.features(clf.canonical(g.decode(w[None].to(a.device)))).cpu() for lam in [0.,.5,1.] for w in ((1-lam)*pop['base_w']+lam*refs[perm])])
 del g;torch.cuda.empty_cache();x=d['x'][idx].to(a.device);y=d['y'][idx].to(a.device);delta=project_delta(x,torch.randn_like(x)*.01,r,n).requires_grad_(True);opt=torch.optim.Adam([delta],lr=settings['learning_rate']);trace=[]
 for step in range(settings['steps']):
  rng=torch.Generator().manual_seed(s+step);ii=torch.randperm(len(x),generator=rng)[:settings['batch_size']].to(a.device);qi=torch.randperm(len(qf),generator=rng)[:settings['query_batch_size']]
  p=surrogate(clf,x[ii],y[ii],delta[ii],qf[qi].to(a.device),c['update']['lr']);loss=F.softplus(10*(.5-p)).mean()/10;grad=torch.autograd.grad(loss,delta)[0];assert torch.isfinite(grad).all()
  opt.zero_grad();delta.grad=-grad;opt.step()
  with torch.no_grad():delta.copy_(project_delta(x,delta,r,n))
  trace.append({'step':step,'surrogate':float(loss.detach()),'gradient_norm':float(grad.norm()),'paired_bound':float(paired_distance(delta.detach(),n))})
 job=out/'jobs'/a.job;pd.DataFrame(trace).to_csv(job/'search.csv',index=False);rows=[]
 for fraction in settings['radial_fractions']:
  shift_delta=project_delta(x,delta.detach()*fraction,r,n).cpu();changed=d['x'].clone();changed[idx]+=shift_delta
  distance=exact_data_w2(d['x'],changed,d['y'],a.device);upper=float(paired_distance(shift_delta,n));assert distance<=r+1e-6 and upper<=r+1e-6
  projection,losses=update(clf,changed,d['y'],c,a.device);path=out/'checkpoints'/f'{a.job}_{fraction}.pt';torch.save({'projection':projection},path);perturbation=job/f'delta_{fraction}.pt';torch.save({'indices':idx,'delta':shift_delta},perturbation)
  rows.append({'candidate':path.stem,'checkpoint':str(path),'checkpoint_sha256':sha(path),'perturbation':str(perturbation),'perturbation_sha256':sha(perturbation),'r':r,'W_data':distance,'paired_bound':upper,'seed':s,'fraction':fraction,'feasible':True,'losses':losses})
 dump(job/'models.json',rows)

def shift(x,family,strength,s):
 if family=='brightness':return (x*strength).clamp(0,1)
 if family=='noise':return (x+strength*torch.randn(x.shape,generator=torch.Generator().manual_seed(s))).clamp(0,1)
 if family=='blur':return x.clone() if strength==0 else gaussian_blur(x,[9,9],[strength,strength])
 raise ValueError(family)

def shifted_model(a,c):
 out=a.output;clf=SmileClassifier(a.device);d=load(out/'training.pt')
 if a.phase=='calmodel':scenario=c['calibration_scenarios'][int(a.job.split('_')[-1])]
 else:
  _,family,si,rep=a.job.split('_');severity=c['future']['severity'][int(si)];scenario={'family':family,'severity':severity,'strength':1-.2*severity if family=='brightness' else .04*severity if family=='noise' else .8*severity,'seed':c['future']['seed_base']+int(rep),'realization':int(rep)}
  frozen=json.loads((out/'frozen.json').read_text());assert sha(out/'frozen.pt')==frozen['sha256']
 x=shift(d['x'],scenario['family'],scenario['strength'],scenario['seed']);projection,losses=update(clf,x,d['y'],c,a.device);path=out/'checkpoints'/f'{a.job}.pt';torch.save({'projection':projection},path)
 dump(out/'jobs'/a.job/'model.json',{'checkpoint':str(path),'checkpoint_sha256':sha(path),'scenario':scenario,'losses':losses})
 if a.phase=='future':
  rows=[]
  for name,item in load(out/'frozen.pt').items():
   p=clf.logits_features(features(clf,item['x'],a.device).to(a.device),projection.to(a.device)).softmax(1)[:,0]
   rows.append({'method':name,**scenario,'future_validity':float((p>=.5).double().mean()),'mean_hinge':float(torch.relu(.5-p).double().mean()),'checkpoint':str(path)})
  pd.DataFrame(rows).to_csv(out/'jobs'/a.job/'metrics.csv',index=False)

def optimize(a,c):
 _,split,ri,bi=a.job.split('_');ri=int(ri);bi=int(bi);r=c['r_grid'][ri];beta=c['beta_grid'][bi];out=a.output;pop=load(out/f'{split}_population.pt');z=pop['base_w'].to(a.device);target=load(out/'references.pt')['w'].to(a.device);plan=exact_w2(z,target);D=plan.distance;perm=torch.empty(len(z),dtype=torch.long);perm[plan.source_index]=plan.target_index;target=target[perm.to(a.device)]
 bank=FaceBank(records(out,r),a.device);candidates=[{'points':((1-lam)*z+lam*target).cpu(),'origin':f'geodesic{lam}'} for lam in c['geodesic_candidates']];trace=[]
 for name,initial in [('midpoint',(z+target)/2),('target',target)]:
  kept,history=solve_raw(bank,z,target,initial,D,beta*D,float('-inf'),float('inf'),c['optimizer'],on_step=lambda row:print(a.job,name,row['step'],row['true_bank_hinge'],row['slack'],flush=True) if row['step']%10==0 else None)
  candidates.extend([{'points':v['points'],'origin':f'{name}_{v["step"]}'} for v in kept]);trace.extend([{**row,'start':name} for row in history])
 rows=[]
 for i,v in enumerate(candidates):
  q=v['points'];length=exact_w2(z,q).distance+exact_w2(q,target).distance;x=bank.canonical(q);v['x']=x;v['features']=bank.cached_features(q.to(a.device)).cpu()
  rows.append({'candidate':i,'length':length,'risk':bank.risk(q.to(a.device)),'pixel_cost':float((x-pop['factual_x']).flatten(1).norm(dim=1).mean()),'origin':v['origin']})
 job=out/'jobs'/a.job;torch.save(candidates,job/'candidates.pt');pd.DataFrame(rows).to_csv(job/'candidates.csv',index=False);pd.DataFrame(trace).to_csv(job/'trace.csv',index=False);dump(job/'meta.json',{'r':r,'beta_geo':beta,'D':D,'bank_ids':[v['candidate'] for v in records(out,r)]})

def pool(a,c):
 _,split,ri=a.job.split('_');ri=int(ri);out=a.output;all_candidates=[];rows=[]
 for bi in range(len(c['beta_grid'])):
  job=out/'jobs'/f'opt_{split}_{ri}_{bi}';items=load(job/'candidates.pt');table=pd.read_csv(job/'candidates.csv');D=json.loads((job/'meta.json').read_text())['D']
  for row in table.to_dict('records'):row['candidate']+=len(all_candidates);rows.append(row)
  all_candidates+=items
 selected={};geom=[]
 for beta in c['beta_grid']:
  feasible=[v for v in rows if v['length']<=beta*D+1e-6*max(1,D)];best=min(feasible,key=lambda v:(v['risk'],v['pixel_cost'],v['candidate']));selected[str(beta)]=all_candidates[best['candidate']]
  geom.append({'r':c['r_grid'][ri],'beta_geo':beta,'D':D,'B_geo':beta*D,'total_length':best['length'],'slack':beta*D-best['length'],'Phi_hat_r':best['risk'],'pixel_L2_cost':best['pixel_cost'],'geometry_feasible':True})
 torch.save(selected,out/'jobs'/a.job/'selected.pt');pd.DataFrame(geom).to_csv(out/'jobs'/a.job/'geometry.csv',index=False)

def select(a,c):
 out=a.output;clf=SmileClassifier(a.device);rows=[]
 for ri,r in enumerate(c['r_grid']):
  selected=load(out/'jobs'/f'pool_calibration_{ri}'/'selected.pt');geom=pd.read_csv(out/'jobs'/f'pool_calibration_{ri}'/'geometry.csv')
  for beta in c['beta_grid']:
   item=selected[str(beta)];scores=[];valid=[]
   for i in range(len(c['calibration_scenarios'])):
    projection=load(out/'checkpoints'/f'calmodel_{i}.pt')['projection'].to(a.device)
    with torch.no_grad():p=clf.logits_features(item['features'].to(a.device),projection).softmax(1)[:,0]
    scores.append(float(torch.relu(.5-p).double().mean()));valid.append(float((p>=.5).double().mean()))
   row=geom[geom.beta_geo==beta].iloc[0].to_dict();rows.append({**row,'r_index':ri,'calibration_max_mean_hinge':max(scores),'calibration_mean_validity':np.mean(valid),'scenario_hinges':json.dumps(scores)})
 pd.DataFrame(rows).to_csv(out/'calibration_results.csv',index=False);best=min(rows,key=lambda v:(round(v['calibration_max_mean_hinge'],6),v['pixel_L2_cost'],v['r'],v['beta_geo']))
 dump(out/'selected_configuration.json',{'r_star':best['r'],'r_index':best['r_index'],'beta_geo_star':best['beta_geo'],'selected_utc':stamp(),'future_outcomes_used':False,'rule':c['selection'],'calibration_sha256':sha(out/'calibration_results.csv')})

def freeze(a,c):
 import lpips
 from torchmetrics.functional.image import structural_similarity_index_measure
 out=a.output;spec=json.loads((out/'selected_configuration.json').read_text());ri=spec['r_index'];pop=load(out/'test_population.pt');selected=load(out/'jobs'/f'pool_test_{ri}'/'selected.pt');bank=FaceBank(records(out,spec['r_star']),a.device)
 methods={'Base':{'points':pop['base_w'],'x':pop['base_x']},'LeapFactual':{'points':pop['robust_w'],'x':pop['robust_x']}}
 methods.update({f'GRACE_beta{b:.2f}':selected[str(b)] for b in c['beta_grid']});methods['GRACE_selected']=selected[str(spec['beta_geo_star'])]
 torch.hub.set_dir('/tmp/ffhq_torch_hub');metric=lpips.LPIPS(net='squeeze').to(a.device).eval();quality=[];risk=[]
 factual_native=decode(bank.g,pop['factual_w'],a.device)
 for name,item in methods.items():
  native=decode(bank.g,item['points'],a.device);x=item['x'];lp=[];ssim=[]
  with torch.no_grad():
   for cf,factual in zip(native,factual_native):
    lp.append(float(metric(cf[None].clamp(-1,1).to(a.device),factual[None].clamp(-1,1).to(a.device))));ssim.append(float(structural_similarity_index_measure(cf[None].clamp(-1,1).to(a.device),factual[None].clamp(-1,1).to(a.device),data_range=2.)))
  normalized=(native.clamp(-1,1)+1)/2;facts=(factual_native.clamp(-1,1)+1)/2;mse=float((normalized-facts).square().mean());quality.append({'method':name,'pixel_L2_224':float((x-pop['factual_x']).flatten(1).norm(dim=1).mean()),'pixel_L2_1024':float((normalized-facts).flatten(1).norm(dim=1).mean()),'latent_displacement':float((item['points']-pop['factual_w']).norm(dim=1).mean()),'LPIPS_squeeze':np.mean(lp),'SSIM':np.mean(ssim),'PSNR':-10*np.log10(max(mse,1e-15))})
  risk.append({'method':name,'Phi_hat_r':bank.risk(item['points'].to(a.device))});item['features']=bank.cached_features(item['points'].to(a.device)).cpu()
  np.save(out/'qualitative_examples'/f'{name}.npy',normalized[:4].numpy())
 np.save(out/'qualitative_examples/factual.npy',((factual_native[:4].clamp(-1,1)+1)/2).numpy());torch.save(methods,out/'frozen.pt');pd.DataFrame(quality).to_csv(out/'image_quality.csv',index=False);pd.DataFrame(risk).to_csv(out/'finite_bank_risk.csv',index=False)
 pd.DataFrame([{'method':name,'identity_score':None,'status':'unavailable: no original identity metric found; no identity preservation claim'} for name in methods]).to_csv(out/'identity_metrics.csv',index=False)
 shutil.copyfile(out/'jobs'/f'pool_test_{ri}'/'geometry.csv',out/'geometry_feasibility.csv');all_records=records(out,max(c['r_grid']));pd.DataFrame(all_records).to_csv(out/'attack_bank_manifest.csv',index=False)
 dump(out/'frozen.json',{'sha256':sha(out/'frozen.pt'),'selected_sha256':sha(out/'selected_configuration.json'),'bank_sha256':sha(out/'attack_bank_manifest.csv'),'frozen_utc':stamp(),'future_outcomes_seen':False})
 clock=json.loads((out/'construction_clock.json').read_text());clock.update(end_unix=time.time());clock['wall_seconds']=clock['end_unix']-clock['start_unix'];dump(out/'construction_clock.json',clock)

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','sample','prepare','base','attack','calmodel','optimize','pool','select','freeze','future']);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();out=a.output
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.16);seed(20260916)
 c=json.loads((out/'resolved_config.json').read_text()) if (out/'resolved_config.json').exists() else json.loads(CONFIG.read_text())
 (out/'jobs').mkdir(exist_ok=True);job=out/'jobs'/a.job;job.mkdir(exist_ok=True)
 if (job/'complete.json').exists():print('SKIP',a.job);return
 start=time.time();meta={'job':a.job,'phase':a.phase,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'hostname':socket.gethostname(),'seed':c['seed'],'start_utc':stamp(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'config':c};dump(job/'started.json',meta)
 if a.phase in ['calmodel','future']:shifted_model(a,c)
 else:globals()[a.phase](a,c)
 torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9})
if __name__=='__main__':main()
