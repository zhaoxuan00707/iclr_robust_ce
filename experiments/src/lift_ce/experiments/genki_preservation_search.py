"""Feasible per-face descent followed by population-objective coordinate selection."""
import argparse,json,os,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import genki_grace_identity_v1 as old
from . import genki_grace_beta_extension as previous
from .galaxy_grace_v1 import load,dump,sha,stamp
from ..counterfactuals.geodesic_excess import exact_w2
ROOT=Path(__file__).resolve().parents[4]

def initialize(a,c):
 previous.initialize(a,c)
 dump(a.output/'protocol_frozen.json',{'utc':stamp(),'config_sha256':sha(a.output/'resolved_config.json'),'fixed_cases':c['fixed_cases'],'r':2,'beta':3,'future_used_for_selection':False,'scope':c['scope']})

def search(a,c):
 out=a.output;i=int(a.job.split('_')[-1]);job=out/'jobs'/a.job
 facts=load(out/'test_factuals.pt');inc=load(Path(c['parent_run'])/'jobs/pool_test_1/selected.pt')['1.2']
 m=old.GenkiMeasurement(a,c,facts['w'][i:i+1].to(a.device));idx=load(out/'checkpoints/controls.pt')['style4']['indices'].to(a.device);std=load(out/'checkpoints/statistics.pt')['style_std'].to(a.device)
 states=[load(Path(r['checkpoint'])) for r in old.records(out,2.)];others=inc['features'].to(a.device)
 q=(inc['points'][i:i+1]-facts['u'][i:i+1]).to(a.device);z=load(out/'test_population.pt')['u'];target=load(out/'references.pt')['u'];D=exact_w2(z,target).distance
 def risk(feature):
  features=torch.cat([others[:i],feature,others[i+1:]])
  return old.population_risk(old.state_scores(features,states).sigmoid())
 def feature(q):return m.clf.features(m.clf.canonical(m.decode(q,idx,std)))
 candidates=[];trace=[]
 def save(q,row,rgb,f):
  j=len(candidates);path=out/'qualitative_examples'/f'proposal_{i}_{j}.png';Image.fromarray(rgb).save(path)
  candidates.append({'point':(q.cpu()+facts['u'][i:i+1]),'features':f.detach().cpu(),'quality':row,'image':str(path)})
 row,rgb,_,_=m.measure(q,idx,std);assert old.passes_quality(row,c)
 with torch.no_grad():current_f=feature(q);current_risk=float(risk(current_f))
 save(q,row,rgb,current_f);candidates[0]['point']=inc['points'][i:i+1].clone();step=c['search']['initial_step']
 for iteration in range(0 if i in c['fixed_cases'] else c['search']['steps']):
  variable=q.detach().requires_grad_(True);loss=risk(feature(variable));gradient,=torch.autograd.grad(loss,variable);assert torch.isfinite(gradient).all()
  if gradient.norm()<1e-12:break
  proposal=(q-step*gradient/(gradient.norm()+1e-12)).clamp(-16,16).detach()
  points=inc['points'].clone();points[i:i+1]=proposal.cpu()+facts['u'][i:i+1]
  length=exact_w2(z,points).distance+exact_w2(points,target).distance
  row,rgb,_,_=m.measure(proposal,idx,std)
  with torch.no_grad():f=feature(proposal);value=float(risk(f))
  quality=old.passes_quality(row,c);accept=quality and length<=3*D+1e-6 and value<current_risk-1e-9
  trace.append(dict(iteration=iteration,step=step,risk=value,previous_risk=current_risk,length=length,quality_pass=quality,accepted=accept,**row))
  if accept:q=proposal;current_risk=value;save(q,row,rgb,f);step=min(c['search']['maximum_step'],step*c['search']['growth'])
  else:step*=c['search']['rejection_factor']
  print(a.job,iteration,'accepted',accept,'risk',current_risk,'step',step,flush=True)
  if step<c['search']['minimum_step']:break
 torch.save(candidates,job/'candidates.pt');pd.DataFrame(trace).to_csv(job/'trace.csv',index=False)
 dump(job/'summary.json',{'factual_index':i,'fixed':i in c['fixed_cases'],'accepted_updates':len(candidates)-1,'attempts':len(trace),'final_conditional_population_risk':current_risk})

def combine(a,c):
 out=a.output;facts=load(out/'test_factuals.pt');z=load(out/'test_population.pt')['u'];target=load(out/'references.pt')['u'];D=exact_w2(z,target).distance
 states=[load(Path(r['checkpoint'])) for r in old.records(out,2.)];allc=[load(out/'jobs'/f'search_{i}'/'candidates.pt') for i in range(8)]
 points=torch.cat([items[0]['point'] for items in allc]);features=torch.cat([items[0]['features'] for items in allc]);selected=[0]*8
 def risk(f):return float(old.population_risk(old.state_scores(f,states).sigmoid()))
 current=risk(features);trace=[]
 for iteration in range(100):
  best=None
  for i,items in enumerate(allc):
   if i in c['fixed_cases']:continue
   for j,item in enumerate(items):
    f=features.clone();f[i:i+1]=item['features'];value=risk(f)
    if value>=current-1e-9:continue
    q=points.clone();q[i:i+1]=item['point'];length=exact_w2(z,q).distance+exact_w2(q,target).distance
    if length>3*D+1e-6:continue
    if best is None or value<best[0]:best=(value,i,j,q,f,length)
  if best is None:break
  current,i,j,points,features,length=best;selected[i]=j;trace.append(dict(iteration=iteration,factual_index=i,proposal=j,risk=current,length=length))
 inc=load(Path(c['parent_run'])/'jobs/pool_test_1/selected.pt')['1.2']
 for i in c['fixed_cases']:assert torch.equal(points[i],inc['points'][i])
 torch.save([{'points':points,'features':features}],out/'candidates.pt')
 dump(out/'selection_frozen.json',{'utc':stamp(),'selected_proposals':selected,'bank_risk':current,'future_outcomes_used':False,'fixed_cases':c['fixed_cases']})
 pd.DataFrame(trace).to_csv(out/'combination_trace.csv',index=False)
 pd.DataFrame([dict(beta_geo=3,D=D,B_geo=3*D,total_length=exact_w2(z,points).distance+exact_w2(points,target).distance,bank_hinge=current)]).to_csv(out/'geometry.csv',index=False)

def freeze(a,c):
 out=a.output;parent=Path(c['parent_run']);rows=[];prev=pd.read_csv(parent/'image_quality.csv');quality=pd.concat([pd.read_csv(out/'jobs'/f'quality_{i}'/'quality.csv') for i in range(8)])
 assert quality.quality_pass.all()
 for i in range(8):
  for name,oldname in [('Factual','factual'),('Base','Base'),('Previous','GRACE_beta1.20')]:
   shutil.copyfile(parent/'qualitative_examples'/f'test_{i}_{oldname}.png',out/'qualitative_examples'/f'test_{i}_{name}.png')
   if name!='Factual':
    r=prev[(prev.method==oldname)&(prev.factual_index==i)].iloc[0].to_dict();r['method']=name;rows.append(r)
  name='Feasible_search';shutil.copyfile(out/'qualitative_examples'/f'candidate_0_{i}.png',out/'qualitative_examples'/f'test_{i}_{name}.png')
  r=quality[quality.factual_index==i].iloc[0].to_dict();r['method']=name;rows.append(r)
  if i in c['fixed_cases']:assert sha(out/'qualitative_examples'/f'test_{i}_{name}.png')==sha(out/'qualitative_examples'/f'test_{i}_Previous.png')
 pd.DataFrame(rows).to_csv(out/'image_quality.csv',index=False)
 dump(out/'outputs_frozen.json',{'utc':stamp(),'image_hashes':{str(p):sha(p) for p in (out/'qualitative_examples').glob('test_*.png')}})

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','search','combine','quality','freeze']);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(20260918);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 cfg=a.output/'resolved_config.json';c=json.loads((cfg if cfg.exists() else ROOT/'experiments/configs/genki_preservation_search.json').read_text());torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
 job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
 if (job/'complete.json').exists():return
 start=time.time();meta={'phase':a.phase,'job':a.job,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta)
 if a.phase=='quality':previous.quality(a,c)
 else:globals()[a.phase](a,c)
 torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start})
if __name__=='__main__':main()
