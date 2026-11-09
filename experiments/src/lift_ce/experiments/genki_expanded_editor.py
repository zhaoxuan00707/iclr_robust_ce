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

from ..ffhq.style_editor import region_mask

def initialize(a,c):
 previous.initialize(a,c)
 parent=Path(c['parent_feasible']);manifest=json.loads((a.output/'input_manifest.json').read_text())
 for p in [parent/'candidates.pt',parent/'outputs_frozen.json',Path(c['parent_ffhq'])/'training.pt']:
  manifest[str(p)]=sha(p)
 dump(a.output/'input_manifest.json',manifest)
 dump(a.output/'protocol_frozen.json',{'utc':stamp(),'config_sha256':sha(a.output/'resolved_config.json'),'scope':c['scope'],'targets':c['targets'],'channels':c['style_counts'],'all_eight_cases_mutable':True})

def discover(a,c):
 shard=int(a.job.split('_')[-1]);train=load(Path(c['parent_ffhq'])/'training.pt');std=load(a.output/'checkpoints/statistics.pt')['style_std'].to(a.device)
 g=old.Generator(a.device);editor=old.StyleEditor(g);clf=old.classifier(a);rows=[];outside=1-region_mask(224,a.device)
 for i in range(shard,c['discovery_training_count'],8):
  w=train['w'][i:i+1].to(a.device)
  with torch.no_grad():s0=editor.encode(w)
  q=torch.zeros_like(s0,requires_grad=True)
  def output(q):return clf.canonical(editor.decode(w,s0+q*std))
  x=output(q);p=clf.logits_features(clf.features(x)).softmax(1)[0,0];grad=torch.autograd.grad(p,q)[0];energies=[]
  assert torch.isfinite(grad).all()
  for probe in range(c['outside_jacobian_probes']):
   x=output(q);rng=torch.Generator(device=a.device).manual_seed(c['seed']+100*i+probe);noise=(torch.randint(0,2,x.shape,device=a.device,generator=rng)*2-1).float()*outside
   derivative=torch.autograd.grad((x*noise).sum()/torch.sqrt(outside.sum()*3),q)[0];energies.append(derivative.square())
  rows.append({'index':i,'gradient':grad.cpu(),'outside':torch.stack(energies).mean(0).cpu()});print(a.job,i,flush=True)
 torch.save(rows,a.output/'jobs'/a.job/'discovery.pt')

def controls(a,c):
 out=a.output;rows=sum([load(out/'jobs'/f'discover_{i}'/'discovery.pt') for i in range(8)],[]);assert sorted(r['index'] for r in rows)==list(range(32))
 grads=torch.cat([r['gradient'] for r in rows]);leak=torch.cat([r['outside'] for r in rows]);mean=grads.mean(0);agreement=(grads.sign()==mean.sign()).float().mean(0)
 g=old.Generator(a.device);editor=old.StyleEditor(g);eligible=torch.zeros(len(mean),dtype=torch.bool)
 for j in c['eligible_layers']:eligible[editor.offsets[j]:editor.offsets[j+1]]=True
 eligible &= agreement>=c['minimum_gradient_sign_agreement'];score=mean.abs()/(leak.mean(0).sqrt()+1e-5);score[~eligible]=-float('inf')
 original=load(out/'checkpoints/controls.pt')['style4']['indices'].tolist();extra=[i for i in score.argsort(descending=True).tolist() if eligible[i] and i not in original];assert len(extra)>=12
 indices=torch.tensor(original+extra[:12]);torch.save({'indices':indices},out/'expanded_controls.pt')
 pd.DataFrame([dict(rank=k,index=i,score=float(score[i]),agreement=float(agreement[i]),retained_original=i in original) for k,i in enumerate(indices.tolist())]).to_csv(out/'channel_ranking.csv',index=False)
 dump(out/'controls_frozen.json',{'utc':stamp(),'sha256':sha(out/'expanded_controls.pt'),'training_indices':list(range(32)),'test_used':False,'calibration_used':False,'original_four_retained':original,'nested_channels':indices.tolist()})

def prepare(a,c):
 out=a.output;idx=load(out/'expanded_controls.pt')['indices'];std=load(out/'checkpoints/statistics.pt')['style_std'];facts=load(out/'test_factuals.pt');refs=load(out/'references.pt');base=load(out/'test_population.pt');inc=load(Path(c['parent_feasible'])/'candidates.pt')[0]
 g=old.Generator(a.device);editor=old.StyleEditor(g)
 with torch.no_grad():
  # Encode each face separately, matching the measurement realization path.
  factual=torch.cat([editor.encode(w[None].to(a.device)).cpu() for w in facts['w']])[:,idx]/std[idx]
  target=torch.cat([editor.encode(w[None].to(a.device)).cpu() for w in refs['w']])[:,idx]/std[idx]
 factual[:,:4]=facts['u'];target[:,:4]=refs['u'];z=factual.clone();z[:,:4]=base['u'];u=factual.clone();u[:,:4]=inc['points']
 facts['u']=factual;torch.save(facts,out/'expanded_factuals.pt');torch.save({'u':target},out/'expanded_references.pt');torch.save({'u':z},out/'expanded_base.pt');torch.save({'points':u,'features':inc['features']},out/'expanded_incumbent.pt')
 D=exact_w2(z,target).distance;length=exact_w2(z,u).distance+exact_w2(u,target).distance;assert length<=3*D
 dump(out/'geometry_setup.json',{'D':D,'initial_length':length,'beta':3,'common_dimensions':16,'inactive_coordinates_factual':True})

def arm(a):
 parts=a.job.split('_');return int(parts[1]),int(parts[2])/10,f'c{parts[1]}_t{parts[2]}'

def search(a,c):
 out=a.output;i=int(a.job.split('_')[-1]);job=out/'jobs'/a.job;count,tau,label=arm(a)
 facts=load(out/'expanded_factuals.pt');inc=load(out/'expanded_incumbent.pt')
 m=old.GenkiMeasurement(a,c,facts['w'][i:i+1].to(a.device));idx=load(out/'expanded_controls.pt')['indices'].to(a.device);std=load(out/'checkpoints/statistics.pt')['style_std'].to(a.device)
 states=[load(Path(r['checkpoint'])) for r in old.records(out,2.)];others=inc['features'].to(a.device)
 q=(inc['points'][i:i+1]-facts['u'][i:i+1]).to(a.device);z=load(out/'expanded_base.pt')['u'];target=load(out/'expanded_references.pt')['u'];D=exact_w2(z,target).distance
 def risk(feature):
  features=torch.cat([others[:i],feature,others[i+1:]])
  return torch.relu(tau-old.state_scores(features,states).sigmoid()).double().mean(1).max()
 def feature(q):return m.clf.features(m.clf.canonical(m.decode(q,idx,std)))
 candidates=[];trace=[]
 def save(q,row,rgb,f):
  j=len(candidates);path=out/'qualitative_examples'/f'proposal_{label}_{i}_{j}.png';Image.fromarray(rgb).save(path)
  candidates.append({'point':(q.cpu()+facts['u'][i:i+1]),'features':f.detach().cpu(),'quality':row,'image':str(path)})
 row,rgb,_,_=m.measure(q,idx,std);assert old.passes_quality(row,c)
 with torch.no_grad():current_f=feature(q);current_risk=float(risk(current_f))
 save(q,row,rgb,current_f);candidates[0]['point']=inc['points'][i:i+1].clone();step=c['search']['initial_step']
 for iteration in range(0 if i in c['fixed_cases'] else c['search']['steps']):
  variable=q.detach().requires_grad_(True);loss=risk(feature(variable));gradient,=torch.autograd.grad(loss,variable);assert torch.isfinite(gradient).all();gradient[:,count:]=0
  if gradient.norm()<1e-12:break
  proposal=(q-step*gradient/(gradient.norm()+1e-12)).clamp(-16,16).detach()
  points=inc['points'].clone();points[i:i+1]=proposal.cpu()+facts['u'][i:i+1]
  length=exact_w2(z,points).distance+exact_w2(points,target).distance
  assert torch.equal(proposal[:,count:],q[:,count:]);row,rgb,_,_=m.measure(proposal,idx,std)
  with torch.no_grad():f=feature(proposal);value=float(risk(f))
  quality=old.passes_quality(row,c);accept=quality and length<=3*D+1e-6 and value<current_risk-1e-9
  trace.append(dict(iteration=iteration,step=step,risk=value,previous_risk=current_risk,length=length,quality_pass=quality,accepted=accept,**row))
  if accept:q=proposal;current_risk=value;save(q,row,rgb,f);step=min(c['search']['maximum_step'],step*c['search']['growth'])
  else:step*=c['search']['rejection_factor']
  print(a.job,iteration,'accepted',accept,'risk',current_risk,'step',step,flush=True)
  if step<c['search']['minimum_step']:break
 torch.save(candidates,job/'candidates.pt');pd.DataFrame(trace).to_csv(job/'trace.csv',index=False)
 dump(job/'summary.json',{'channels':count,'target':tau,'factual_index':i,'fixed':i in c['fixed_cases'],'accepted_updates':len(candidates)-1,'attempts':len(trace),'final_conditional_population_risk':current_risk})

def combine(a,c):
 out=a.output;count,tau,label=arm(a);facts=load(out/'expanded_factuals.pt');z=load(out/'expanded_base.pt')['u'];target=load(out/'expanded_references.pt')['u'];D=exact_w2(z,target).distance
 states=[load(Path(r['checkpoint'])) for r in old.records(out,2.)];allc=[load(out/'jobs'/f'search_{count}_{int(tau*10)}_{i}'/'candidates.pt') for i in range(8)]
 points=torch.cat([items[0]['point'] for items in allc]);features=torch.cat([items[0]['features'] for items in allc]);selected=[0]*8
 def risk(f):return float(torch.relu(tau-old.state_scores(f,states).sigmoid()).double().mean(1).max())
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
 inc=load(out/'expanded_incumbent.pt');assert torch.equal(points[:,count:],facts['u'][:,count:])
 for i in c['fixed_cases']:assert torch.equal(points[i],inc['points'][i])
 torch.save([{'points':points,'features':features}],out/f'{label}_selected.pt')
 dump(out/f'{label}_selection_frozen.json',{'utc':stamp(),'selected_proposals':selected,'bank_risk':current,'future_outcomes_used':False,'fixed_cases':c['fixed_cases']})
 pd.DataFrame(trace).to_csv(out/f'{label}_combination_trace.csv',index=False)
 pd.DataFrame([dict(channels=count,target=tau,beta_geo=3,D=D,B_geo=3*D,total_length=exact_w2(z,points).distance+exact_w2(points,target).distance,bank_hinge=current)]).to_csv(out/f'{label}_geometry.csv',index=False)


def quality(a,c):
 out=a.output;count,tau,label=arm(a);i=int(a.job.split('_')[-1]);facts=load(out/'expanded_factuals.pt');point=load(out/f'{label}_selected.pt')[0]['points'][i:i+1]
 m=old.GenkiMeasurement(a,c,facts['w'][i:i+1].to(a.device));idx=load(out/'expanded_controls.pt')['indices'].to(a.device);std=load(out/'checkpoints/statistics.pt')['style_std'].to(a.device)
 row,rgb,arc,face=m.measure((point-facts['u'][i:i+1]).to(a.device),idx,std);assert old.passes_quality(row,c)
 Image.fromarray(rgb).save(out/'qualitative_examples'/f'test_{i}_{label}.png');pd.DataFrame([dict(method=label,factual_index=i,quality_pass=True,**row)]).to_csv(out/'jobs'/a.job/'quality.csv',index=False)
 np.savez(out/'jobs'/a.job/'embeddings.npz',arc_factual=m.arc0,arc=arc,facenet_factual=m.face0,facenet=face)

def freeze(a,c):
 out=a.output;parent=Path(c['parent_feasible']);rows=[];prev=pd.read_csv(parent/'image_quality.csv')
 for i in range(8):
  for name,oldname in [('Factual','Factual'),('Base','Base'),('Previous','Feasible_search')]:
   shutil.copyfile(parent/'qualitative_examples'/f'test_{i}_{oldname}.png',out/'qualitative_examples'/f'test_{i}_{name}.png')
   if name!='Factual':
    r=prev[(prev.method==oldname)&(prev.factual_index==i)].iloc[0].to_dict();r['method']=name;rows.append(r)
 for tau in c['targets']:
  for count in c['style_counts']:
   for i in range(8):rows+=pd.read_csv(out/'jobs'/f'quality_{count}_{int(tau*10)}_{i}'/'quality.csv').to_dict('records')
 pd.DataFrame(rows).to_csv(out/'image_quality.csv',index=False)
 dump(out/'outputs_frozen.json',{'utc':stamp(),'image_hashes':{str(p):sha(p) for p in (out/'qualitative_examples').glob('test_*.png')}})

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','discover','controls','prepare','search','combine','quality','freeze']);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(20260918);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 cfg=a.output/'resolved_config.json';c=json.loads((cfg if cfg.exists() else ROOT/'experiments/configs/genki_expanded_editor.json').read_text());torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
 job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
 if (job/'complete.json').exists():return
 start=time.time();meta={'phase':a.phase,'job':a.job,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta);globals()[a.phase](a,c)
 torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start})
if __name__=='__main__':main()
