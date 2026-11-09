"""Matched strong/weak classifier stress experiment; checkpoint choice precedes CFs."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import MNIST
from ..models.vision import SmallCNN
from ..config import BaseCEConfig
from ..counterfactuals.vision import generate_near_boundary_ce
from ..counterfactuals.update_aware import surrogate
from .stylemnist_grace_v1 import assets,load,sha,dump
from .mnist_latent_base_fix import optimize
from .mnist_small_radius import futures,merge,score
ROOT=Path(__file__).resolve().parents[4];CONFIG=ROOT/'experiments/configs/mnist_weaker_classifier.json'

def prediction(m,x,device):
 with torch.no_grad():return torch.cat([m(v.to(device)).sigmoid().cpu() for v in x.split(128)])
def diagnostics(m,x,y,device):
 p=prediction(m,x,device);z=p>=.5;return {'accuracy':float((z==y.bool()).float().mean()),'recall4':float((~z[y==0]).float().mean()),'recall9':float(z[y==1].float().mean()),'bce':float(torch.nn.functional.binary_cross_entropy(p,y))}

def prepare(a,c):
 d=load(Path(c['parent_run'])/'population.pt');old=load(Path(c['old_run'])/'population.pt');ids=torch.from_numpy(np.load(Path(c['old_run'])/'train_ids.npy'));excluded=set(d['train_ids'].tolist()+d['target_ids'].tolist());available=torch.tensor([int(i) not in excluded for i in ids]);v=torch.cat([torch.where(available&(old['y']==k))[0][:256] for k in [0,1]]);t=torch.cat([torch.where(d['y']==k)[0][:64] for k in [0,1]]);assert len(v)==512 and len(t)==128
 torch.save({'x':d['x'][t],'y':d['y'][t],'ids':d['train_ids'][t],'vx':old['x'][v],'vy':old['y'][v],'vids':ids[v]},a.output/'classifier_data.pt');assert not set(ids[v].tolist())&set(d['train_ids'].tolist())
 m,_,_=assets(c,a.device);dump(a.output/'strong_validation.json',diagnostics(m,old['x'][v],old['y'][v],a.device));manifest=json.loads((Path(c['parent_run'])/'input_manifest.json').read_text())
 for p in [CONFIG,ROOT/c['task'],Path(c['parent_run'])/'population.pt',Path(c['old_run'])/'train_ids.npy',Path(c['old_run'])/'calibration_ids.npy']:manifest[str(p)]=sha(p)
 for p,h in manifest.items():assert sha(Path(p))==h
 dump(a.output/'input_manifest.json',manifest);dump(a.output/'resolved_config.json',c)

def train(a,c):
 d=load(a.output/'classifier_data.pt');seed=c['weak_training']['seeds'][a.worker];torch.manual_seed(seed);m=SmallCNN().to(a.device);opt=torch.optim.Adam(m.parameters(),lr=.001);g=torch.Generator().manual_seed(seed);folder=a.output/'candidates';folder.mkdir(exist_ok=True);rows=[]
 for step in range(1,129):
  ix=torch.randperm(128,generator=g)[:32];m.train();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.binary_cross_entropy_with_logits(m(d['x'][ix].to(a.device)),d['y'][ix].to(a.device));loss.backward();opt.step()
  if step in c['weak_training']['checkpoints']:
   m.eval();row=diagnostics(m,d['vx'],d['vy'],a.device);path=folder/f'seed{seed}_step{step}.pt';torch.save({k:v.cpu() for k,v in m.state_dict().items()},path);rows.append({**row,'seed':seed,'step':step,'path':str(path),'sha256':sha(path)})
 pd.DataFrame(rows).to_csv(a.output/f'candidate_worker{a.worker}.csv',index=False)

def choose(a,c):
 d=pd.concat([pd.read_csv(a.output/f'candidate_worker{i}.csv') for i in range(8)]);strong=json.loads((a.output/'strong_validation.json').read_text());d['eligible']=(d.accuracy>=.8)&(d.accuracy<=.95)&(d.accuracy<=strong['accuracy']-.03);d['distance_to_target']=(d.accuracy-.9).abs();d.to_csv(a.output/'candidate_summary.csv',index=False);eligible=d[d.eligible];assert len(eligible),'No predeclared weaker checkpoint qualifies';row=eligible.sort_values(['distance_to_target','step','seed']).iloc[0].to_dict();dump(a.output/'weak_selection.json',row);payload=load(Path(c['checkpoint']));assert sha(Path(row['path']))==row['sha256'];payload['model']=load(Path(row['path']));torch.save(payload,a.output/'weak_payload.pt');dump(a.output/'classifier_frozen.json',{'weak_payload_sha256':sha(a.output/'weak_payload.pt'),'selection_sha256':sha(a.output/'weak_selection.json'),'strong_checkpoint_sha256':sha(Path(c['checkpoint']))});print('WEAK CLASSIFIER',row,flush=True)

def population(a,c):
 root=a.output;d=load(Path(c['parent_run'])/'population.pt');old=load(Path(c['old_run'])/'population.pt');strong,ae,_=assets(c,a.device);wc={**c,'checkpoint':str(root/'weak_payload.pt')};weak,_,_=assets(wc,a.device);models=[strong,weak];raw=MNIST(c['data_dir'],train=True,download=False);test=MNIST(c['data_dir'],train=False,download=False);calids=np.load(Path(c['old_run'])/'calibration_ids.npy');shared={k:d[k] for k in ['x','y','train_ids']};common={}
 for split,ds,pool in [('cal',raw,calids),('test',test,np.where(test.targets.numpy()==4)[0])]:
  ordered=list(dict.fromkeys(d[split]['ids'].tolist()+list(map(int,pool))));ids=torch.tensor(ordered);ids=ids[ds.targets[ids]==4];x=ds.data[ids,None].float()/255;mask=(prediction(strong,x,a.device)<.5)&(prediction(weak,x,a.device)<.5);chosen=torch.where(mask)[0][:64];assert len(chosen)==64;common[split]={'factual':x[chosen],'ids':ids[chosen]}
 ids=torch.from_numpy(np.load(Path(c['old_run'])/'train_ids.npy'));mask=old['y']==1;validation_ids=set(load(root/'classifier_data.pt')['vids'].tolist());mask&=torch.tensor([int(i) not in validation_ids for i in ids])
 for m in models:mask&=prediction(m,old['x'],a.device)>=.5
 candidates=torch.where(mask)[0];refs=old['x'][candidates]
 with torch.no_grad():decoded=torch.cat([ae.decode(ae.encode(v.to(a.device))).cpu() for v in refs.split(128)])
 good=(prediction(strong,decoded,a.device)>=.6)&(prediction(weak,decoded,a.device)>=.6);chosen=candidates[good][:64];assert len(chosen)==64;shared.update(target=old['x'][chosen],target_ids=ids[chosen]);assert not set(common['cal']['ids'].tolist())&set(shared['train_ids'].tolist()+shared['target_ids'].tolist())
 validation_ids=set(load(root/'classifier_data.pt')['vids'].tolist());assert not validation_ids&set(common['cal']['ids'].tolist())
 with torch.no_grad():shared['target_z']=ae.encode(shared['target'].to(a.device)).cpu()
 for arm,m in zip(['strong','weak'],models):
  folder=root/arm;folder.mkdir();ac={**c,'checkpoint':c['checkpoint'] if arm=='strong' else str(root/'weak_payload.pt')};pop=copy.deepcopy(shared);audit=[]
  for split in ['cal','test']:
   p=copy.deepcopy(common[split]);b=generate_near_boundary_ce(m,ae,p['factual'],shared['target'],BaseCEConfig(steps=96,autoencoder_type='strong',latent_dim=64),torch.device(a.device));assert torch.isfinite(b.latents).all();p.update(base=b.counterfactuals,base_z=b.latents,historical_base=b.counterfactuals.clone());pop[split]=p
   with torch.no_grad():im=ae.decode(b.latents.to(a.device));assert torch.allclose(im.cpu(),b.counterfactuals,atol=1e-5,rtol=0);assert (m(im).sigmoid()>=.5).all()
   q=b.latents.to(a.device).requires_grad_();f,n,s=surrogate(m,pop['x'].to(a.device),pop['y'].to(a.device),q,.1,steps=3,lr=.01,decode=ae.decode,hinge_temperature=.05);grad=torch.autograd.grad(f,q)[0];direction=grad.detach()/grad.norm().clamp_min(1e-20);analytic=float(grad.norm());row={'split':split,'nominal_hinge':float(n.detach()),'sensitivity':float(s.detach()),'current_validity':1.,'directional_autograd':analytic};q=q.detach();del f,n,s,grad
   values=[]
   for sign in [-1,1]:
    f,n,s=surrogate(m,pop['x'].to(a.device),pop['y'].to(a.device),q+sign*.001*direction,.1,steps=3,lr=.01,decode=ae.decode,hinge_temperature=.05);values.append(float(f.detach()));del f,n,s
   fd=(values[1]-values[0])/.002;row.update(finite_difference=fd,relative_error=abs(fd-analytic)/max(analytic,1e-8));assert row['relative_error']<.15,(arm,row);audit.append(row)
  torch.save(pop,folder/'population.pt');dump(folder/'base_recovery.json',audit);dump(folder/'resolved_config.json',ac)
  manifest=json.loads((root/'input_manifest.json').read_text());manifest[str(root/'weak_payload.pt')]=sha(root/'weak_payload.pt');manifest[str(root/'weak_selection.json')]=sha(root/'weak_selection.json');dump(folder/'input_manifest.json',manifest)
 # Official test classifier diagnostics only after model choice is frozen.
 ix=torch.where((test.targets==4)|(test.targets==9))[0];xx=test.data[ix,None].float()/255;yy=(test.targets[ix]==9).float();dump(root/'classifier_test.json',{arm:diagnostics(m,xx,yy,a.device) for arm,m in zip(['strong','weak'],models)});dump(root/'population_checks.json',{'matched_factual_ids':True,'common_reference_ids':True,'common_update_ids':True,'cal_update_disjoint':True,'cal_validation_disjoint':True,'test_official_split':True,'base_valid_both_arms':True})

def collect(a,split):
 d=load(a.output/'population.pt');rows=[];images={'Base':d[split]['base']}
 for p in sorted((a.output/split).glob('*/metrics.json')):
  row=json.loads(p.read_text());row['method']=p.parent.name;rows.append(row)
  if row['feasible']:images[p.parent.name]=load(p.parent/'output.pt')['images']
 assert len(rows)==4
 return rows,images

def select(a,c):
 rows,images=collect(a,'cal');f=score(a,c,images,'cal');f.to_csv(a.output/'calibration_per_model.csv',index=False)
 for row in rows:
  v=f[f.method==row['method']];row['cal_FV']=float(v.future_validity.mean()) if len(v) else None
 frame=pd.DataFrame(rows);frame.to_csv(a.output/'calibration_summary.csv',index=False);best=frame[frame.feasible].sort_values(['cal_FV','pixel_cost','r'],ascending=[False,True,True]).iloc[0].to_dict();dump(a.output/'selection.json',best)

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
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','train','choose','population','futures','merge','optimize','select','evaluate'],required=True);p.add_argument('--device',default='cuda');p.add_argument('--worker',type=int,default=0);p.add_argument('--arm',choices=['strong','weak']);p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1.02);a=p.parse_args();c=json.loads(CONFIG.read_text());torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065)
 if a.arm:a.output=a.output/a.arm;c=json.loads((a.output/'resolved_config.json').read_text())
 print('RUN',vars(a),'GPU',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True);globals()[a.phase](a,c)
if __name__=='__main__':main()
