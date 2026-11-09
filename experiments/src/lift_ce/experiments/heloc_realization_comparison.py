"""Task005: frozen-Q realization comparison, with explicit unequal-mass scope."""
import argparse,json,hashlib,time,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..counterfactuals.geodesic_excess import exact_w2,glued_arclength_path
from ..counterfactuals.direct_realization import direct_realization
from ..counterfactuals.minimal_validity_repair import minimal_restore,validity
from ..counterfactuals.population_path import population_mean
from ..counterfactuals.robust_waypoints import ModelBank
from .heloc_diabetes_official_lift_v1 import load_model
from .fixed_risk_geometry_v1 import exact_lambda_choices
ROOT=Path(__file__).resolve().parents[4]

def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def frozen(c):return Path(c['raw_parent']),Path(c['old_parent'])
def pathof(base,q,target,lambdas):
 p0=exact_w2(base,q);p1=exact_w2(q,target)
 return glued_arclength_path(base,q,target,p0,p1,lambdas),p0,p1

def audit(a,c):
 out=a.output;raw,old=frozen(c);manifest={};checks=[]
 for parent,filename in [(raw,'construction_manifest.json'),(raw,'frozen_raw.json'),(old,'frozen_paths.json')]:
  for name,digest in read(parent/filename)['files'].items():
   assert sha(parent/name)==digest,(name,'hash mismatch');manifest[str(parent/name)]=digest
 for name in ['current.pt','training.pt','train_population.pt']:assert sha(raw/name)==sha(old/name)
 bank=read(raw/'old_bank.json')['candidates'];assert len(bank)==5927
 meta=read(raw/'instance_manifest.json');budget=3*meta['b_ref']
 for record in bank:
  path=record['model_path'];digest=sha(path);assert digest==record['model_sha256'];assert record['measured_budget']<=budget+budget*1e-6;manifest[path]=digest
 future=pd.read_csv(Path(meta['original_instance'])/'future_final_manifest.csv');assert len(future)==45
 archived_manifest=read(old/'evaluation_manifest.json')
 for path,digest in archived_manifest['model_hashes'].items():assert sha(path)==digest;manifest[path]=digest
 for name in ['selection.json','evaluation_bank.json','selected_future_results.csv','future_per_model.csv','bank_test_curves.csv','paths_test.pt','paths_cal.pt','instance_manifest.json']:
  manifest[str(old/name)]=sha(old/name)
 for split in ['train','cal','test']:manifest[str(old/f'{split}_population.pt')]=sha(old/f'{split}_population.pt')
 current=load_model(raw/'current.pt',22);training=load(raw/'training.pt');sizes=[]
 for split in ['train','cal','test']:
  pop=load(old/f'{split}_population.pt');assert len(pop['source_ids'].unique())==len(pop['source']);assert validity(current,pop['path']['base'])['invalid_support_count']==0
  sizes.append(dict(split=split,factuals=len(pop['source']),base_count=len(pop['path']['base']),Q_count=3336))
 archived=load(old/'paths_test.pt');selection=read(old/'selection.json');saved=pd.read_csv(old/'selected_future_results.csv');per=pd.read_csv(old/'future_per_model.csv');reproduction=[];old_outputs={}
 for beta in c['betas']:
  tag=round(beta*100);name=f'S_beta{tag:03d}';q=load(old/f'learned_control_distribution_beta{tag:03d}.pt')['points'];manifest[str(old/f'learned_control_distribution_beta{tag:03d}.pt')]=sha(old/f'learned_control_distribution_beta{tag:03d}.pt')
  lam=next(r['lambda'] for r in selection['selected'] if r['method']==name);pop=load(old/'test_population.pt')
  path,_,_=pathof(pop['path']['base'],q,training['target'],[lam]);li=c['lambda_grid'].index(lam);expected=archived['methods'][name]
  torch.testing.assert_close(path['q'][0],expected['q'][li],rtol=0,atol=0);torch.testing.assert_close(path['weights'],expected['weights'],rtol=0,atol=0);assert torch.equal(path['source_index'],expected['source_index'])
  x=path['q'][0];w=path['weights'];old_outputs[str(tag)]={'points':x,'weights':w,'source_index':path['source_index'],'lambda':lam,'method':name}
  cv=validity(current,x,w)['current_validity'];cost=float(population_mean((x-pop['source'][path['source_index']]).norm(dim=1).double(),w))
  for operator in c['future_operators']:
   values=[];hinges=[]
   for j,r in future.iterrows():
    mp=old/'aligned_future_models'/f'model{j}.pt' if operator=='aligned_full_update' else Path(r.path)
    m=load_model(mp,22)
    with torch.no_grad():prob=m(x).sigmoid();v=float(population_mean((prob>=.5).double(),w));h=float(population_mean(torch.relu(.5-prob).double(),w))
    expectedrow=per[(per.method==name)&(per.operator==operator)&(per['lambda']==lam)&(per.model_index==j)].iloc[0]
    assert abs(v-expectedrow.validity)<1e-7 and abs(h-expectedrow.hinge)<1e-7
    values.append(v);hinges.append(h)
   ref=saved[(saved.method==name)&(saved.operator==operator)].iloc[0]
   errors=[abs(cv-ref.current_validity),abs(cost-ref.recourse_cost),abs(np.mean(values)-ref.mean_future_validity),abs(np.mean(hinges)-ref.mean_future_hinge)];assert max(errors)<1e-7,errors
   reproduction.append(dict(beta_geo=beta,operator=operator,current_validity=cv,recourse_cost=cost,future_validity=np.mean(values),max_absolute_error=max(errors)))
  print('archived OLD reproduced',beta,flush=True)
 for beta in c['betas']:
  q=load(raw/f'raw_supports_U_beta{round(beta*100):03d}.pt');assert q['points'].shape==(3336,22);assert len(torch.unique(q['points'],dim=0))>128;torch.testing.assert_close(q['weights'],torch.full((3336,),1/3336,dtype=torch.float64))
 torch.save(old_outputs,out/'archived_selected.pt');pd.DataFrame(reproduction).to_csv(out/'old_reproduction.csv',index=False);pd.DataFrame(sizes).to_csv(out/'population_counts.csv',index=False)
 dump(out/'input_manifest.json',manifest);dump(out/'audit.json',{'archived_OLD_reproduced':True,'max_absolute_error':max(r['max_absolute_error'] for r in reproduction),'common_bank_count':len(bank),'one_to_one_possible':False,'source_count':128,'Q_count':3336,'selected_beta':None,'selected_beta_reason':'Archived pipeline calibrates lambda per method at fixed radius; beta grid is diagnostic, no single beta selected.','newest_raw_OLD_previously_evaluated':False,'newest_raw_OLD_policy':'unchanged archived transport and calibration-bank lambda rule, replayed with newest fixed Q and common 5927-model bank','future_models_used_for_assignment':False})
 dump(out/'resolved_config.json',c)

@torch.no_grad()
def bank_scores(bank,q,w,device):
 q=q.to(device);w=w.to(device);hinges=[]
 for start in range(0,bank.count,32):
  pred=bank.logits(q,start,start+32).sigmoid();hinges.append((torch.relu(.5-pred).double()*w).sum(1).cpu())
 return torch.cat(hinges).numpy()

def compare(a,c):
 out=a.output;raw,old=frozen(c);tag=a.beta;beta=tag/100;job=out/f'beta{tag}';job.mkdir(exist_ok=True);q=load(raw/f'raw_supports_U_beta{tag}.pt')['points'];target=load(raw/'training.pt')['target'];current=load_model(raw/'current.pt',22)
 records=read(raw/'old_bank.json')['candidates'];models=[load_model(Path(r['model_path']),22) for r in records];bank=ModelBank(models,a.device);del models
 # Before primary scoring, independently replay archived selected OLD bank risks on its original bank subset.
 archived=load(out/'archived_selected.pt')[str(tag)];archrisk=bank_scores(bank,archived['points'],archived['weights'],a.device)
 oldids={r['candidate_id'] for r in read(old/'evaluation_bank.json')['candidates']};mask=np.array([r['candidate_id'] in oldids for r in records]);assert int(mask.sum())==5387
 ref=pd.read_csv(old/'bank_test_curves.csv');rr=ref[(ref.method==archived['method'])&(ref['lambda']==archived['lambda'])&(ref.r_over_b_ref==3)].iloc[0];error=abs(float(archrisk[mask].max())-rr.bank_hinge);assert error<2e-6,error
 dump(job/'archived_bank_reproduction.json',{'bank_size':int(mask.sum()),'recomputed':float(archrisk[mask].max()),'saved':float(rr.bank_hinge),'absolute_error':error})
 cal=load(old/'cal_population.pt');calpath,_,_=pathof(cal['path']['base'],q,target,c['lambda_grid']);curves=[]
 for li,lam in enumerate(c['lambda_grid']):
  pts=calpath['q'][li];w=calpath['weights']
  if lam in [0,1]:pts=cal['path']['base'] if lam==0 else target;w=torch.full((len(pts),),1/len(pts),dtype=torch.float64)
  risks=bank_scores(bank,pts,w,a.device);curves.append(dict(method='OLD-native-replayed',r_over_b_ref=3,lambda_value=lam,bank_hinge=float(risks.max())))
  if li%5==0:print(tag,'calibration lambda',lam,flush=True)
 frame=pd.DataFrame(curves).rename(columns={'lambda_value':'lambda'});chosen=exact_lambda_choices(frame)[0];lam=chosen['lambda'];frame.to_csv(job/'calibration_curves.csv',index=False);dump(job/'selection.json',chosen)
 pop=load(old/'test_population.pt');base=pop['path']['base'];source=pop['source'];start=time.perf_counter();path,p0,p1=pathof(base,q,target,c['lambda_grid']);old_seconds=time.perf_counter()-start
 li=c['lambda_grid'].index(lam);variants={'OLD-native-replayed':{'points':path['q'][li],'weights':path['weights'],'source_index':path['source_index'],'lambda':lam,'runtime_seconds':old_seconds}}
 start=time.perf_counter();direct=direct_realization(base,q);direct_seconds=time.perf_counter()-start;assert not direct['one_per_factual'];assert abs(direct['plan']['distance']-p0.distance)<1e-10
 variants['NEW-DIRECT-weighted']={**direct,'lambda':None,'runtime_seconds':direct_seconds}
 # Evaluate unmodified direct outputs before any correction.
 before=validity(current,direct['points'],direct['weights']);dump(job/'direct_before_correction.json',before);torch.save(direct,job/'direct_before_correction.pt')
 if before['invalid_mass']>1e-12:
  start=time.perf_counter();fixed,corrections=minimal_restore(current,direct['points'],base[direct['source_index']],direct['weights']);repair_seconds=time.perf_counter()-start
  variants['NEW-DIRECT-VALID-weighted']={**direct,'points':fixed,'lambda':None,'runtime_seconds':direct_seconds+repair_seconds,'correction_seconds':repair_seconds}
  pd.DataFrame(corrections).to_csv(job/'corrections.csv',index=False)
 torch.save(variants,job/'outputs.pt');dump(job/'outputs_frozen.json',{'sha256':sha(job/'outputs.pt'),'before_future_evaluation':True,'Q_sha256':sha(raw/f'raw_supports_U_beta{tag}.pt')})
 train_base=load(raw/'train_population.pt')['path']['base'];D=exact_w2(train_base,target).distance;tr0=exact_w2(train_base,q).distance;tr1=p1.distance
 learned=bank_scores(bank,q,torch.full((len(q),),1/len(q),dtype=torch.float64),a.device)
 rawg=pd.read_csv(raw/'raw_geometry.csv');expected=float(rawg[rawg.beta_geo==beta].iloc[0].true_bank_hinge);assert abs(float(learned.max())-expected)<2e-6
 summaries=[];allrisk=[];atoms=[];checks=[]
 freshmask=np.array([r.get('parent_phase')=='fresh' for r in records])
 for name,v in variants.items():
  pts=v['points'];w=v['weights'];idx=v['source_index'];risk=bank_scores(bank,pts,w,a.device)
  distance=exact_w2(pts,q,a=w).distance
  if name=='NEW-DIRECT-weighted':assert distance<1e-7;assert abs(float(risk.max())-float(learned.max()))<2e-6
  cv=validity(current,pts,w);cost=(pts-source[idx]).norm(dim=1).double();disp=(pts-base[idx]).norm(dim=1).double()
  deployed0=exact_w2(base,pts,b=w).distance;deployed1=exact_w2(pts,target,a=w).distance
  row=dict(beta_geo=beta,method=name,representation='weighted atoms; conditional distribution per factual',seed=1,split='test',current_validity=cv['current_validity'],recourse_cost=float((cost*w).sum()),base_displacement=float((disp*w).sum()),train_bank_risk=float(risk.max()),learned_Q_bank_risk=float(learned.max()),inherited_fresh_subset_risk=float(risk[freshmask].max()) if freshmask.any() else None,W2_deploy_Q=distance,train_W2_Base_Q=tr0,W2_Q_P1=tr1,train_Q_total=tr0+tr1,budget=beta*D,test_W2_Base_Q=p0.distance,test_Q_total=p0.distance+p1.distance,test_Q_budget_feasible=p0.distance+p1.distance<=beta*D+1e-6*max(1,D),deployed_total=deployed0+deployed1,deployed_budget_feasible=deployed0+deployed1<=beta*D+1e-6*max(1,D),assignment_W2=direct['plan']['distance'],runtime_seconds=v['runtime_seconds'],lambda_value=v['lambda'],atoms=len(pts),factuals=len(source))
  summaries.append(row)
  for j,r in enumerate(records):allrisk.append(dict(method=name,candidate_id=r['candidate_id'],hinge=float(risk[j])))
  for j in range(len(pts)):atoms.append(dict(beta_geo=beta,method=name,source_index=int(idx[j]),factual_id=int(pop['source_ids'][idx[j]]),mass=float(w[j]),recourse_cost=float(cost[j]),base_displacement=float(disp[j])))
  assert len(idx.unique())==128
  checks.append({'method':name,'all_factuals_preserved':True,'current_invalid_mass':cv['invalid_mass'],'W2_deploy_Q':distance})
 pd.DataFrame(allrisk).to_csv(job/'bank_risks.csv',index=False);pd.DataFrame(atoms).to_csv(job/'per_atom_costs.csv',index=False)
 del bank
 meta=read(raw/'instance_manifest.json');future=pd.read_csv(Path(meta['original_instance'])/'future_final_manifest.csv');future_rows=[]
 for operator in c['future_operators']:
  for model_index,r in future.iterrows():
   mp=old/'aligned_future_models'/f'model{model_index}.pt' if operator=='aligned_full_update' else Path(r.path);m=load_model(mp,22)
   for name,v in variants.items():
    with torch.no_grad():prob=m(v['points']).sigmoid();w=v['weights'];fv=float(population_mean((prob>=.5).double(),w));hinge=float(population_mean(torch.relu(.5-prob).double(),w))
    future_rows.append(dict(beta_geo=beta,method=name,seed=1,operator=operator,model_index=model_index,environment=r.environment,severity_control=float(r.severity_control),realization=int(r.realization),future_seed=int(r.seed),future_validity=fv,future_hinge=hinge))
 f=pd.DataFrame(future_rows);f.to_csv(job/'future_per_model.csv',index=False)
 final=[]
 for row in summaries:
  for operator in c['future_operators']:
   values=f[(f.method==row['method'])&(f.operator==operator)];final.append({**row,'operator':operator,'future_validity_mean':float(values.future_validity.mean()),'future_validity_std':float(values.future_validity.std(ddof=1)),'future_validity_min':float(values.future_validity.min()),'heldout_future_bank_risk':float(values.future_hinge.max())})
 pd.DataFrame(final).to_csv(job/'summary.csv',index=False);dump(job/'checks.json',{'checks':checks,'archived_OLD_bank_reproduced':True,'learned_Q_bank_risk_reproduced':True,'one_to_one_possible':False,'optimal_transport_solver':direct['plan']['solver'],'OT_marginal_residual':direct['plan']['marginal_residual'],'Q_unchanged':sha(raw/f'raw_supports_U_beta{tag}.pt')==read(job/'outputs_frozen.json')['Q_sha256'],'no_P1_in_direct_or_repair':True,'no_future_feedback':True})
 print(pd.DataFrame(final)[['beta_geo','method','operator','current_validity','future_validity_mean','recourse_cost','W2_deploy_Q']].to_string(index=False),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','compare']);p.add_argument('--output',type=Path,required=True);p.add_argument('--beta',type=int);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.14)
 c=read(ROOT/'experiments/configs/heloc_realization_comparison.json');globals()[a.phase](a,c)
if __name__=='__main__':main()
