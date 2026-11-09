"""Higher geometry budgets on frozen GENKI-head GRACE inputs and banks."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from . import genki_grace_identity_v1 as old
from .galaxy_grace_v1 import load,dump,sha,stamp
from ..counterfactuals.geodesic_excess import exact_w2
ROOT=Path(__file__).resolve().parents[4]


def key(q):return hashlib.sha256(q.numpy().tobytes()).hexdigest()


def initialize(a,c):
    out=a.output;parent=Path(c['parent_run'])
    for name in ['jobs','logs','qualitative_examples']:(out/name).mkdir(exist_ok=True)
    for name in ['checkpoints','test_factuals.pt','test_population.pt','references.pt']:(out/name).symlink_to(parent/name)
    for p in sorted((parent/'jobs').glob('attack_*')):(out/'jobs'/p.name).symlink_to(p,target_is_directory=True)
    (out/'jobs/clean').symlink_to(parent/'jobs/clean',target_is_directory=True)
    inputs=[parent/'resolved_config.json',parent/'frozen.pt',parent/'selected_configuration.json',parent/'references.pt',parent/'test_factuals.pt',parent/'test_population.pt',parent/'attack_bank_manifest.csv']
    for p in (parent/'checkpoints').glob('*.pt'):inputs.append(p)
    for p in (parent/'jobs').glob('opt_test_1_*/candidates.pt'):inputs.append(p);inputs.append(p.parent/'candidates.csv')
    inputs.append(parent/'jobs/pool_test_1/selected.pt')
    for p in (parent/'qualitative_examples').glob('test_*.png'):inputs.append(p)
    dump(out/'input_manifest.json',{str(p):sha(p) for p in inputs});dump(out/'resolved_config.json',c)
    dump(out/'protocol_frozen.json',{'utc':stamp(),'config_sha256':sha(out/'resolved_config.json'),'fixed_r':2.,'new_betas':[1.5,2.,3.],
                                  'parent_outcomes_previously_seen':True,'future_models_reused':True,'optimizer_settings_changed':False})


def merge(a,c):
    out=a.output;parent=Path(c['parent_run']);z=load(out/'test_population.pt')['u'];target=load(out/'references.pt')['u'];facts=load(out/'test_factuals.pt')
    D=exact_w2(z,target).distance;states=[load(Path(r['checkpoint'])) for r in old.records(out,2.)]
    incumbent=load(parent/'jobs/pool_test_1/selected.pt')['1.2'];incumbent_risk=float(old.population_risk(old.state_scores(incumbent['features'],states).sigmoid()))
    allitems=[(incumbent,'previous_beta1.2')]
    for root,pattern in [(parent,'opt_test_1_*'),(out,'opt_test_1_*')]:
        for job in sorted((root/'jobs').glob(pattern)):
            allitems.extend((v,str(job)) for v in load(job/'candidates.pt'))
    items=[];rows=[];seen=set()
    for v,origin in allitems:
        digest=key(v['points'])
        if digest in seen:continue
        seen.add(digest);risk=float(old.population_risk(old.state_scores(v['features'],states).sigmoid()))
        length=exact_w2(z,v['points']).distance+exact_w2(v['points'],target).distance
        box=bool(((v['points']-facts['u']).abs()<=16+1e-6).all())
        # The previous winner is feasible for all new budgets; worse-risk points
        # cannot win. Keep equal-risk points for pixel-cost tie-breaking.
        if not box or length>3*D+1e-6*max(1,D) or risk>incumbent_risk+1e-12:continue
        rows.append(dict(candidate=len(items),origin=origin,length=length,risk=risk,pixel_cost=float((v['x']-facts['x']).flatten(1).norm(dim=1).mean()),coordinate_sha256=digest))
        items.append(v)
    assert rows and rows[0]['origin']=='previous_beta1.2'
    torch.save(items,out/'candidates.pt');pd.DataFrame(rows).to_csv(out/'candidate_table.csv',index=False)
    dump(out/'merge.json',{'D':D,'unique_evaluated_candidates':len(seen),'screened_candidates':len(items),'previous_incumbent_risk':incumbent_risk,'previous_incumbent_candidate':0})


def quality(a,c):
    i=int(a.job.split('_')[-1]);out=a.output;facts=load(out/'test_factuals.pt');m=old.GenkiMeasurement(a,c,facts['w'][i:i+1].to(a.device))
    idx=load(out/'checkpoints/controls.pt')['style4']['indices'].to(a.device);std=load(out/'checkpoints/statistics.pt')['style_std'].to(a.device)
    candidates=load(out/'candidates.pt');rows=[];emb={'arc_factual':m.arc0,'facenet_factual':m.face0}
    for j,v in enumerate(candidates):
        row,rgb,arc,face=m.measure((v['points'][i:i+1]-facts['u'][i:i+1]).to(a.device),idx,std)
        rows.append(dict(candidate=j,factual_index=i,quality_pass=old.passes_quality(row,c),**row))
        if arc is not None:emb[f'arc_{j}']=arc
        if face is not None:emb[f'facenet_{j}']=face
        Image.fromarray(rgb).save(out/'qualitative_examples'/f'candidate_{j}_{i}.png')
        if j%10==0:print(a.job,j,'/',len(candidates),flush=True)
    pd.DataFrame(rows).to_csv(out/'jobs'/a.job/'quality.csv',index=False);np.savez(out/'jobs'/a.job/'embeddings.npz',**emb)


def select(a,c):
    out=a.output;table=pd.read_csv(out/'candidate_table.csv');quality=pd.concat([pd.read_csv(out/'jobs'/f'quality_{i}'/'quality.csv') for i in range(8)],ignore_index=True)
    ok=quality.groupby('candidate').quality_pass.all();table['preservation_feasible']=table.candidate.map(ok)
    D=json.loads((out/'merge.json').read_text())['D'];selected={};rows=[];items=load(out/'candidates.pt')
    for beta in c['beta_grid']:
        eligible=table[table.preservation_feasible & (table.length<=beta*D+1e-6*max(1,D))]
        assert len(eligible)>0;best=eligible.sort_values(['risk','pixel_cost','candidate']).iloc[0];j=int(best.candidate)
        selected[str(beta)]=items[j];rows.append(dict(beta_geo=beta,candidate=j,D=D,B_geo=beta*D,total_length=float(best.length),slack=beta*D-float(best.length),
              Phi_hat_r_cached=float(best.risk),pixel_cost=float(best.pixel_cost),coordinate_sha256=best.coordinate_sha256,
              same_coordinates_as_previous_beta1_2=bool(key(items[j]['points'])==key(items[0]['points']))))
    torch.save(selected,out/'selected.pt');pd.DataFrame(rows).to_csv(out/'geometry.csv',index=False)
    quality.to_csv(out/'quality_screen.csv',index=False);table.to_csv(out/'screened_candidates.csv',index=False)
    dump(out/'selection_frozen.json',{'utc':stamp(),'selected_sha256':sha(out/'selected.pt'),'future_outcomes_used_for_new_selection':False,'no_best_beta_selected':True})


def freeze(a,c):
    out=a.output;parent=Path(c['parent_run']);geometry=pd.read_csv(out/'geometry.csv');quality=pd.read_csv(out/'quality_screen.csv');rows=[]
    previous=pd.read_csv(parent/'image_quality.csv')
    for i in range(8):
        for name,oldname in [('Factual','factual'),('Base','Base'),('Previous_beta1.20','GRACE_beta1.20')]:
            shutil.copyfile(parent/'qualitative_examples'/f'test_{i}_{oldname}.png',out/'qualitative_examples'/f'test_{i}_{name}.png')
            if name!='Factual':
                r=previous[(previous.method==oldname)&(previous.factual_index==i)].iloc[0].to_dict();r['method']=name;rows.append(r)
        for beta in [1.5,2.,3.]:
            j=int(geometry[geometry.beta_geo==beta].iloc[0].candidate);name=f'GRACE_beta{beta:.2f}'
            shutil.copyfile(out/'qualitative_examples'/f'candidate_{j}_{i}.png',out/'qualitative_examples'/f'test_{i}_{name}.png')
            r=quality[(quality.candidate==j)&(quality.factual_index==i)].iloc[0].to_dict();r['method']=name;rows.append(r)
    pd.DataFrame(rows).to_csv(out/'image_quality.csv',index=False)
    dump(out/'outputs_frozen.json',{'utc':stamp(),'image_hashes':{str(p):sha(p) for p in (out/'qualitative_examples').glob('test_*.png')},
                                  'selection_sha256':sha(out/'selection_frozen.json'),'evaluation_scope':'exploratory reuse of previously seen future models'})


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','optimize','merge','quality','select','freeze']);p.add_argument('--job',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20260918);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    cfg=a.output/'resolved_config.json';c=json.loads((cfg if cfg.exists() else ROOT/'experiments/configs/genki_grace_beta_extension.json').read_text())
    torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction']);job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
    if (job/'complete.json').exists():return
    start=time.time();meta={'phase':a.phase,'job':a.job,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta)
    if a.phase=='optimize':old.optimize(a,c)
    else:globals()[a.phase](a,c)
    torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start})


if __name__=='__main__':main()
