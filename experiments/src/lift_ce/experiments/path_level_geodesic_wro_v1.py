"""Isolated path-level constrained WRO experiment; never overwrite parent runs."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import pandas as pd
import torch
import yaml

from . import fixed_risk_geometry_v1 as e
from ..counterfactuals.path_level_solver import (
    path_state, frozen_state, audit_path, straight_derived_initial, solve_path)
from ..counterfactuals.robust_waypoints import ModelBank

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/'configs/path_level_geodesic_wro_v1.yaml'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def verify_files(files):
    for name, expected in files.items():
        if e.p.file_hash(Path(name)) != expected: raise ValueError(f'frozen file changed: {name}')


def verify_bank_once(records, meta, train, budget):
    """Check each unique replay/delta artifact once, preserving every candidate."""
    groups = defaultdict(list)
    checked, hashes = [], {}
    for record in records:
        if record['split'] != 'train' or record['full_update_seed'] != meta['full_update_seed']:
            raise ValueError('incompatible candidate provenance')
        path = Path(record['model_path'])
        digest = hashes.setdefault(str(path), e.p.file_hash(path))
        if digest != record['model_sha256']: raise ValueError('model hash changed')
        groups[record['delta_file']].append(record)
    for name, members in groups.items():
        path = Path(name); digest = e.p.file_hash(path); hashes[name] = digest
        saved = torch.load(path, weights_only=False)
        if saved.get('discovery_split') != 'train' or saved.get('full_update_seed') != meta['full_update_seed']:
            raise ValueError('delta provenance mismatch')
        for record in members:
            if record.get('delta_file_sha256', digest) != digest: raise ValueError('delta hash mismatch')
            value = saved['delta'] if record.get('delta_key') == 'delta' else saved['candidates'][record['delta_index']]
            delta = value['delta'] if isinstance(value, dict) else value
            measured = float(e.rms(delta))
            if delta.shape != train['x'].shape or not torch.isfinite(delta).all(): raise ValueError('invalid displacement')
            if measured > min(budget, record['nominal_radius']*meta['b_ref'])+1e-5:
                raise ValueError('candidate outside data budget')
            checked.append({**record, 'phase': 'old', 'parent_phase': record['phase'],
                            'measured_budget': measured, 'delta_file_sha256': digest})
    if len({a['candidate_id'] for a in checked}) != len(checked): raise ValueError('duplicate bank IDs')
    return checked, hashes


def prepare(args):
    out = args.output/'heloc'; out.mkdir(parents=True, exist_ok=True)
    if (out/'construction_manifest.json').exists(): raise ValueError('already prepared')
    config = yaml.safe_load(CONFIG.read_text())
    parent = Path(config['parent'])
    meta = json.loads((parent/'instance_manifest.json').read_text())
    verify_files({str(parent/name): h for name, h in meta['files'].items()})
    for name in ['instance_manifest.json','training.pt','current.pt','train_population.pt',
                 'cal_population.pt','test_population.pt','frozen_waypoint.pt','controls_manifest.json']:
        shutil.copyfile(parent/name, out/name)
    _, _, train, _ = e.context(args)
    records, hashes = verify_bank_once(json.loads(Path(config['construction_bank']).read_text())['candidates'],
                                      meta, train, config['absolute_data_radius'])
    config.update(status='implemented_experiment', construction_bank_size=len(records),
                  optimizer={**config['optimizer'], 'model_chunk':64,
                             'feasibility_relative_tolerance':config['geometry_relative_tolerance']},
                  search=e.SETTINGS['search'], source_lambdas=e.SETTINGS['source_lambdas'],
                  domain_note='training box applies to controls only; path range remains diagnostic',
                  current_constraint_residual='half(mean_grid_weighted_hinge_f0 + mean_control_hinge_f0); zero required',
                  confirmation='previously inspected benchmark; exploratory only')
    write_json(out/'resolved_config.json',config)
    write_json(out/'old_bank.json', {'instance_id':meta['instance_id'],'candidates':records})
    pd.DataFrame(records).to_csv(out/'attack_bank_manifest.csv',index=False)
    pd.DataFrame([{'beta_geo':b,'D_train':config['D_train'],'B_geo':b*config['D_train']} for b in config['beta_geo']]).to_csv(out/'geometry_budget_grid.csv',index=False)
    deps = {str(parent/'frozen_waypoint.pt'):e.p.file_hash(parent/'frozen_waypoint.pt'),
            str(parent/'controls_manifest.json'):e.p.file_hash(parent/'controls_manifest.json'),
            str(Path(config['construction_bank'])):e.p.file_hash(Path(config['construction_bank']))}
    for item in json.loads((out/'controls_manifest.json').read_text()):
        file=parent/item['control_file']; deps[str(file)]=item['control_sha256']
    verify_files(deps)
    fixed = {str(out/name):e.p.file_hash(out/name) for name in ['resolved_config.json','old_bank.json','instance_manifest.json','training.pt','current.pt','train_population.pt','cal_population.pt','test_population.pt']}
    write_json(out/'construction_manifest.json', {'instance_id':meta['instance_id'],'files':fixed,
        'external_dependencies':deps,'candidate_artifact_hashes':hashes,'training_only':True})
    print('Prepared:', len(records), 'verified full-update models',flush=True)


def inputs(args, models=True, small=False):
    out, meta, train, current = e.context(args)
    manifest=json.loads((out/'construction_manifest.json').read_text())
    verify_files(manifest['files']);verify_files(manifest['external_dependencies'])
    config=json.loads((out/'resolved_config.json').read_text())
    records=json.loads((out/'old_bank.json').read_text())['candidates']
    current=current.to(args.device).eval().requires_grad_(False)
    bank=None
    if models:
        chosen=records[:8] if small else records
        states=[]
        for row in chosen:
            file=Path(row['model_path'])
            if e.p.file_hash(file)!=row['model_sha256']:raise ValueError('model changed')
            states.append(e.legacy.load_model(file,train['x'].shape[1]))
        bank=ModelBank(states,args.device)
    return out,config,train,current,bank


def initial(out,config,source,target,beta,kind):
    k=config['intermediate_distributions']
    if kind=='straight_derived':return straight_derived_initial(source,target,k)
    if kind=='random':
        g=torch.Generator().manual_seed(config['optimizer']['random_seed'])
        return (target.cpu()[None].repeat(k,1,1)+config['optimizer']['random_sd']*torch.randn((k,*target.shape),generator=g)).to(target)
    if kind=='old_waypoint':
        point=torch.load(out/'frozen_waypoint.pt',weights_only=True)['waypoints']
    elif kind=='previous_control':
        item=next(a for a in json.loads((out/'controls_manifest.json').read_text()) if a['beta_geo']==beta)
        point=torch.load(Path(config['parent'])/item['control_file'],weights_only=True)['points']
    else:raise ValueError(kind)
    return point.to(target)[None].repeat(k,1,1)


def write_trace_callback(file, job):
    file.parent.mkdir(parents=True,exist_ok=True)
    def callback(row):
        data={**job,**row}
        with file.open('a') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data))
            if handle.tell()==0:writer.writeheader()
            writer.writerow(data)
        if row['step']%10==0:
            print(job,'step',row['step'],'risk',row['true_path_risk'],'slack',row['slack'],
                  'invalid',row['maximum_grid_invalid_mass'],'seconds',row['seconds'],flush=True)
    return callback


def smoke(args):
    out,config,train,current,bank=inputs(args,small=args.phase=='smoke')
    population=torch.load(out/'train_population.pt',weights_only=True)
    source=population['path']['base'].to(args.device);target=train['target'].to(args.device)
    if args.phase=='smoke':source=source[:8];target=target[:16]
    controls=straight_derived_initial(source,target,2)
    lower,upper=train['x'].amin(0).to(args.device),train['x'].amax(0).to(args.device)
    steps=2 if args.phase=='smoke' else 1
    started=time.perf_counter()
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    result=solve_path(bank,current,source,target,controls,1.1*config['D_train'],config['D_train'],lower,upper,
                      config['lambdas'],{**config['optimizer'],'steps':steps},
                      on_step=write_trace_callback(out/f'{args.phase}_trace.csv',{'phase':args.phase}))
    seconds=time.perf_counter()-started
    write_json(out/f'{args.phase}.json',{'seconds':seconds,'status':result['status'],
        'models':bank.count,'source_count':len(source),'target_count':len(target),'trace':result['trace'],
        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
        'excluded_from_primary_pool':True,
        'estimated_primary_gpu_hours':sum(a['seconds'] for a in result['trace'][:steps])/max(1,steps)*121*16/3600})
    print(args.phase,seconds,'seconds',result['status'],flush=True)


def optimize(args):
    out,config,train,current,bank=inputs(args)
    source=torch.load(out/f'{args.split}_population.pt',weights_only=True)['path']['base'].to(args.device)
    target=train['target'].to(args.device)
    lower,upper=train['x'].amin(0).to(args.device),train['x'].amax(0).to(args.device)
    jobs=[(b,kind) for b in config['beta_geo'] for kind in config['initializations']]
    for job,(beta,kind) in enumerate(jobs):
        if job%args.workers!=args.worker:continue
        if args.split!='train':
            frozen=verify_paths(out)
            if f'P_beta{round(beta*100):03d}' not in frozen['feasible_training_methods']:continue
        file=out/'optimization'/args.split/f'beta{round(beta*100):03d}_{kind}.pt'
        if file.exists():continue
        tracefile=file.with_suffix('.csv')
        if tracefile.exists():raise ValueError(f'incomplete job needs explicit recovery: {tracefile}')
        w=initial(out,config,source,target,beta,kind)
        if args.split!='train' and kind=='previous_control':
            w=torch.load(out/'controls'/f'P_beta{round(beta*100):03d}.pt',weights_only=True)['controls'].to(target)
        result=solve_path(bank,current,source,target,w,beta*config['D_train'],config['D_train'],lower,upper,
            config['lambdas'],config['optimizer'],on_step=write_trace_callback(tracefile,{'split':args.split,'beta_geo':beta,'initialization':kind}))
        result.update(split=args.split,beta_geo=beta,initialization=kind)
        e.save(result,file)
        print('Completed',args.split,beta,kind,result['status'],flush=True)


def verify_paths(out):
    manifest=json.loads((out/'frozen_paths.json').read_text())
    verify_files(manifest['files'])
    return manifest


def collect(args):
    out,config,train,current,bank=inputs(args,models=False)
    current=current.cpu()
    pop=torch.load(out/f'{args.split}_population.pt',weights_only=True)
    source,target=pop['path']['base'],train['target']
    lower,upper=train['x'].amin(0),train['x'].amax(0)
    files=list((out/'optimization'/args.split).glob('*.pt'))
    expected=16 if args.split=='train' else 4*len(verify_paths(out)['feasible_training_methods'])
    if len(files)!=expected:raise ValueError(f'incomplete {args.split} optimization: {len(files)}/{expected}')
    jobs=[torch.load(f,weights_only=True) for f in sorted(files)]
    pool=[(j,j['candidate']) for j in jobs if j['candidate'] is not None]
    # Historic controls are preserved verbatim, with their known infeasibility.
    parent_paths=torch.load(Path(config['parent'])/f'paths_{args.split}.pt',weights_only=True)
    methods={name:value for name,value in parent_paths['methods'].items()}
    statuses=[];frozen_files={};prior=float('inf')
    for beta in config['beta_geo']:
        method=f'P_beta{round(beta*100):03d}'
        candidates=[(j,c) for j,c in pool if c['audit']['total_length']<=beta*config['D_train']+config['geometry_relative_tolerance']*max(1,config['D_train'])]
        status={'split':args.split,'method':method,'beta_geo':beta,'B_geo':beta*config['D_train'],'feasible_candidates':len(candidates)}
        if not candidates:
            statuses.append({**status,'status':'NO_FEASIBLE_ITERATE'});continue
        job,candidate=min(candidates,key=lambda x:(x[1]['true_path_risk'],x[0]['initialization'],x[1]['step']))
        state=path_state(source,candidate['controls'],target,config['lambdas'])
        audit=audit_path(state,candidate['controls'],current,lower,upper,beta*config['D_train'],config['geometry_relative_tolerance']*max(1,config['D_train']))
        if not audit['feasible']:raise ValueError('independent final feasibility failed')
        if candidate['true_path_risk']>prior+1e-12:raise ValueError('nested incumbent risk worsened')
        prior=candidate['true_path_risk']
        methods[method]=frozen_state(state)
        file=out/'controls'/f'{method}{"" if args.split=="train" else "_"+args.split}.pt'
        e.save({'controls':candidate['controls'],'audit':audit,'path':methods[method]},file)
        frozen_files[str(file)]=e.p.file_hash(file)
        statuses.append({**status,**audit,'status':'FEASIBLE','true_path_risk':candidate['true_path_risk']})
    e.save({'source':pop['source'],'source_ids':pop['source_ids'],'methods':methods},out/f'paths_{args.split}.pt')
    frozen_files[str(out/f'paths_{args.split}.pt')]=e.p.file_hash(out/f'paths_{args.split}.pt')
    pd.DataFrame(statuses).to_csv(out/f'{args.split}_path_feasibility.csv',index=False)
    traces=[pd.read_csv(f.with_suffix('.csv')) for f in sorted(files)]
    if traces:
        pd.concat(traces).to_csv(out/f'optimization_trace_{args.split}.csv',index=False)
    if args.split=='train':
        write_json(out/'frozen_paths.json',{'files':frozen_files,'feasible_training_methods':[s['method'] for s in statuses if s['status']=='FEASIBLE'],
                   'construction_manifest_sha256':e.p.file_hash(out/'construction_manifest.json'),'frozen_before_discovery':True})
    else:write_json(out/f'frozen_paths_{args.split}.json',{'files':frozen_files,'frozen_training_paths_sha256':e.p.file_hash(out/'frozen_paths.json')})
    print(pd.DataFrame(statuses).to_string(index=False),flush=True)


def require_query(args,split):
    out=args.output/args.dataset
    if not (out/f'frozen_paths_{split}.json').exists():raise ValueError('run and freeze held-out batch refinement first')
    verify_files(json.loads((out/f'frozen_paths_{split}.json').read_text())['files'])


def report(args):
    out=args.output/args.dataset
    frozen=verify_paths(out)
    rows=pd.read_csv(out/'train_path_feasibility.csv')
    if not frozen['feasible_training_methods']:
        missing=['held-out refinement: no feasible trained path to transfer',
                 'fresh discovery: feasibility gate failed for every new beta path',
                 'expanded-bank comparison and future metrics: no admissible new path',
                 'optional K=3/4: deferred until primary feasibility passes']
        result={'status':'COMPLETED_CONSTRUCTION_REJECTION','training_solver_jobs':16,
                'feasible_new_paths':0,'deployment_feasible':False,'missing_or_gated_cells':missing,
                'robustness_improvement_established':False,'unrestricted_geometric_infeasibility_proved':False}
        write_json(out/'completion.json',result)
        (out/'report.md').write_text('# Path-level construction rejected\n\n'+rows.to_string(index=False)+'\n\n'+
            '\n'.join('- '+s for s in missing)+'\n\nNo feasible iterate was found by the declared local optimizer. This is not proof that a feasible path does not exist. No robustness or matched-cost improvement is established.\n')
    else:
        write_json(out/'completion.json',{'status':'CONSTRUCTION_COMPLETE_DOWNSTREAM_REQUIRED','feasible_new_paths':len(frozen['feasible_training_methods'])})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','smoke','benchmark','optimize','collect','discover','freeze_eval_bank','select','future_models','evaluate','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',default='heloc',choices=['heloc'])
    parser.add_argument('--split',default='train',choices=['train','cal','test'])
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    e.SETTINGS={**e.SETTINGS,'evaluation_radii':[3.]}
    e.verify_geometry=verify_paths;e.query_paths=require_query
    if args.phase in ['prepare','optimize','collect','report']:globals()[args.phase](args)
    elif args.phase in ['smoke','benchmark']:smoke(args)
    else:
        if not verify_paths(args.output/'heloc')['feasible_training_methods']:
            raise ValueError('no feasible new path: post-construction evaluation gated')
        getattr(e,args.phase)(args)


if __name__=='__main__':main()
