"""Task 004: continuous geometry-only WRO, then invalid-only restoration."""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import fixed_risk_geometry_v1 as e
from . import single_budget_geodesic_wro_v1 as previous
from . import transport_spline_path_v1 as merger
from .favorable_region_geodesic_wro_v1 import read,verify_bank,verify_files
from ..counterfactuals.minimal_validity_repair import solve_raw,minimal_restore,restore_with_budget,validity,geometry
from ..counterfactuals.geodesic_excess import exact_w2,glued_arclength_path
from ..counterfactuals.robust_waypoints import ModelBank
from ..counterfactuals.single_budget_solver import bank_value_gradient

PARENT=Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc')
GRAPH=Path('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1/heloc')
BETAS=[1.,1.05,1.10,1.20]
LAMBDAS=[round(i*.05,2) for i in range(21)]
OPT={**previous.OPT}
STARTS=['target','old_waypoint','random']
RAW=[f'U_beta{round(b*100):03d}' for b in BETAS]
REPAIRED=[f'P_beta{round(b*100):03d}' for b in BETAS]
CONTROLS=previous.METHODS+[f'F_beta{round(b*100):03d}' for b in BETAS]


def prepare(args):
    out=args.output/'heloc';out.mkdir(parents=True,exist_ok=True)
    for name in ['instance_manifest.json','training.pt','current.pt','train_population.pt','data_audit.pt','frozen_waypoint.pt','waypoint_manifest.json']:
        shutil.copyfile(PARENT/name,out/name)
    _,meta,train,current=e.context(args)
    verify_files(GRAPH,'construction_manifest.json');verify_files(GRAPH,'frozen_paths.json')
    historical=read(PARENT/'frozen_paths.json')
    for name,digest in historical['files'].items():
        if e.p.file_hash(PARENT/name)!=digest:raise ValueError('historical continuous control changed')
    records=verify_bank(read(GRAPH/'evaluation_bank.json')['candidates'],meta,train)
    records=[{**a,'parent_phase':a['phase'],'phase':'old'} for a in records]
    if len(records)!=5927:raise ValueError('incomplete current clean bank')
    e.p.write_json(out/'old_bank.json',{'candidates':records,'parent':str(GRAPH/'evaluation_bank.json'),'parent_sha256':e.p.file_hash(GRAPH/'evaluation_bank.json')})
    pd.DataFrame(records).to_csv(out/'attack_bank_manifest.csv',index=False)
    source=torch.load(out/'train_population.pt',weights_only=True)['path']['base'];D=exact_w2(source,train['target']).distance
    config={'task':'tasks/004_baseline.md','dataset':'heloc','D':D,'beta_geo':BETAS,'tau':.5,'radius_over_b_ref':3.,
            'absolute_data_budget':3*meta['b_ref'],'lambdas':LAMBDAS,'optimizer':OPT,'initializations':STARTS,
            'geometry':'same unregularized Euclidean W2 as historical continuous single-budget solver; no Sinkhorn',
            'raw_objective':'max_model population_mean_hinge at Q; smooth hinge50 and outer max100 only for gradients',
            'raw_constraints':'geometry AL and observed continuous feature box only; f0 not passed into raw optimizer',
            'historical_feasible_incumbents':'include frozen old continuous controls, rescored on the same5927-model bank; no future selection',
            'repair':'first favorable interval along each corresponding target segment; float64 ReLU pieces then reference-float32 bisection; valid supports bitwise unchanged',
            'repair_alpha_tolerance':1e-7,'geometry_tolerance':OPT['feasibility_relative_tolerance']*max(1.,D),
            'joint_repair':'if first repair exceeds budget, search common progress of invalid particles only with32 ordered proposals and20 bracket refinements; report numerical failure, no global infeasibility proof',
            'feature_semantics':'frozen HELOC all-mutable continuous encoding, no categorical columns; no physical integer/actionability guarantee',
            'selection':'fixed r3; expanded training-bank exact argmin over current-valid final grid distributions; smallest lambda on exact ties; no beta selection',
            'query':'same frozen Q; separate batch OT; no heldout optimization; verify selected output validity and fixed absolute budget; unsupported transfer marked undeployable',
            'secondary_diagnostic':'repair only final outputs of historical S_beta at predeclared frozen-old-bank training argmin; fixed deterministic f0-only rule; preserve split masses and target pairing; no targeted fresh search for this secondary diagnostic; not a new path or optimizer',
            'fresh_source_lambdas':[.25,.5,.75],'replications':3,'fresh_labels':['raw','repaired'],
            'search':{('softplus_sharpness' if k=='kappa' else k):v for k,v in e.SETTINGS['search'].items()},
            'future':'reuse45 verified aligned-Update models; scratch45 separately labeled; retrospective not untouched confirmation',
            'construction_bank_count':len(records),
            'secondary_training_lambdas':{m:float(group.loc[group.bank_hinge.idxmin(),'lambda']) for m,group in pd.read_csv(GRAPH/'bank_train_curves.csv').groupby('method') if m.startswith('S_beta')}}
    e.p.write_json(out/'resolved_config.json',config)
    files=['instance_manifest.json','training.pt','current.pt','train_population.pt','old_bank.json','resolved_config.json']
    e.p.write_json(out/'construction_manifest.json',{'files':{n:e.p.file_hash(out/n) for n in files}})
    print('prepared continuous D',D,'bank',len(records),flush=True)


def load(args):
    out,meta,train,current=e.context(args);verify_files(out,'construction_manifest.json')
    records=read(out/'old_bank.json')['candidates']
    bank=ModelBank([e.legacy.load_model(Path(a['model_path']),22) for a in records],args.device)
    source=torch.load(out/'train_population.pt',weights_only=True)['path']['base'].to(args.device)
    return out,meta,train,current,bank,source,train['target'].to(args.device)


def optimize(args):
    out,meta,train,current,bank,source,target=load(args)
    config=read(out/'resolved_config.json');lower=train['x'].min(0).values.to(args.device);upper=train['x'].max(0).values.to(args.device)
    for job,(beta,kind) in enumerate((b,k) for b in BETAS for k in STARTS):
        if job%args.workers!=args.worker:continue
        file=out/'optimization'/f'beta{round(beta*100):03d}_{kind}.pt'
        if file.exists():continue
        initial=previous.initial_points(out,train,kind,args.device)
        def progress(row):
            if row['step']%10==0:print(beta,kind,'step',row['step'],'risk',row['true_bank_hinge'],'slack',row['slack'],flush=True)
        started=time.perf_counter()
        candidates,trace=solve_raw(bank,source,target,initial,config['D'],beta*config['D'],lower,upper,OPT,progress)
        e.save({'candidates':candidates,'beta_geo':beta,'initialization':kind,'construction':verify_files(out,'construction_manifest.json')},file)
        pd.DataFrame([{**r,'beta_geo':beta,'initialization':kind} for r in trace]).to_csv(file.with_suffix('.csv'),index=False)
        e.p.write_json(file.with_suffix('.runtime.json'),{'seconds':time.perf_counter()-started})


def smoke(args):
    out,meta,train,current,bank,source,target=load(args);config=read(out/'resolved_config.json')
    start=time.perf_counter()
    candidates,trace=solve_raw(bank,source,target,target,config['D'],1.1*config['D'],train['x'].min(0).values.to(args.device),train['x'].max(0).values.to(args.device),{**OPT,'steps':2})
    seconds=time.perf_counter()-start
    q=candidates[-1]['points'];fixed,rows,info=restore_with_budget(current,source.cpu(),q,target.cpu(),config['D'],1.1*config['D'],tolerance=config['geometry_tolerance'])
    if fixed is None:raise ValueError('smoke restoration failed')
    e.p.write_json(out/'smoke.json',{'seconds':seconds,'full_bank':bank.count,'full_particles':len(target),'steps':2,
                   'estimated_gpu_hours_12_jobs':seconds/3*121*12/3600,'raw_feasible_candidates':len(candidates),
                   'repair_invalid_count':sum(r['raw_invalid'] for r in rows),'repair_feasible':True,'excluded_from_primary_pool':True})
    print('full-support smoke',seconds,'seconds',flush=True)


def freeze_raw(args):
    out,meta,train,current,bank,source,target=load(args);config=read(out/'resolved_config.json');pool=[];traces=[]
    for beta in BETAS:
        for kind in STARTS:
            file=out/'optimization'/f'beta{round(beta*100):03d}_{kind}.pt';obj=torch.load(file,weights_only=True)
            if obj['construction']!=verify_files(out,'construction_manifest.json'):raise ValueError('raw construction mismatch')
            pool.extend([{**r,'origin_beta':beta,'initialization':kind} for r in obj['candidates']]);traces.append(pd.read_csv(file.with_suffix('.csv')))
    # Historical strong supports are preserved as explicit training-only feasible incumbents.
    for beta in BETAS:
        q=torch.load(PARENT/f'learned_control_distribution_beta{round(beta*100):03d}.pt',weights_only=True)['points'].to(args.device)
        _,risk,_=bank_value_gradient(bank,q,gradient=False)
        _,_,g=geometry(source,q,target,config['D'],beta*config['D'])
        pool.append({'points':q.cpu(),'true_bank_hinge':risk,'origin_beta':beta,'initialization':'historical_continuous_incumbent',**g})
    rows=[];v=[]
    for beta,method in zip(BETAS,RAW):
        eligible=[r for r in pool if r['total_length']<=beta*config['D']+config['geometry_tolerance']]
        chosen=min(eligible,key=lambda r:r['true_bank_hinge']);q=chosen['points']
        _,_,g=geometry(source.cpu(),q,target.cpu(),config['D'],beta*config['D'])
        if g['slack'] < -config['geometry_tolerance']:raise ValueError('raw exact geometry failed')
        e.save({'points':q,'weights':torch.full((len(q),),1/len(q),dtype=torch.float64)},out/f'raw_supports_{method}.pt')
        rows.append({'method':method,'beta_geo':beta,'true_bank_hinge':chosen['true_bank_hinge'],
                     'initialization':chosen['initialization'],'origin_beta':chosen['origin_beta'],**g})
        v.append({'method':method,'beta_geo':beta,**validity(current,q)})
    pd.concat(traces).to_csv(out/'raw_optimization_trace.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'raw_geometry.csv',index=False);pd.DataFrame(v).to_csv(out/'raw_validity.csv',index=False)
    files=[f'raw_supports_{m}.pt' for m in RAW]+['raw_geometry.csv']
    e.p.write_json(out/'frozen_raw.json',{'files':{n:e.p.file_hash(out/n) for n in files},'before_any_repair':True})


def repair(args):
    out,meta,train,current=e.context(args);verify_files(out,'frozen_raw.json');config=read(out/'resolved_config.json')
    source=torch.load(out/'train_population.pt',weights_only=True)['path']['base'];rows=[];alpha=[];v=[];manifest=[]
    for beta,raw,method in zip(BETAS,RAW,REPAIRED):
        q=torch.load(out/f'raw_supports_{raw}.pt',weights_only=True)['points']
        repaired,entries,info=restore_with_budget(current,source,q,train['target'],config['D'],beta*config['D'],tolerance=config['geometry_tolerance'])
        e.p.write_json(out/f'repair_{method}.json',info)
        alpha.extend([{**r,'method':method,'beta_geo':beta} for r in entries])
        if repaired is None:
            manifest.append({'method':method,'beta_geo':beta,'status':'infeasible_under_declared_repair_search'});continue
        _,_,g=geometry(source,repaired,train['target'],config['D'],beta*config['D'])
        if g['slack'] < -config['geometry_tolerance']:raise ValueError('over-budget repair retained')
        torch.testing.assert_close(repaired[torch.tensor([not r['raw_invalid'] for r in entries])],q[torch.tensor([not r['raw_invalid'] for r in entries])],atol=0,rtol=0)
        e.save({'points':repaired,'weights':torch.full((len(repaired),),1/len(repaired),dtype=torch.float64)},out/f'repaired_supports_{method}.pt')
        rows.append({'method':method,'beta_geo':beta,**g});v.append({'method':method,'beta_geo':beta,**validity(current,repaired)})
        manifest.append({'method':method,'beta_geo':beta,'status':'feasible','invalid_count':sum(r['raw_invalid'] for r in entries),
                         'invalid_mass':sum(r['mass'] for r in entries if r['raw_invalid']),
                         'alpha_max':max(r['alpha'] for r in entries),'alpha_mean':np.mean([r['alpha'] for r in entries]),
                         'mean_displacement':np.mean([r['displacement'] for r in entries]),'max_displacement':max(r['displacement'] for r in entries),
                         'joint_budget_repair_used':info['fallback_used']})
    pd.DataFrame(manifest).to_csv(out/'repair_manifest.csv',index=False);pd.DataFrame(alpha).to_csv(out/'repair_alpha.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'repaired_geometry.csv',index=False);pd.DataFrame(v).to_csv(out/'repaired_validity.csv',index=False)
    files=[f'repaired_supports_{r["method"]}.pt' for r in manifest if r['status']=='feasible']+['repair_manifest.csv']
    e.p.write_json(out/'frozen_repair.json',{'files':{n:e.p.file_hash(out/n) for n in files},'raw_sha256':e.p.file_hash(out/'frozen_raw.json')})


def paths(args):
    out,meta,train,current=e.context(args);verify_files(out,'frozen_repair.json');verify_files(out,'frozen_raw.json')
    split=args.split;config=read(out/'resolved_config.json')
    if split!='train':
        verify_files(out,'frozen_paths.json')
        if not (out/'selection.json').exists():raise ValueError('freeze training selection before heldout path evaluation')
        src=PARENT/f'{split}_population.pt'
        if e.p.file_hash(src)!=(Path(meta['v3_instance'])/src.name).with_suffix('.sha256').read_text().strip():raise ValueError('heldout population changed')
        shutil.copyfile(src,out/src.name)
    pop=torch.load(out/f'{split}_population.pt',weights_only=True)
    old=torch.load(PARENT/f'paths_{split}.pt',weights_only=True);graph=torch.load(GRAPH/f'paths_{split}.pt',weights_only=True)
    for collection in [old,graph]:
        torch.testing.assert_close(collection['source'],pop['source'],rtol=0,atol=0)
        torch.testing.assert_close(collection['source_ids'],pop['source_ids'],rtol=0,atol=0)
    collection={'source':pop['source'],'source_ids':pop['source_ids'],'methods':{**old['methods'],**{m:graph['methods'][m] for m in CONTROLS if m.startswith('F_')}}}
    active=[m for m in RAW+REPAIRED if (out/f'{"raw" if m.startswith("U_") else "repaired"}_supports_{m}.pt').exists()]
    checks=[];geometries=[]
    for method in active:
        beta=int(method[-3:])/100;label='raw' if method in RAW else 'repaired'
        q=torch.load(out/f'{label}_supports_{method}.pt',weights_only=True)['points']
        p0,p1,g=geometry(pop['path']['base'],q,train['target'],exact_w2(pop['path']['base'],train['target']).distance,beta*config['D'])
        path=glued_arclength_path(pop['path']['base'],q,train['target'],p0,p1,LAMBDAS)
        collection['methods'][method]=path;geometries.append({'split':split,'method':method,**g,'control_budget_feasible':g['slack']>=-config['geometry_tolerance']})
    for method,path in collection['methods'].items():
        for lam,q in zip(LAMBDAS,path['q']):checks.append({'split':split,'method':method,'lambda':lam,**validity(current,q,path['weights'])})
    e.save(collection,out/f'paths_{split}.pt');pd.DataFrame(checks).to_csv(out/f'path_validity_{split}.csv',index=False)
    pd.DataFrame(geometries).to_csv(out/f'path_control_geometry_{split}.csv',index=False)
    if split=='train':
        files=['paths_train.pt','path_validity_train.csv','path_control_geometry_train.csv']
        e.p.write_json(out/'frozen_paths.json',{'files':{n:e.p.file_hash(out/n) for n in files},'raw_sha256':e.p.file_hash(out/'frozen_raw.json'),'repair_sha256':e.p.file_hash(out/'frozen_repair.json')})


def discover(args):
    out,meta,train,current=e.context(args);frozen=verify_files(out,'frozen_paths.json')
    paths=torch.load(out/'paths_train.pt',weights_only=True)['methods']
    jobs=[(m,l,r) for m in RAW+REPAIRED if m in paths for l in [.25,.5,.75] for r in range(3)]
    for job,(method,lam,rep) in enumerate(jobs):
        if job%args.workers!=args.worker:continue
        file=out/'fresh_attacks'/f'{method}_r3_l{lam:.2f}_rep{rep}.pt'
        if file.with_suffix('.models.json').exists():continue
        seed=174000000+(RAW+REPAIRED).index(method)*100000+round(lam*100)*10+rep
        path=paths[method];start=time.perf_counter()
        candidates,tags,trace,full,runtime,peak=e.run_search(current,train,path['q'][LAMBDAS.index(lam)],path['weights'],3*meta['b_ref'],seed,meta['full_update_seed'],args.device)
        obj={'instance_id':meta['instance_id'],'phase':'fresh','method':method,'lambda':lam,'radius':3.,'replication':rep,
             'search_seed':seed,'full_update_seed':meta['full_update_seed'],'candidates':candidates,'shifted_training_distributions':[train['x']+d.cpu() for d in candidates],
             'source_beta_geo':int(method[-3:])/100,'source_label':'raw' if method in RAW else 'repaired','tags':tags,'discovery_split':'train','frozen_geometry':frozen}
        e.save(obj,file);trace.to_csv(file.with_suffix('.truncated.csv'),index=False);full.to_csv(file.with_suffix('.full.csv'),index=False)
        e.replay_candidates(out,meta,train,current,file,obj,args.device)
        replay=read(file.with_suffix('.models.json'))
        for i,a in enumerate(replay['candidates']):a.update(candidate_id=f'{args.output.name}::{a["candidate_id"]}',source_beta_geo=obj['source_beta_geo'],source_label=obj['source_label'],shifted_training_file=str(file),shifted_training_index=i,labels_and_row_ids_reference=str(out/'training.pt'))
        e.p.write_json(file.with_suffix('.models.json'),replay)
        e.p.write_json(file.with_suffix('.runtime.json'),{'seconds':time.perf_counter()-start,'peak_cuda_bytes':peak})
        verify_files(out,'frozen_paths.json');print('fresh complete',method,lam,rep,flush=True)


def freeze_bank(args):
    out,meta,train,current=e.context(args);verify_files(out,'frozen_paths.json')
    methods=torch.load(out/'paths_train.pt',weights_only=True)['methods'];fresh=[]
    for m in RAW+REPAIRED:
        if m not in methods:continue
        for l in [.25,.5,.75]:
            for rep in range(3):fresh.extend(read(out/'fresh_attacks'/f'{m}_r3_l{l:.2f}_rep{rep}.models.json')['candidates'])
    fresh=verify_bank(fresh,meta,train);old=verify_bank(read(out/'old_bank.json')['candidates'],meta,train)
    all_records=old+fresh
    if len(set(a['candidate_id'] for a in all_records))!=len(all_records):raise ValueError('bank identity collision')
    e.p.write_json(out/'evaluation_bank.json',{'candidates':all_records,'frozen_paths_sha256':e.p.file_hash(out/'frozen_paths.json')})
    pd.DataFrame(fresh).to_csv(out/'fresh_attack_manifest.csv',index=False);pd.DataFrame(all_records).to_csv(out/'full_attack_bank_manifest.csv',index=False)


def score(args):
    out,meta,train,current=e.context(args);verify_files(out,'frozen_paths.json')
    records=read(out/'evaluation_bank.json')['candidates']
    e.curve_evaluation(out,meta,train,current,records[args.worker::args.workers],args.split,args.device,f'shard{args.worker}')


def merge(args):
    out=args.output/'heloc';merger.METHODS=list(torch.load(out/f'paths_{args.split}.pt',weights_only=True)['methods'])
    e.verify_eval_bank=lambda out,meta,train:read(out/'evaluation_bank.json')['candidates']
    merger.merge(args)


def select(args):
    out=args.output/'heloc';frame=pd.read_csv(out/'bank_train_curves.csv');checks=pd.read_csv(out/'path_validity_train.csv')
    merged=frame.merge(checks[['method','lambda','invalid_support_count']],on=['method','lambda'])
    rows=[]
    for method,group in merged.groupby('method',sort=False):
        # Raw/legacy curves are diagnostic controls; deployable repaired methods must be fully valid.
        eligible=group[group.invalid_support_count==0] if method in REPAIRED else group
        if eligible.empty:continue
        tied=eligible[eligible.bank_hinge==eligible.bank_hinge.min()].sort_values('lambda')
        row=tied.iloc[0].to_dict();row['tied_lambdas']=tied['lambda'].tolist();row['selection_current_validity_required']=method in REPAIRED;rows.append(row)
    e.p.write_json(out/'selection.json',{'selected':rows,'selection_split':'train','evaluation_bank_sha256':e.p.file_hash(out/'evaluation_bank.json'),
                  'selection_paths_sha256':e.p.file_hash(out/'paths_train.pt'),'rule':'training-bank exact argmin; repaired methods restricted to wholly current-valid final distributions'})
    pd.DataFrame(rows).to_csv(out/'selected_configs.csv',index=False)


def direct(args):
    """Evaluate raw/repaired controls themselves and selected realized outputs."""
    out,meta,train,current=e.context(args);config=read(out/'resolved_config.json')
    records=read(out/'evaluation_bank.json')['candidates'];bank=ModelBank([e.legacy.load_model(Path(a['model_path']),22) for a in records],args.device)
    future_manifest=pd.read_csv(Path(meta['original_instance'])/'future_final_manifest.csv');old_eval=read(PARENT/'evaluation_manifest.json')
    future_banks={}
    for protocol in ['aligned_full_update','scratch_stress']:
        files=[PARENT/'aligned_future_models'/f'model{i}.pt' for i in future_manifest.index] if protocol=='aligned_full_update' else [Path(p) for p in future_manifest.path]
        for f in files:
            if e.p.file_hash(f)!=old_eval['model_hashes'][str(f)]:raise ValueError('future model changed')
        future_banks[protocol]=ModelBank([e.legacy.load_model(f,22) for f in files],args.device)
    raw_mask=np.array([a['phase']!='fresh' for a in records]);per_model=[];rows=[]
    pop=torch.load(out/'train_population.pt',weights_only=True)
    def evaluate(name,q,weights,source_index,geometry_row,kind,beta,split,lam=None):
        point=q.to(args.device);w=weights.to(args.device).double();hs=[]
        with torch.no_grad():
            for start in range(0,bank.count,64):
                pred=torch.sigmoid(bank.logits(point,start,start+64));hs.extend((torch.relu(.5-pred).double()*w).sum(1).cpu().tolist())
        hs=np.asarray(hs)
        for i,h in enumerate(hs):per_model.append({'kind':kind,'method':name,'split':split,'lambda':lam,'candidate_id':records[i]['candidate_id'],'hinge':h})
        factual=torch.load(out/f'{split}_population.pt',weights_only=True)['source'][source_index]
        cost=float((torch.linalg.vector_norm(q-factual,dim=1).double()*weights).sum())
        for protocol,future in future_banks.items():
            with torch.no_grad():pred=torch.sigmoid(future.logits(point));h=(torch.relu(.5-pred).double()*w).sum(1).cpu().numpy();v=((pred>=.5).double()*w).sum(1).cpu().numpy()
            rows.append({'kind':kind,'method':name,'beta_geo':beta,'split':split,'lambda':lam,'protocol':protocol,
                         'training_bank_hinge':float(hs[raw_mask].max()),'fresh_bank_hinge':float(hs.max()),
                         'mean_future_validity':float(v.mean()),'worst_future_validity':float(v.min()),'mean_future_hinge':float(h.mean()),
                         'worst_future_hinge':float(h.max()),'recourse_cost':cost,**validity(current,q,weights),**geometry_row})
    for beta,u,p in zip(BETAS,RAW,REPAIRED):
        for method,label in [(u,'raw'),(p,'repaired')]:
            file=out/f'{label}_supports_{method}.pt'
            if not file.exists():continue
            q=torch.load(file,weights_only=True)['points'];p0,p1,g=geometry(pop['path']['base'],q,train['target'],config['D'],beta*config['D'])
            # Split Q by its exact factual-to-control plan to preserve recourse identities.
            evaluate(method,q[p0.target_index],p0.weights,p0.source_index,g,'control',beta,'train')
    selection=read(out/'selection.json')
    for split in ['train','test']:
        paths=torch.load(out/f'paths_{split}.pt',weights_only=True)['methods'];population=torch.load(out/f'{split}_population.pt',weights_only=True)
        D=exact_w2(population['path']['base'],train['target']).distance
        for chosen in selection['selected']:
            method=chosen['method'];lam=chosen['lambda'];path=paths[method];q=path['q'][LAMBDAS.index(lam)]
            beta=int(method[-3:])/100 if 'beta' in method else None
            # Graph controls retain their own distinct intrinsic budget diagnostics.
            if method.startswith('F_'):g={'D':D,'geometry_metric':'legacy_graph_control_not_reinterpreted_as_budget'}
            else:
                budget=(beta or 1.)*config['D'];_,_,g=geometry(population['path']['base'],q,train['target'],D,budget,path['weights'].numpy())
                g['final_geometry_feasible']=g['slack']>=-config['geometry_tolerance'];g['geometry_metric']='Euclidean W2'
            evaluate(method,q,path['weights'],path['source_index'],g,'selected_path',beta,split,lam)
            if method.startswith('S_beta'):
                # Secondary matched final-output diagnostic, frozen training-selected lambda;
                # no new Q optimizer, no rematching of raw/repair factual assignments.
                lam=config['secondary_training_lambdas'][method];q=path['q'][LAMBDAS.index(lam)]
                # W(Q,P1) must use the actual original target population, not duplicated
                # per-atom anchors. Perform minimal point repair then exact original-marginal OT.
                repaired,entries=minimal_restore(current,q,train['target'][path['target_index']],path['weights'])
                _,_,g2=geometry(population['path']['base'],repaired,train['target'],D,(beta or 1.)*config['D'],path['weights'].numpy())
                g2['final_geometry_feasible']=g2['slack']>=-config['geometry_tolerance'];g2['geometry_metric']='Euclidean W2'
                e.save({'raw':q,'repaired':repaired,'weights':path['weights'],'source_index':path['source_index'],'target_index':path['target_index'],'lambda':lam},out/f'final_output_diagnostic_{split}_{method}.pt')
                pd.DataFrame(entries).to_csv(out/f'final_output_alpha_{split}_{method}.csv',index=False)
                evaluate('T_'+method,repaired,path['weights'],path['source_index'],g2,'secondary_final_output_repair',beta,split,lam)
    pd.DataFrame(rows).to_csv(out/'all_metrics.csv',index=False);pd.DataFrame(per_model).to_csv(out/'direct_cross_eval_matrix.csv',index=False)
    data=pd.DataFrame(rows);data[data.kind=='control'].to_csv(out/'raw_vs_repaired_metrics.csv',index=False)
    data[data.kind!='control'].to_csv(out/'final_metrics.csv',index=False)


def report(args):
    out=args.output/'heloc'
    raw=pd.read_csv(out/'raw_geometry.csv');fixed=pd.read_csv(out/'repaired_geometry.csv');metrics=pd.read_csv(out/'all_metrics.csv')
    curves=pd.concat([pd.read_csv(out/f'bank_{s}_curves.csv') for s in ['train','test']]);curves.to_csv(out/'lambda_curves.csv',index=False)
    # Compact pointer manifest avoids duplicating multi-million-row replay data.
    pd.DataFrame([{'split':s,'matrix':str(out/f'bank_{s}_replay.npz'),'contents':'hinge,invalidity,candidate_ids,methods,lambdas'} for s in ['train','test']]).to_csv(out/'cross_eval_matrix.csv',index=False)
    text='# Continuous WRO with minimal validity restoration\n\nRaw continuous geometry-only optimization; f0 is absent from learning. Repair occurs only after raw freeze. No graph restriction, validity loss or construction margin.\n\n'
    text+='## Raw geometry\n\n'+raw.to_markdown(index=False)+'\n\n## Repaired geometry\n\n'+fixed.to_markdown(index=False)+'\n\n'
    text+='## Matched metrics\n\n'+metrics.to_markdown(index=False)+'\n\n'
    text+='The old learned controls were already current-valid: the previous failure was mainly on interpolated outputs. Primary repaired-control paths therefore use valid-only training-bank selection. The separately labeled final-output diagnostic repairs only the final split population of the historical strong paths; it is not the same control or path and receives no targeted fresh discovery. It is evaluated against the same expanded bank, with no feedback into learning. Final-output diagnostics whose exact budgets fail are undeployable; they are not silently accepted.\n\nFinite bank, support and numerical local optimization; no unrestricted global WRO or repair optimality certificate. First crossings are numerically enumerated for a ReLU network, not interval-certified. Future models are previously inspected retrospective scenarios. Query validity/budget feasibility does not automatically follow from training.\n'
    (out/'report.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','smoke','optimize','freeze_raw','repair','paths','discover','freeze_bank','score','merge','select','direct','report'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dataset',default='heloc')
    parser.add_argument('--device',default='cpu');parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--split',default='train')
    args=parser.parse_args();e.SETTINGS={**e.SETTINGS,'evaluation_radii':[3.],'lambdas':LAMBDAS}
    globals()[args.phase](args)

if __name__=='__main__':main()
