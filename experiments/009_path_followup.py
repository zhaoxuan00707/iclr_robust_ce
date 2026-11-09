"""Post-construction stages with namespaced attacks and fixed-r selection.

This entry point preserves the active construction source fingerprint. It uses
only the repository-local implementation and writes into its new run directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import run
run.bootstrap()
from lift_ce.experiments import path_level_geodesic_wro_v1 as m
from lift_ce.experiments import transport_spline_path_v1 as shard
from lift_ce.experiments import fixed_risk_geometry_v1 as e


def freeze_followup_source(out):
    file=Path(__file__).resolve()
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest=out/'followup_source.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['sha256']!=digest:raise ValueError('follow-up implementation changed')
    else:
        m.write_json(manifest,{'file':str(file),'sha256':digest,'training_paths_sha256':e.p.file_hash(out/'frozen_paths.json')})
        (out/'followup_source.py').write_bytes(file.read_bytes())


def collect_heldout(args):
    allowed=m.verify_paths(args.output/'heloc')['feasible_training_methods']
    original=m.inputs
    def restricted_inputs(*a,**kw):
        values=list(original(*a,**kw)); config=dict(values[1])
        config['beta_geo']=[b for b in config['beta_geo'] if f'P_beta{round(b*100):03d}' in allowed]
        values[1]=config;return tuple(values)
    m.inputs=restricted_inputs
    m.collect(args)


def discover(args):
    out,meta,train,current=e.context(args)
    frozen=m.verify_paths(out)
    for split in ['cal','test']:m.require_query(args,split)
    collection=torch.load(out/'paths_train.pt',weights_only=True)
    jobs=[(method,lam,rep) for method in collection['methods'] for lam in e.SETTINGS['source_lambdas'] for rep in range(3)]
    for job,(method,lam,rep) in enumerate(jobs):
        if job%args.workers!=args.worker:continue
        file=out/'fresh_attacks'/f'{method}_r3_l{lam:.2f}_rep{rep}.pt'
        if file.with_suffix('.models.json').exists():
            obj=torch.load(file,weights_only=True)
            if obj['frozen_geometry']!=frozen:raise ValueError('fresh source path changed')
            continue
        seed=163000000+list(collection['methods']).index(method)*100000+int(lam*100)*10+rep
        path=collection['methods'][method];q=path['q'][e.SETTINGS['lambdas'].index(lam)]
        started=time.perf_counter()
        candidates,tags,trace,full,runtime,peak=e.run_search(current,train,q,path['weights'],3*meta['b_ref'],seed,meta['full_update_seed'],args.device)
        source_beta=float(method.split('beta')[-1])/100 if '_beta' in method else None
        obj={'instance_id':meta['instance_id'],'phase':'fresh','method':method,'lambda':lam,'radius':3.,
             'replication':rep,'search_seed':seed,'full_update_seed':meta['full_update_seed'],
             'candidates':candidates,'tags':tags,'discovery_split':'train','frozen_geometry':frozen,
             'source_beta_geo':source_beta,'shifted_training_distributions':[train['x']+d.cpu() for d in candidates]}
        e.save(obj,file);trace.to_csv(file.with_suffix('.truncated.csv'),index=False);full.to_csv(file.with_suffix('.full.csv'),index=False)
        e.replay_candidates(out,meta,train,current,file,obj,args.device)
        side=json.loads(file.with_suffix('.models.json').read_text())
        for index,row in enumerate(side['candidates']):
            row.update(candidate_id=f'path_level/{args.output.name}/{row["candidate_id"]}',
                       source_beta_geo=source_beta,shifted_training_file=str(file),shifted_training_index=index,
                       labels_and_row_ids_reference=str(out/'training.pt'))
        m.write_json(file.with_suffix('.models.json'),side)
        m.write_json(file.with_suffix('.runtime.json'),{'total_seconds':time.perf_counter()-started,'search_seconds':runtime,'peak_cuda_bytes':peak})
        m.verify_paths(out)
        print('fresh complete',method,lam,rep,flush=True)


def score(args):
    out,meta,train,current=e.context(args)
    records=e.verify_eval_bank(out,meta,train)
    if args.split!='train':m.require_query(args,args.split)
    e.curve_evaluation(out,meta,train,current,records[args.worker::args.workers],args.split,args.device,f'shard{args.worker}')


def merge(args):
    out=args.output/'heloc'
    shard.METHODS=list(torch.load(out/f'paths_{args.split}.pt',weights_only=True)['methods'])
    shard.merge(args)


def select(args):
    out,meta,train,_=e.context(args)
    m.require_query(args,'cal');e.verify_eval_bank(out,meta,train)
    frame=pd.read_csv(out/'bank_cal_curves.csv')
    choices=e.exact_lambda_choices(frame)
    m.write_json(out/'selection.json',{'instance_id':meta['instance_id'],'frozen_geometry':m.verify_paths(out),
        'evaluation_bank_sha256':e.p.file_hash(out/'evaluation_bank.json'),'selection_split':'cal',
        'selection_paths_sha256':e.p.file_hash(out/'paths_cal.pt'),'calibration_model_hashes':{},
        'deployment_calibrated':True,'selected':choices,'fixed_radius_choices':choices,
        'radius_rule':'fixed r/b_ref=3; no radius or beta selection; exact ties choose smallest lambda'})
    pd.DataFrame(choices).to_csv(out/'selected_configs.csv',index=False)


def finish(args):
    out=args.output/'heloc'; repo=Path(__file__).resolve().parents[1]
    selection=e.verify_selection(out)
    config=json.loads((out/'resolved_config.json').read_text())
    futures=pd.read_csv(out/'selected_future_results.csv')
    choices={row['method']:row for row in selection['selected']}
    rows=[]
    for beta in config['beta_geo']:
        method=f'P_beta{round(beta*100):03d}'
        statuses={split:pd.read_csv(out/f'{split}_path_feasibility.csv') for split in ['train','cal','test']}
        row={'method':method,'beta_geo':beta,'r_over_b_ref':3.,'B_geo':beta*config['D_train']}
        feasible=True
        for split,frame in statuses.items():
            found=frame[frame.method==method]
            status=found.iloc[0].status if len(found) else 'NO_FEASIBLE_TRAINING_PATH'
            row[f'{split}_status']=status
            feasible &= status=='FEASIBLE'
            if len(found):
                for field in ['total_length','slack','maximum_grid_invalid_mass','minimum_f0_score']:
                    row[f'{split}_{field}']=found.iloc[0].get(field)
        row['deployment_feasible']=feasible
        chosen=choices.get(method)
        future=futures[(futures.method==method)&(futures.operator=='aligned_full_update')]
        if chosen and len(future):
            row.update(selected_lambda=chosen['lambda'],fresh_bank_hinge=chosen['bank_hinge'],construction_bank_hinge=chosen['before_fresh_hinge'])
            row.update({k:future.iloc[0][k] for k in ['mean_future_validity','worst_future_validity','mean_future_hinge','worst_future_hinge','recourse_cost','current_validity']})
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(out/'final_metrics.csv',index=False)
    result.to_csv(repo/'results/005_path_final_metrics.csv',index=False)
    # Keep all empirical curves and the complete model-by-population replay.
    curves=pd.concat([pd.read_csv(out/f'bank_{s}_curves.csv') for s in ['train','cal','test']])
    curves.to_csv(out/'lambda_curves.csv',index=False)
    first=True
    for split in ['train','cal','test']:
        a=np.load(out/f'bank_{split}_replay.npz')
        for mi,method in enumerate(a['methods']):
            for li,lam in enumerate(a['lambdas']):
                frame=pd.DataFrame({'candidate_id':a['candidate_ids'],'method':method,'split':split,'lambda':lam,
                                    'hinge':a['hinge'][:,mi,li],'invalidity':a['invalidity'][:,mi,li]})
                frame.to_csv(out/'cross_eval_matrix.csv',mode='w' if first else 'a',header=first,index=False);first=False
    records=json.loads((out/'evaluation_bank.json').read_text())['candidates']
    pd.DataFrame([r for r in records if r['phase']=='fresh']).to_csv(out/'fresh_attack_manifest.csv',index=False)
    pd.concat([pd.read_csv(out/f'{s}_path_feasibility.csv') for s in ['cal','test']]).to_csv(out/'heldout_feasibility.csv',index=False)
    pd.read_csv(out/'train_path_feasibility.csv').to_csv(out/'path_feasibility.csv',index=False)
    traces=pd.concat([pd.read_csv(out/f'optimization_trace_{s}.csv') for s in ['train','cal','test']])
    traces.to_csv(out/'optimization_trace.csv',index=False)
    traces[['split','beta_geo','initialization','step','nu_geo','nu_valid','rho_geo','rho_valid','g_geo','g_valid']].to_csv(out/'dual_trace.csv',index=False)
    # Straight and legacy controls remain separately labeled diagnostics.
    futures.to_csv(repo/'results/005_path_all_method_future_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'plots';folder.mkdir(exist_ok=True)
    fc=pd.read_csv(out/'future_curves.csv');fc=fc[(fc.operator=='aligned_full_update')&(fc.split=='test')]
    for name,column,frame in [('bank_hinge','bank_hinge',curves[curves.split=='cal']),('future_validity','mean_future_validity',fc),('recourse_cost','recourse_cost',fc)]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for method,g in frame.groupby('method'):
            ax.plot(g['lambda'],g[column],label=method)
            if method in choices:
                lam=choices[method]['lambda'];point=g[g['lambda']==lam]
                if len(point):ax.scatter([lam],[point.iloc[0][column]],s=20)
        ax.set(xlabel='lambda',ylabel=column,title='HELOC fixed r=3, exploratory; no smoothing')
        ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=140);plt.close(fig)
    completed={'status':'COMPLETED_EVALUATION','deployment_feasible_beta_count':int(result.deployment_feasible.sum()),
               'construction_bank_size':config['construction_bank_size'],'expanded_bank_size':len(records),
               'no_global_guarantee':True,'exploratory_only':True}
    m.write_json(out/'completion.json',completed)
    m.write_json(repo/'results/005_path_completion.json',completed)
    report='# Path-level experiment completed\n\n'+result.to_string(index=False)+'\n\n'
    report+='All methods use a shared expanded attack bank. Only rows passing train/cal/test path feasibility are accepted. Historical controls may be infeasible and remain diagnostic. Mean future validity at a higher cost is not matched-cost superiority.\n\n'
    report+='Finite bank/support, local numerical OT, discrete validity grid, previously inspected future scenarios and overlapping historical future states limit interpretation. No unrestricted global optimum or fresh confirmation is claimed.\n\nRun: '+str(args.output)+'\n'
    (out/'report.md').write_text(report);(repo/'results/005_path_level_task_report.md').write_text(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['initialize','optimize','collect','discover','freeze_eval_bank','score','merge','select','future_models','evaluate','finish'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dataset',default='heloc')
    parser.add_argument('--split',default='train');parser.add_argument('--device',default='cpu')
    parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();out=args.output/'heloc'
    if not args.output.resolve().is_relative_to(run.RUN_ROOT):raise ValueError('only new repository run outputs are allowed')
    if not m.verify_paths(out)['feasible_training_methods']:raise ValueError('no feasible training path to evaluate')
    freeze_followup_source(out)
    e.SETTINGS={**e.SETTINGS,'evaluation_radii':[3.]};e.verify_geometry=m.verify_paths;e.query_paths=m.require_query
    if args.phase=='initialize':return
    if args.phase=='optimize':
        if args.split not in ['cal','test']:raise ValueError('follow-up optimizer is held-out inference only')
        m.optimize(args)
    elif args.phase=='collect':collect_heldout(args)
    elif args.phase in ['discover','score','merge','select','finish']:globals()[args.phase](args)
    elif args.phase=='evaluate':e.evaluate(args,score_bank=False)
    else:getattr(e,args.phase)(args)

if __name__=='__main__':main()
