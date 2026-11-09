"""Independent saved-output checks and lightweight Task100 tables."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/src'),str(ROOT/'experiments/vendor/RobustX'),str(ROOT/'experiments/vendor/ellice')]
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.geodesic_excess import exact_w2
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_worker(path):
    p=load(path/'prepared.pt');d=p['data'];meta=json.loads((path/'metadata.json').read_text());cfg=meta['config']
    m=TabularMLP(d['train_x'].shape[1],p['hidden']);m.load_state_dict(load(path/'current.pt'));m.eval()
    domain=MixedDomain(d['train_x'],p['groups']); checks=[]
    splits=[set(map(str,d[k+'_ids'])) for k in ['train','cal','test']]
    assert all(not a&b for i,a in enumerate(splits) for b in splits[i+1:])
    assert (path/'frozen.json').stat().st_mtime<(path/'future_test.pt').stat().st_mtime
    banks={};hashes={}
    for split in ['cal','test']:
        obj=load(path/f'future_{split}.pt');models=[];hashes[split]=[]
        assert len(obj['states'])==45
        for state,buffers in obj['states']:
            mm=TabularMLP(d['train_x'].shape[1],p['hidden']);mm.load_state_dict({**state,**buffers});mm.eval();models.append(mm)
            h=hashlib.sha256()
            for key,value in state.items():h.update(key.encode());h.update(value.numpy().tobytes())
            hashes[split].append(h.hexdigest())
        banks[split]=models
    checks.append(dict(worker=path.name,check='split_and_freeze',passed=True,cal_unique_models=len(set(hashes['cal'])),test_unique_models=len(set(hashes['test'])),shared_cal_test_models=len(set(hashes['cal'])&set(hashes['test']))))
    # Independently reconstruct all held-out update trajectories; no generator helper.
    for index,record in enumerate(load(path/'future_test.pt')['records']):
        x=d['train_x'].clone();y=d['train_y'].clone();ids=record['numeric_features'];g=torch.Generator().manual_seed(record['seed']);severity=record['severity'];target=severity*p['bref']
        if record['family']=='mean':x[:,ids]+=target/np.sqrt(len(ids))*torch.where(torch.rand(len(ids),generator=g)>.5,1.,-1.)
        elif record['family']=='scale':
            v=x[:,ids];center=v.mean(0);norm=((v-center).square().sum(1).mean()).sqrt().clamp_min(1e-6);x[:,ids]=center+(1+target/norm)*(v-center)
        else:
            v=x[:,ids[0]];cut=v.median();lo=(v<=cut).nonzero().flatten();hi=(v>cut).nonzero().flatten()
            if not len(hi):hi=lo
            nlo=int(len(x)*min(.5+.18*severity,.95));sel=torch.cat([lo[torch.randint(len(lo),(nlo,),generator=g)],hi[torch.randint(len(hi),(len(x)-nlo,),generator=g)]])
            x=x[sel];y=y[sel]
        audit_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mm=TabularMLP(x.shape[1],p['hidden']);mm.load_state_dict(load(path/'current.pt'));mm=mm.to(audit_device);mm.eval();x=x.to(audit_device);y=y.to(audit_device)
        for _ in range(cfg['update']['steps']):
            loss=torch.nn.functional.binary_cross_entropy_with_logits(mm(x),y)
            grads=torch.autograd.grad(loss,tuple(mm.parameters()))
            with torch.no_grad():
                for parameter,grad in zip(mm.parameters(),grads):parameter.copy_(parameter-cfg['update']['lr']*grad)
        error=max(float((v.cpu()-banks['test'][index].state_dict()[k]).abs().max()) for k,v in mm.state_dict().items())
        # CPU versus CUDA arithmetic; absolute tolerance small relative to weights.
        assert error<2e-5,(path,index,error)
        checks.append(dict(worker=path.name,check='future_reconstruction',index=index,error=error,device=str(audit_device),passed=True))
    frame=pd.read_csv(path/'results.csv')
    for row_index,row in enumerate(frame.to_dict('records')):
        obj=load(path/f"test_{row['method']}.pt");z=obj['cf']
        if row['realization']=='hard':z=domain.project(z,True)
        with torch.no_grad():
            predictions=torch.stack([mm(z).sigmoid()>=.5 for mm in banks['test']]);cv=float((m(z).sigmoid()>=.5).float().mean())
        cost=float((z-p['sources']['test']).norm(dim=1).mean());fv=float(predictions.float().mean())
        frame.loc[row_index,'invalid_current_count']=int(round(len(z)*(1-cv)))
        frame.loc[row_index,'valid_cf_count']=int(round(len(z)*cv))
        assert abs(cost-row['Cost'])<1e-5 and abs(fv-row['FV'])<1e-5 and abs(cv-row['current_validity'])<1e-5,(path,row)
        if row['method'].startswith('grace'):
            base=load(path/f"test_{row['method'].removeprefix('grace_')}.pt")['cf']
            assert len(obj['q'])==len(base) and len(torch.unique(obj['assignment']))==len(base)
            assert torch.equal(obj['q'][obj['assignment']],obj['cf'])
            detour=exact_w2(base,z).distance+exact_w2(z,p['p1']).distance
            assert abs(detour-row['detour'])<1e-5
        checks.append(dict(worker=path.name,check='output_metrics',method=row['method'],realization=row['realization'],passed=True))
    # Hard realization can break geometry even when the relaxed solve is feasible.
    if 'geometry_feasible' in frame:
        bad=(frame.realization=='hard') & frame.geometry_feasible.eq(False)
        frame.loc[bad,'status']='infeasible_hard_realization'
        frame.loc[bad,'failure_aware_FV']=np.nan
    cal=pd.read_csv(path/'calibration.csv');selected=json.loads((path/'selection.json').read_text())
    for row in cal.to_dict('records'):
        if row['method'].startswith('grace_'):
            name=row['method'].removeprefix('grace_')
            artifact=path/f"cal_grace_{name}_b{row['beta']:g}_r{row['r_ratio']:g}.pt"
        else:artifact=path/f"cal_{row['method']}_{row['param']:g}.pt"
        obj=load(artifact);z=obj['cf']
        with torch.no_grad():
            fv=float(torch.stack([mm(z).sigmoid()>=.5 for mm in banks['cal']]).float().mean())
        cost=float((z-p['sources']['cal']).norm(dim=1).mean())
        assert abs(fv-row['FV'])<1e-5 and abs(cost-row['Cost'])<1e-5,(path,artifact.name,'calibration_metric')
        if row['method'].startswith('grace_'):
            base=load(path/f'cal_{name}_0.pt')['cf'];distance=exact_w2(base,z).distance+exact_w2(z,p['p1']).distance
            feasible=distance<=obj['budget']+cfg['solver']['feas_tol']*max(1.,obj['D'])
            assert feasible==obj['feasible']
        checks.append(dict(worker=path.name,check='calibration_output',artifact=artifact.name,passed=True))
    # Audit native ElliCE robustness separately from FV and nonempty returns.
    if 'ellice' in selected and (path/'test_ellice.pt').exists():
        from lift_ce.baselines.official_adapters import _tabular_wrapper_class
        from ellice import Data
        from ellice.generators.continuous import ContinuousGenerator
        lo,sc=domain.low,domain.scale
        training=pd.DataFrame(((d['train_x']-lo)/sc).numpy(),columns=d['feature_names']);training['target']=d['train_y'].numpy()
        generator=ContinuousGenerator(model=_tabular_wrapper_class(sc,lo)(m),data=Data(dataframe=training,target_column='target'),eps=selected['ellice']['param'],reg_coef=1e-4,device='cpu')
        z=(load(path/'test_ellice.pt')['cf']-lo)/sc
        with torch.no_grad():
            h=generator._get_penult_features(z);h=torch.cat([h,torch.ones(len(h),1)],1)
            margin=h@generator.omega_c-torch.linalg.vector_norm(generator.Q_inv_sqrt@h.T,dim=0)
        frame.loc[frame.method=='ellice','native_ellipsoid_robust_fraction']=float((margin>0).float().mean())
        checks.append(dict(worker=path.name,check='ellice_native_robustness_measured',passed=True,robust_count=int((margin>0).sum()),total=len(z)))

    for method,choice in selected.items():
        if method.startswith('grace_') and 'r_ratio' in choice:
            frame.loc[frame.method==method,'selected_r_ratio']=choice['r_ratio']
            frame.loc[frame.method==method,'selected_r']=choice['r_ratio']*p['bref']
            frame.loc[frame.method==method,'selected_beta']=choice['beta']
        elif 'param' in choice:frame.loc[frame.method==method,'selected_native_parameter']=choice['param']
        subset=cal[cal.method==method]
        if method.startswith('grace'):
            subset=subset[subset.status=='feasible'].sort_values(['FV','Cost','r_ratio','beta'],ascending=[False,True,True,True])
            if len(subset):assert subset.iloc[0].r_ratio==choice['r_ratio'] and subset.iloc[0].beta==choice['beta']
            else:assert choice['status']=='no_feasible_calibration'
        else:
            subset=subset[subset.status=='executed'].sort_values(['FV','Cost','param'],ascending=[False,True,True])
            if len(subset):assert subset.iloc[0].param==choice['param']
    checks.append(dict(worker=path.name,check='calibration_selection',passed=True))
    for method,choice in selected.items():
        if choice.get('status')=='no_feasible_calibration':
            frame=pd.concat([frame,pd.DataFrame([dict(dataset=path.name.rsplit('_',1)[0],classifier=path.name.rsplit('_',1)[1],method=method,realization='relaxed' if domain.groups else 'native',n=len(p['sources']['test']),status='no_feasible_calibration')])],ignore_index=True)
    runtime=pd.read_csv(path/'runtime.csv') if (path/'runtime.csv').exists() else pd.DataFrame()
    if len(runtime):
        runtime['dataset']=path.name.rsplit('_',1)[0]
        for row in runtime.to_dict('records'):
            if row['status']=='no_feasible_calibration':continue
            o=load(path/f"timed_{row['method']}.pt");assert len(o['cf'])==10
            if row['method']=='grace_wachter':assert len(o['q'])==10 and len(o['assignment'].unique())==10
            with torch.no_grad():
                z=o['cf'];fv=float(torch.stack([mm(z).sigmoid()>=.5 for mm in banks['test']]).float().mean())
                cv=float((m(z).sigmoid()>=.5).float().mean())
            cost=float((z-p['sources']['test'][:10]).norm(dim=1).mean())
            runtime.loc[runtime.method==row['method'],'invalid_current_count']=int(round(10*(1-cv)))
            runtime.loc[runtime.method==row['method'],'valid_cf_count']=int(round(10*cv))
            assert abs(fv-row['FV'])<1e-5 and abs(cv-row['current_validity'])<1e-5 and abs(cost-row['Cost'])<1e-5
            if row['method']=='grace_wachter':
                b=load(path/'timed_wachter.pt')['cf'];distance=exact_w2(b,z).distance+exact_w2(z,p['p1']).distance
                runtime.loc[runtime.method=='grace_wachter','geometry_feasible']=distance<=o['budget']+cfg['solver']['feas_tol']*max(1.,o['D'])
            checks.append(dict(worker=path.name,check='timed_output_metrics',method=row['method'],passed=True))

        checks.append(dict(worker=path.name,check='fresh_ten_cf_population',passed=True))
    return frame,runtime,checks

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);args=a.parse_args();torch.set_num_threads(1)
    workers=sorted(p for p in args.run.iterdir() if p.is_dir() and (p/'complete.json').exists())
    if len(workers)!=8:raise RuntimeError(f'Only {len(workers)}/8 completed; refusing a final report')
    frames=[];runtimes=[];checks=[]
    for p in workers:
        print('CHECK',p.name,flush=True);f,r,c=check_worker(p);frames.append(f);runtimes.append(r);checks+=c
    results=pd.concat(frames,ignore_index=True);runtime=pd.concat(runtimes,ignore_index=True)
    out=ROOT/'results/exp3_table1_single_run';out.mkdir(exist_ok=True)
    # Previous CSVs were explicitly labeled not-run status markers.
    results.to_csv(out/'combined_summary.csv',index=False)
    results[(results.classifier=='mlp')&~results.method.isin(['dice','grace_dice'])].to_csv(out/'current_mlp.csv',index=False)
    results[results.classifier=='logistic'].to_csv(out/'logistic_regression.csv',index=False)
    results[results.method.isin(['dice','grace_dice'])].to_csv(out/'dice_base_generator.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'independent_checks.csv',index=False)
    manifest={str(p.relative_to(args.run)):digest(p) for p in args.run.rglob('*') if p.is_file()}
    (out/'raw_manifest.json').write_text(json.dumps({'run':str(args.run),'sha256':manifest},indent=2))
    comparisons=[]
    for (dataset,classifier),group in results.groupby(['dataset','classifier']):
        for generator in ['wachter','dice']:
            base=group[group.method==generator]
            refined=group[(group.method=='grace_'+generator)&(group.realization!='hard')]
            if len(base) and len(refined):
                b=base.iloc[0];q=refined.iloc[0]
                comparisons.append(dict(dataset=dataset,classifier=classifier,base=generator,base_FV=b.FV,grace_FV=q.FV,base_Cost=b.Cost,grace_Cost=q.Cost,status=q.status))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/'base_grace_comparisons.csv',index=False)
    table=results[['dataset','classifier','method','realization','n','FV','Cost','status','native_failures']].to_markdown(index=False)
    text=f'''# Task100 Experiment3: single-seed diagnostic reconstruction

Raw run: `{args.run}`. Seed0, eight A100 GPUs, one worker per dataset/classifier.
All {len(checks)} independent checks pass. Train/cal/test split identities,
calibration selection, complete test updates, geometry, assignments and metrics
are independently verified. Classifier FV is reported over all factuals. Missing
or infeasible constructions are failures; inspect `status`, `native_failures`,
`geometry_feasible`, `hard_category_violations` and `failure_aware_FV` alongside FV.
Unsupported categorical configurations have no usable performance claim.

This is **not an exact Table1 reproduction**: original four-dataset run provenance
was untraceable. The declared reconstruction reuses the loader, MLP and logistic
training conventions, official baseline adapters and revised update-aware GRACE.
HELOC and mixed FICO are representations of the same corpus; their loaded versions
have different row/feature counts, not independent benchmark populations.

A.5 categorical uncertainty uses logit softmax with temperature1 and nominal
smoothing1e-4; numerical coordinates are additive. Supports use a weighted simplex
projection in standardized coordinates. `relaxed` scores are not legal discrete
recourse. `hard` is a separate argmax-realization diagnostic with geometry checked
again, not a post-hoc validity correction. No protected/immutable constraints or
causal actionability are established.

45 calibration and45 test scenarios use mean/scale/tail shifts, severities
.5/1/2.5 and five realizations. Full10-step SGD(.01) updates start from the same
fitted classifier. Tail sampling keeps labels paired with rows. Mean signs and
scale transformations can repeat across seeds: independent_checks.csv records
unique/checkpoint overlap counts. Do not describe these as45 independent novel
shift distributions or as complete retraining robustness.

Calibrated GRACE uses r/b_ref in{{0,.25,1,3}}, beta in{{1.05,1.1,1.2}}, up to800
projected augmented-Lagrangian steps, primal1, rho1, temperature.05; the final
iterate is returned and infeasibility is retained. A small successive-objective
change is not a stationarity/global-optimality guarantee. All selections use
calibration FV with cost and smaller-parameter tie-breaks; test scores never tune.

The retained RobustX STCE searches label-negative training examples despite its
positive-candidate docstring; this inherited behavior is disclosed rather than
silently edited. ROAR/PROPLACE mixed-domain constraints are unsupported. Gurobi
license failures/timeouts and returned invalid candidates remain visible. ElliCE
can return a valid yet non-robust point under its native default; inspect the
separately audited `native_ellipsoid_robust_fraction`. The German/logistic
calibration-feasibility-gated follow-up is saved in results/101_ellice_convergence
and does not replace this original protocol. DiCE is
official0.12 random generation, four candidates,5000 samples, minimum valid L2
selection; no handmade replacement. Dependency and source hashes are recorded.

{table}

## Answers to the diagnostic questions

1. Exact Table1 reproduction cannot be established because the historical source
   configuration/run is unverified. The table above gives the new seed0 ranking;
   unsupported/infeasible rows cannot be used to claim a method outperforms another.
2. Logistic-regression results are included on the same split definitions and
   classifier-specific unfavorable cohorts. The Base-to-GRACE comparison below
   reports whether FV improves and the associated proximity change; it does not
   establish a multi-seed ranking.
3. Official DiCE initializes its own independently calibrated GRACE population.
   The comparison below shows its observed outcome without attributing Base
   generator differences to the robustness penalty alone.

{comparison.to_markdown(index=False)}

Separate hard-category shift diagnostics are saved in
`hard_category_shift_summary.csv`:15 hard categorical interventions per mixed
classifier, excluded from calibration and generated only after outputs freeze.
These perturb categories in training examples, unlike merely hardening the CF.

Single-seed rankings are diagnostic and cannot replace the main multi-run table.
'''
    (out/'summary.md').write_text(text)
    rt=ROOT/'results/exp4_runtime_10cf';rt.mkdir(exist_ok=True);runtime.to_csv(rt/'raw.csv',index=False)
    runtime['seconds_per_factual']=runtime.total_seconds/10;runtime.to_csv(rt/'summary.csv',index=False)
    display=runtime[runtime.method!='wachter'][['dataset','method','device','seconds','total_seconds','FV','valid_cf_count','native_failures','status']]
    (rt/'runtime_table.tex').write_text(display.to_latex(index=False,float_format='%.3f',na_rep='--'))
    (rt/'summary.md').write_text('# Task100 ten-CF runtime\n\nRaw run: `'+str(args.run)+'`. One timing per dataset/method; seed0.\n\nAll methods use the first ten frozen unfavorable test factuals. GRACE uses a fresh M=n=10 solve and independently generated ten-CF Wachter Base. Refinement and Base-inclusive total are separated. CUDA synchronization brackets refinement; shared fitting/calibration/loading are excluded. Baselines here execute on CPU with one thread; GRACE uses its assigned GPU. Dataset workers overlap on separate GPUs and share host resources, so these are observed wall times, not isolated latency or statistically significant speedups. Failures and unsupported configurations are retained; ten attempts do not imply ten valid CFs.\n\n'+display.to_markdown(index=False)+'\n')
    state=yaml.safe_load((ROOT/'state.yaml').read_text());state['status']='complete';state['current_work'].update(status='complete_verified',phase='tabular_verified',experiment_3='complete_diagnostic_with_failures',experiment_4='complete_with_failures',active_experiment_processes=0,completed_tabular_workers=[p.name for p in workers],postprocessing_status='complete_verified',tabular_checks=len(checks),report_exp3=str(out.relative_to(ROOT)/'summary.md'),report_exp4=str(rt.relative_to(ROOT)/'summary.md'));state['next_actions']=[];(ROOT/'state.yaml').write_text(yaml.safe_dump(state,sort_keys=False))
    print('VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
