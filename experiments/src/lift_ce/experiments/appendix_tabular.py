"""Task100 single-seed diagnostic reconstruction; immutable per-worker outputs."""
from __future__ import annotations
import argparse, copy, json, math, os, signal, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from ..data.tabular import load_tabular_three_way
from ..models.tabular import train_model
from ..counterfactuals.mixed_update_aware import MixedDomain, mixed_surrogate
from ..counterfactuals.update_aware import unroll, differentiable_w2
from ..counterfactuals.geodesic_excess import exact_w2
from .german_credit_strict_audit import category_groups, prototypes, solve_wachter
from .heloc_benchmark_milestone import bootstrap_budget
from .heloc_diabetes_official_lift_v1 import DATASETS, roar_one
from ..baselines.official_adapters import generate_ellice_cf

REPO=Path(__file__).resolve().parents[4]
_RNCE_CACHE={}

def save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def dump(obj,path):
    path.write_text(json.dumps(obj,indent=2,allow_nan=True)+'\n')

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def load(path): return torch.load(path,map_location='cpu',weights_only=False)

def score(model,z,futures,source):
    device=next(model.parameters()).device; q=z.to(device)
    with torch.no_grad():
        cv=(model(q).sigmoid()>=.5).cpu()
        predictions=torch.stack([(functional_call(model,(p,b),(q,)).sigmoid()>=.5).cpu() for p,b in futures])
    return dict(current_validity=float(cv.float().mean()),FV=float(predictions.float().mean()),
                Cost=float(torch.linalg.vector_norm(z-source,dim=1).mean()),predictions=predictions,current=cv)

def future_bank(model,x,y,domain,bref,cfg,offset):
    # Numerical shift families retain the old mean/scale definitions. Tail
    # resampling now moves labels with rows (old helper mismatched labels).
    numeric=domain.numeric
    corr=[]
    for j in numeric:
        v=np.corrcoef(x[:,j].cpu().numpy(),y.cpu().numpy())[0,1]
        corr.append(0. if not np.isfinite(v) else abs(float(v)))
    ids=[numeric[j] for j in np.argsort(corr)[-3:][::-1]]
    bank=[]; records=[]; states=[]
    for fi,family in enumerate(cfg['future']['families']):
      for si,severity in enumerate(cfg['future']['severities']):
       for k in range(cfg['future']['realizations']):
        seed=offset+fi*10000+si*100+k; gen=torch.Generator().manual_seed(seed)
        z=x.cpu().clone(); labels=y.cpu().clone(); target=severity*bref
        if family=='mean': z[:,ids]+=target/math.sqrt(len(ids))*torch.where(torch.rand(len(ids),generator=gen)>.5,1.,-1.)
        elif family=='scale':
            v=z[:,ids]; center=v.mean(0); norm=v.sub(center).square().sum(1).mean().sqrt().clamp_min(1e-6)
            z[:,ids]=center+(1+target/norm)*(v-center)
        else:
            v=z[:,ids[0]]; cut=v.median(); lo=(v<=cut).nonzero().flatten(); hi=(v>cut).nonzero().flatten()
            if not len(hi): hi=lo
            nlo=int(len(z)*min(.5+.18*severity,.95))
            index=torch.cat([lo[torch.randint(len(lo),(nlo,),generator=gen)],hi[torch.randint(len(hi),(len(z)-nlo,),generator=gen)]])
            z=z[index]; labels=labels[index]
        params,buffers=unroll(model,z.to(x),labels.to(y),**cfg['update'],differentiable=False)
        params={a:b.detach() for a,b in params.items()}; bank.append((params,buffers))
        states.append(({a:b.cpu() for a,b in params.items()},{a:b.cpu() for a,b in buffers.items()}))
        records.append(dict(family=family,severity=severity,realization=k,seed=seed,numeric_features=ids))
    return bank,states,records

def grace(model,x,y,base,p1,domain,r,beta,cfg,adversary_mode="population",sensitivity_chunk_size=1):
    device=x.device; b=base.to(device); p1=p1.to(device); q=b.clone().requires_grad_(True)
    sync(); start=time.perf_counter()
    D=exact_w2(base,p1.cpu()).distance; budget=beta*D; dual=0.; previous=None; history=[]
    for step in range(cfg['solver']['steps']):
        f,nom,sens=mixed_surrogate(model,x,y,q,r,domain,**cfg['update'],temperature=cfg['smooth_temperature'],cat_temperature=cfg['categorical_temperature'],smoothing=cfg['categorical_smoothing'],adversary_mode=adversary_mode,sensitivity_chunk_size=sensitivity_chunk_size)
        g=differentiable_w2(b,q)[0]+differentiable_w2(q,p1)[0]-budget
        objective=float(f.detach()); change=math.inf if previous is None else abs(objective-previous)/max(1.,abs(previous))
        history.append(dict(step=step,objective=objective,nominal=float(nom.detach()),sensitivity=float(sens.detach()),violation=float(g.detach()),dual=dual,relative_change=change))
        if float(g.detach())<=cfg['solver']['feas_tol']*max(1.,D) and change<=cfg['solver']['obj_tol']: break
        loss=f+dual*g+.5*cfg['solver']['rho']*torch.relu(g).square()
        gradient=torch.autograd.grad(loss,q)[0]
        with torch.no_grad(): qnew=domain.project(q-cfg['solver']['primal_lr']*gradient)
        if not torch.isfinite(qnew).all(): raise RuntimeError('Nonfinite GRACE supports')
        q=qnew.detach().requires_grad_(True)
        newg=exact_w2(base,q.detach().cpu()).distance+exact_w2(q.detach().cpu(),p1.cpu()).distance-budget
        dual=max(0.,dual+cfg['solver']['rho']*newg)
        previous=objective
    z=q.detach().cpu(); plan=exact_w2(base,z); assignment=plan.target_index[plan.source_index.argsort()]; assigned=z[assignment]
    w1=plan.distance; w2=exact_w2(z,p1.cpu()).distance
    sync(); seconds=time.perf_counter()-start
    # Final diagnostics separately from generation runtime.
    f,nom,sens=mixed_surrogate(model,x,y,q,r,domain,**cfg['update'],temperature=cfg['smooth_temperature'],cat_temperature=cfg['categorical_temperature'],smoothing=cfg['categorical_smoothing'],adversary_mode=adversary_mode,sensitivity_chunk_size=sensitivity_chunk_size)
    return dict(q=z,cf=assigned,assignment=assignment,history=history,seconds=seconds,r=r,beta=beta,D=D,budget=budget,W_base_Q=w1,W_Q_P1=w2,feasible=w1+w2<=budget+cfg['solver']['feas_tol']*max(1.,D),nominal=float(nom.detach()),sensitivity=float(sens.detach()),surrogate=float(f.detach()),iterations=len(history))

class SemanticClassifier:
    def __init__(self,model,names,domain): self.model=model;self.names=names;self.domain=domain;self.classes_=np.array([0,1])
    def encode(self,frame):
        frame=pd.DataFrame(frame,columns=self.columns); z=torch.zeros((len(frame),len(self.names)))
        for j in self.domain.numeric: z[:,j]=torch.tensor(frame[self.names[j]].to_numpy(float),dtype=torch.float32)
        for name,ids in self.domain.groups.items():
            k=torch.tensor(frame[name].astype(str).astype(int).to_numpy())
            z[:,ids]=torch.nn.functional.one_hot(k,len(ids)).float()*self.domain.scale.cpu()[ids]+self.domain.low.cpu()[ids]
        return z
    def predict_proba(self,frame):
        with torch.no_grad(): p=self.model(self.encode(frame)).sigmoid().numpy()
        return np.stack([1-p,p],1)
    def predict(self,frame): return (self.predict_proba(frame)[:,1]>=.5).astype(int)
    def frame(self,z):
        values={self.names[j]:z[:,j].numpy().astype('float64') for j in self.domain.numeric}
        for name,ids in self.domain.groups.items(): values[name]=z[:,ids].argmax(1).numpy().astype(str)
        return pd.DataFrame(values)

def generate(method,param,model,d,source,domain,cfg,path,categorical_mode='unsupported'):
    if categorical_mode not in ('unsupported','relaxed','hard'):
        raise ValueError('categorical_mode must be unsupported, relaxed, or hard')
    if path.exists():
        if categorical_mode != 'unsupported': raise FileExistsError(path)
        return load(path)
    model=copy.deepcopy(model).cpu(); names=d['feature_names']; groups=domain.groups
    cpudomain=MixedDomain(d['train_x'],groups); errors=[]; values=[]; times=[]
    if groups and method in ('roar','proplace') and categorical_mode == 'unsupported':
        result=dict(cf=source.clone(),status='unsupported_categorical_constraints',seconds=None,errors=['Canonical adapter lacks mixed categorical domain constraints']*len(source),native_failure=[True]*len(source),device='cpu')
        save(result,path);return result
    generator=None
    if method in ('trex','proplace'):
        from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
        from robustx.lib.tasks.ClassificationTask import ClassificationTask
        from .run_official_proplace import FrozenFrameDataset
        from robustx.generators.robust_CE_methods.STCE import STCE
        from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
        layers=[v for v in model.features if isinstance(v,torch.nn.Linear)]+[model.classifier]
        wrapped=SimpleNNModel(len(names),[v.out_features for v in layers[:-1]],1,seed=0)
        with torch.no_grad():
            for a,b in zip(layers,[v for v in wrapped.model if isinstance(v,torch.nn.Linear)]): b.weight.copy_(a.weight);b.bias.copy_(a.bias)
        wrapped.model.eval(); frame=pd.DataFrame(d['train_x'].numpy(),columns=names);frame['target']=d['train_y'].numpy()
        data=FrozenFrameDataset(seed=0);data.data=frame; task=ClassificationTask(wrapped,data)
        generator=(STCE if method=='trex' else PROPLACE)(task)
        if cfg.get('replication_fixes') and method=='trex':
            from ..baselines.tabular_replication_adapters import PositiveSTCE
            generator=PositiveSTCE(task)
    if method=='dice':
        import dice_ml
        semantic=SemanticClassifier(model,names,cpudomain); frame=semantic.frame(d['train_x']);semantic.columns=list(frame.columns)
        frame['target']=d['train_y'].numpy().astype(int)
        data=dice_ml.Data(dataframe=frame,continuous_features=[names[j] for j in cpudomain.numeric],outcome_name='target')
        generator=dice_ml.Dice(data,dice_ml.Model(model=semantic,backend='sklearn'),method='random')
    sync(); started=time.perf_counter()
    if method=='wachter':
        z,records=solve_wachter(model,source,.5,groups,prototypes(d['train_x'],groups),5,500,90000+cfg.get('seed',0))
        with torch.no_grad(): failed=(model(z).sigmoid()<.5).tolist()
        result=dict(cf=z,status='executed',seconds=time.perf_counter()-started,errors=['no_valid_base' if v else '' for v in failed],native_failure=failed,records=records,device='cpu')
        save(result,path);return result
    def alarm(signum,frame): raise TimeoutError('30 second per-factual generation limit')
    old=signal.signal(signal.SIGALRM,alarm)
    infrastructure=None
    candidates=None
    preparation_seconds=0.
    master_solver_records=[]
    if method=='proplace' and cfg.get('replication_fixes'):
        # Candidate membership is invariant across factuals. Cache it once per
        # fitted model/data/parameter, including across cal and test generation.
        from ..baselines.tabular_replication_adapters import SignedDeltaEvaluator
        key=(id(d),float(param))
        prep_start=time.perf_counter()
        try:
            signal.setitimer(signal.ITIMER_REAL,cfg.get('candidate_preparation_timeout_seconds',1800))
            if key not in _RNCE_CACHE:
                evaluator=SignedDeltaEvaluator(task)
                frame=task.training_data.X
                with torch.no_grad(): nominal_positive=model(d['train_x']).sigmoid().numpy()>=.5
                keep=[bool(nominal_positive[j]) and evaluator.evaluate(row,delta=param,bias_delta=param) for j,(_,row) in enumerate(frame.iterrows())]
                _RNCE_CACHE[key]=frame.loc[keep].copy()
            candidates=_RNCE_CACHE[key]
            if len(candidates)<10: raise RuntimeError('Fewer than 10 native-robust RNCE candidates')
        except Exception as exc:
            infrastructure=f'{type(exc).__name__}: candidate preparation: {exc}'
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            preparation_seconds=time.perf_counter()-prep_start
        print('PROPLACE candidate preparation',preparation_seconds,infrastructure,flush=True)
    try:
      for i,factual in enumerate(source):
        now=time.perf_counter(); error=''
        try:
            if infrastructure: raise RuntimeError(infrastructure)
            signal.setitimer(signal.ITIMER_REAL,cfg['baseline_instance_timeout_seconds'])
            if method in ('trex','proplace'):
                if method=='proplace' and groups and categorical_mode!='unsupported':
                    from ..baselines.mixed_tabular_adapters import mixed_proplace
                    generator=mixed_proplace(task,d['train_x'],groups,hard=categorical_mode=='hard')
                if method=='proplace' and cfg.get('replication_fixes'):
                    from ..baselines.tabular_replication_adapters import corrected_proplace
                    generator=corrected_proplace(task,d['train_x'],groups,hard=categorical_mode=='hard',license_fallback=cfg.get('proplace_license_fallback',False))
                    if cfg.get('proplace_license_fallback'): generator.master_deadline=now+cfg['baseline_instance_timeout_seconds']
                    generator.rnce.getCandidates=lambda *args, **kwargs: candidates
                kwargs={'threshold':param} if method=='trex' else {'delta':param,'bias_delta':param}
                result=generator.generate_for_instance(pd.Series(factual.numpy(),index=names),**kwargs)
                if result.empty: raise ValueError('Empty official result')
                if result.shape[1]!=len(names): raise ValueError('Wrong official result width')
                z=torch.tensor(result.iloc[0].to_numpy(float),dtype=torch.float32)
            elif method=='roar':
                projector=None
                if groups and categorical_mode!='unsupported':
                    from ..baselines.mixed_tabular_adapters import categorical_projector
                    projector=categorical_projector(d['train_x'],groups,hard=categorical_mode=='hard')
                z=roar_one(factual,d,model,i,delta_max=param,project=projector)
            elif method=='ellice':
                lo=cpudomain.low; sc=cpudomain.scale
                result=generate_ellice_cf(((factual-lo)/sc).numpy(),model,((d['train_x']-lo)/sc).numpy(),d['train_y'].numpy(),names,robustness_epsilon=param,one_hot_groups=[[names[j] for j in ids] for ids in groups.values()] or None,model_input_scale=sc,model_input_shift=lo,requires=cfg.get('ellice_requires','valid'))
                if result.empty: raise ValueError('Empty ElliCE result')
                z=torch.tensor(result[names].iloc[0].to_numpy(float),dtype=torch.float32)*sc+lo
            elif method=='dice':
                result=generator.generate_counterfactuals(semantic.frame(factual[None]),total_CFs=cfg['dice']['total_CFs'],desired_class=1,sample_size=cfg['dice']['sample_size'],random_seed=90000+cfg.get('seed',0)*1000+i,posthoc_sparsity_param=0,verbose=False)
                frame=result.cf_examples_list[0].final_cfs_df
                if frame is None or frame.empty: raise ValueError('No DiCE candidate')
                candidates=semantic.encode(frame.drop(columns='target'))
                with torch.no_grad(): candidates=candidates[model(candidates).sigmoid()>=.5]
                if not len(candidates): raise ValueError('No valid DiCE candidate')
                z=candidates[torch.linalg.vector_norm(candidates-factual,dim=1).argmin()]
            else: raise ValueError(method)
            if z.shape!=factual.shape or not torch.isfinite(z).all(): raise ValueError('Malformed counterfactual')
            if cfg.get('replication_fixes'):
                with torch.no_grad(): is_valid=bool(model(z[None]).sigmoid().item()>=.5)
                if not is_valid: error='no_valid_counterfactual'
        except Exception as exc:
            import traceback
            if i==0: traceback.print_exc()
            error=f'{type(exc).__name__}: {exc}';z=factual.clone()
            if type(exc).__name__=='GurobiError': infrastructure=error
        finally: signal.setitimer(signal.ITIMER_REAL,0)
        if method=='proplace' and cfg.get('proplace_license_fallback'): master_solver_records.append(getattr(generator,'master_solver_records',[]))
        values.append(z);errors.append(error);times.append(time.perf_counter()-now)
        if i%16==0: print(method,param,i,len(source),error,flush=True)
    finally: signal.signal(signal.SIGALRM,old)
    sync()
    result=dict(cf=torch.stack(values),status='infrastructure_failure' if infrastructure else 'executed',seconds=time.perf_counter()-started,errors=errors,native_failure=[bool(e) for e in errors],per_factual_seconds=times,device='cpu')
    if cfg.get('replication_fixes'): result['candidate_preparation_seconds']=preparation_seconds
    if cfg.get('proplace_license_fallback'): result['master_solver_records']=master_solver_records
    if categorical_mode != 'unsupported': result['categorical_mode']=categorical_mode
    save(result,path);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--model',choices=['mlp','logistic'],required=True);ap.add_argument('--audit',action='store_true');ap.add_argument('--config',default=str(REPO/'experiments/configs/appendix_tabular.json'));a=ap.parse_args()
    cfg=json.loads(Path(a.config).read_text());out=a.output;torch.set_num_threads(1);torch.manual_seed(cfg.get('seed',0));np.random.seed(cfg.get('seed',0));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.12)
    ds=DATASETS[a.dataset]['config'] if a.dataset in DATASETS else {'source':'openml','openml_id':31,'positive_label':'good'}
    if a.dataset=='fico': ds=json.loads((REPO/'experiments/configs/fico_mixed_task100.json').read_text())
    # FICO audit config may nest the dataset specification.
    ds=ds.get('dataset',ds)
    d=load_tabular_three_way(ds,cfg['data_home'],.6,.2,cfg.get('seed',0));names=d['feature_names']
    groups=category_groups(names) if a.dataset=='german' else ({f:[i for i,n in enumerate(names) if n.startswith(f+'_')] for f in ds.get('categorical_columns',[])} if a.dataset=='fico' else {})
    domain=MixedDomain(d['train_x'],groups)
    hidden=cfg['training']['hidden'] if a.model=='mlp' else []
    epochs=cfg['training']['epochs'] if hidden else max(cfg['training']['epochs'],math.ceil(1000/math.ceil(len(d['train_x'])/256)))
    fitstart=time.perf_counter()
    model,acc=train_model(d['train_x'],d['train_y'],d['cal_x'],d['cal_y'],hidden,epochs,cfg['training']['lr'],cfg.get('seed',0))
    fitsec=time.perf_counter()-fitstart
    save(model.state_dict(),out/'current.pt')
    sources={};source_ids={}
    for split in ('cal','test'):
        with torch.no_grad(): loc=(model(d[split+'_x']).sigmoid()<.5).nonzero().flatten()
        perm=torch.randperm(len(loc),generator=torch.Generator().manual_seed((2 if split=='test' else 3)+1000*cfg.get('seed',0)))
        loc=loc[perm[:cfg['max_factuals']]];sources[split]=d[split+'_x'][loc];source_ids[split]=np.asarray(d[split+'_ids'])[loc.numpy()].tolist()
        if len(loc)<10: raise RuntimeError('Fewer than ten unfavorable factuals')
    with torch.no_grad(): p1=d['train_x'][model(d['train_x']).sigmoid()>=.5]
    bref,_=bootstrap_budget(d['cal_x'],20,min(128,len(d['cal_x'])),4+1000*cfg.get('seed',0))
    save(dict(data=d,sources=sources,source_ids=source_ids,p1=p1,groups=groups,bref=bref,hidden=hidden),out/'prepared.pt')
    dump(dict(config=cfg,dataset_config=ds,seed=cfg.get('seed',0),fit_seconds=fitsec,cal_accuracy=acc,epochs=epochs,gpu=torch.cuda.get_device_name(),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),threads=1,groups=groups,bref=bref),out/'metadata.json')
    model=model.to(device);x=d['train_x'].to(device);y=d['train_y'].to(device)
    if a.audit:
        auditcfg=copy.deepcopy(cfg);auditcfg['solver']['steps']=3
        for method in ['wachter','trex','roar','proplace','ellice']+(['dice'] if a.model=='mlp' or cfg.get('all_baselines') else []):
            param=cfg['baseline_grid'].get(method,[0.])[0]
            obj=generate(method,param,model,d,sources['cal'][:1],domain,cfg,out/f'audit_{method}.pt',categorical_mode=cfg.get('categorical_mode','unsupported'))
            print('AUDIT',method,obj['status'],obj['errors'],flush=True)
        base=load(out/'audit_wachter.pt')['cf']
        obj=grace(model,x,y,base,p1,domain,.25*bref,1.1,auditcfg);save(obj,out/'audit_grace.pt')
        dump({'complete':True},out/'audit_complete.json');return
    calbank,states,records=future_bank(model,x,y,domain,bref,cfg,cfg['future']['cal_seed']);save(dict(states=states,records=records),out/'future_cal.pt')
    selected={};calrows=[];outputs={};timing=[]
    methods=['wachter','trex','roar','proplace','ellice']+(['dice'] if a.model=='mlp' or cfg.get('all_baselines') else [])
    for method in methods:
        print('CALIBRATION',a.dataset,a.model,method,flush=True)
        params=cfg['baseline_grid'].get(method,[0.]); candidates=[]
        for param in params:
            obj=generate(method,param,model,d,sources['cal'],domain,cfg,out/f'cal_{method}_{param:g}.pt',categorical_mode=cfg.get('categorical_mode','unsupported'))
            metric=score(model,obj['cf'],calbank,sources['cal']); row=dict(method=method,param=param,status=obj['status'],**{k:v for k,v in metric.items() if k not in ('predictions','current')});calrows.append(row)
            if cfg.get('replication_fixes'):
                usable=~torch.tensor(obj['native_failure'],dtype=torch.bool)
                row['native_failures']=int((~usable).sum())
                row['selection_FV']=float((metric['predictions']&usable[None]).float().mean())
            if obj['status']=='executed': candidates.append((row.get('selection_FV',row['FV']),-row['Cost'],-param,param))
        choice=max(candidates)[-1] if candidates else params[0];selected[method]=dict(param=choice)
        outputs[method]=generate(method,choice,model,d,sources['test'],domain,cfg,out/f'test_{method}.pt',categorical_mode=cfg.get('categorical_mode','unsupported'))
        if method in ('wachter','dice'):
            basecal=load(out/f'cal_{method}_0.pt')['cf']; feasible=[]
            for beta in cfg['beta_grid']:
              for ratio in cfg['r_ratios']:
                print('GRACE calibration',method,beta,ratio,flush=True)
                obj=grace(model,x,y,basecal,p1,domain,ratio*bref,beta,cfg);save(obj,out/f'cal_grace_{method}_b{beta:g}_r{ratio:g}.pt')
                metric=score(model,obj['cf'],calbank,sources['cal']); row=dict(method='grace_'+method,r_ratio=ratio,beta=beta,status='feasible' if obj['feasible'] else 'infeasible',**{k:v for k,v in metric.items() if k not in ('predictions','current')});calrows.append(row)
                if obj['feasible']: feasible.append((row['FV'],-row['Cost'],-ratio,-beta,ratio,beta))
            if feasible:
                *_,ratio,beta=max(feasible);selected['grace_'+method]=dict(r_ratio=ratio,beta=beta)
                obj=grace(model,x,y,outputs[method]['cf'],p1,domain,ratio*bref,beta,cfg);obj['status']='executed' if obj['feasible'] else 'infeasible';obj['native_failure']=[not obj['feasible']]*len(sources['test']);save(obj,out/f'test_grace_{method}.pt');outputs['grace_'+method]=obj
            else: selected['grace_'+method]=dict(status='no_feasible_calibration')
        pd.DataFrame(calrows).to_csv(out/'calibration.csv',index=False);dump(selected,out/'selection.json')
    # Freeze all outputs and selection before constructing held-out future updates.
    dump(dict(time=time.time(),methods=list(outputs)),out/'frozen.json')
    testbank,states,records=future_bank(model,x,y,domain,bref,cfg,cfg['future']['test_seed']);save(dict(states=states,records=records),out/'future_test.pt')
    rows=[]
    for method,obj in outputs.items():
        for realization in (['relaxed','hard'] if groups and (method.startswith('grace') or (cfg.get('categorical_mode')=='relaxed' and method in ('roar','proplace'))) else ['native']):
            z=domain.project(obj['cf'],True) if realization=='hard' else obj['cf']
            metric=score(model,z,testbank,sources['test']);save(metric,out/f'scores_{method}_{realization}.pt')
            row=dict(seed=cfg.get('seed',0),dataset=a.dataset,classifier=a.model,method=method,realization=realization,n=len(z),status=obj['status'],native_failures=sum(obj.get('native_failure',[])),**{k:v for k,v in metric.items() if k not in ('predictions','current')})
            if method.startswith('grace'):
                base=outputs[method.removeprefix('grace_')]['cf'];detour=exact_w2(base,z).distance+exact_w2(z,p1).distance
                row.update(geometry_feasible=detour<=obj['budget']+cfg['solver']['feas_tol']*max(1,obj['D']),detour=detour,budget=obj['budget'],nominal=obj['nominal'],sensitivity=obj['sensitivity'])
            hard_error=(domain.project(z,True)-z).abs().amax(1)>1e-5 if groups else torch.zeros(len(z),dtype=torch.bool)
            row['hard_category_violations']=int(hard_error.sum())
            failures=torch.tensor(obj.get('native_failure',[False]*len(z)),dtype=torch.bool)
            usable=(~failures)&(~hard_error if realization!='relaxed' else torch.ones(len(z),dtype=torch.bool))
            row['failure_aware_FV']=float((metric['predictions']&usable[None]).float().mean()) if obj['status']=='executed' else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    if a.model=='mlp' and cfg.get('run_timing',True):
        source=sources['test'][:10];base=None
        for method in ['wachter','trex','roar','proplace','ellice','grace_wachter']:
            print('TIMING',method,flush=True)
            if method=='grace_wachter':
                choice=selected[method]
                if 'beta' not in choice:
                    timing.append(dict(method=method,status='no_feasible_calibration',n=10));continue
                obj=grace(model,x,y,base['cf'],p1,domain,choice['r_ratio']*bref,choice['beta'],cfg);obj['status']='executed' if obj['feasible'] else 'infeasible';obj['native_failure']=[not obj['feasible']]*10;save(obj,out/'timed_grace_wachter.pt')
            else:
                obj=generate(method,selected[method]['param'],model,d,source,domain,cfg,out/f'timed_{method}.pt',categorical_mode=cfg.get('categorical_mode','unsupported'))
                if method=='wachter':base=obj
            metric=score(model,obj['cf'],testbank,source)
            timing.append(dict(dataset=a.dataset,method=method,status=obj['status'],n=10,seconds=obj['seconds'],total_seconds=(obj['seconds']+base['seconds'] if method=='grace_wachter' else obj['seconds']),native_failures=sum(obj.get('native_failure',[])),device='cuda' if method=='grace_wachter' else 'cpu',**{k:v for k,v in metric.items() if k not in ('predictions','current')}))
        pd.DataFrame(timing).to_csv(out/'runtime.csv',index=False)
    dump(dict(completed=time.time(),rows=len(rows)),out/'complete.json')

if __name__=='__main__':main()
