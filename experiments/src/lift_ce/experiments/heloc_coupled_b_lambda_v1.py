"""HELOC coupled data-uncertainty / LIFT-depth calibration experiment.

The external future benchmark and every baseline tensor are loaded frozen.
Only the training-derived proxy construction and coupled calibration are new.
"""
from __future__ import annotations

import argparse, copy, json, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor

from .corrected_future_benchmark_v1 import load_model
from .geo_uniform_lift_ablation_v1 import geometry
from .heloc_benchmark_milestone import bootstrap_budget, valid
from .heloc_debug_diagnostics import w2_sinkhorn
from .path_lift_decision_compatible_ot_v1 import pair_violations, score
from .constrained_lift_ablation_v1 import exact_constrained_plan
from .stability_weighted_anchor_v1 import weighted_sinkhorn
from ..counterfactuals.tabular import generate_wachter_ce
from ..data.tabular import load_tabular_three_way

TAU=.5; K_GEO=10; M_PRIMARY=5; V_TARGET=.95
LAMS=np.round(np.linspace(0,1,101),2)

def fine_tune(current,x,y,seed,epochs=8,lr=2e-4,wd=1e-4):
    torch.manual_seed(seed); model=copy.deepcopy(current); model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    for ep in range(epochs):
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+ep))
        for a in range(0,len(x),256):
            q=order[a:a+256]; loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[q]),y[q])
            opt.zero_grad();loss.backward();opt.step()
    model.eval();return model

def adverse_directions(current,coarse,b,m,seed,previous):
    """Diverse coarse-path fragility-gradient directions, exactly RMS projected.

    This is a deliberately low-rank empirical perturbation family: every row
    receives the same feature direction. It satisfies mean_i ||Delta_i||^2=b^2.
    """
    g=torch.Generator().manual_seed(seed); dirs=[]
    for member in range(m):
        z=coarse.clone().requires_grad_(True); p=torch.sigmoid(current(z))
        weights=.5+torch.rand(len(z),generator=g)
        objective=(weights*torch.relu(TAU-p)).mean()-.05*(weights*p).mean()
        grad=torch.autograd.grad(objective,z)[0]
        v=grad.mean(0)+.15*torch.randn(grad.shape[1],generator=g)*grad.std(0).clamp_min(1e-6)
        for old in dirs+previous:
            v=v-(v@old)*old
        if v.norm()<1e-10:v=torch.randn(v.shape,generator=g)
        v=v/v.norm();dirs.append(v.detach())
    return [b*v for v in dirs],dirs

def build_geometry(current,base,fav):
    eu=torch.cdist(base,fav);_,_,_,mmin=pair_violations(current,base,fav)
    straight=(mmin>=-1e-10).numpy(); geo,hops,connected,audit=geometry(current,base,fav,straight,eu.numpy(),K_GEO)
    finite=geo[connected];penalty=(finite.max()*10 if finite.size else float(eu.max()*100));numeric=geo.copy();numeric[~connected]=penalty
    w=np.full(len(fav),1/len(fav));pg,eta,_,_=exact_constrained_plan(numeric**2,connected,w)
    pu=weighted_sinkhorn(eu.square(),torch.tensor(w,dtype=eu.dtype),iterations=1000,tol=1e-8).numpy()
    dest={"Uniform-LIFT":torch.tensor(pu/pu.sum(1,keepdims=True),dtype=fav.dtype)@fav,
          "Geo-LIFT":torch.tensor(pg/pg.sum(1,keepdims=True),dtype=fav.dtype)@fav}
    audit.update({'connected_pair_rate':float(connected.mean()),'eta_geo_star':eta,
                  'dominant_target_change_rate':float((pu.argmax(1)!=pg.argmax(1)).mean()),
                  'coupling_l1_difference':float(np.abs(pu-pg).sum())})
    # Coupling-weighted intrinsic path lengths are used only when a realized
    # straight refinement segment leaves F0.
    cond_u=pu/pu.sum(1,keepdims=True);cond_g=pg/pg.sum(1,keepdims=True)
    geo_num=np.where(connected,geo,penalty)
    fallback={'Uniform-LIFT':torch.tensor((cond_u*geo_num).sum(1),dtype=base.dtype),
              'Geo-LIFT':torch.tensor((cond_g*geo_num).sum(1),dtype=base.dtype)}
    return dest,fallback,audit

def path_costs(current,source,base,dest,fallback):
    basecost=torch.linalg.vector_norm(base-source,dim=1);rows=[]
    for lam in LAMS:
        z=(1-lam)*base+lam*dest; grid=torch.linspace(0,1,21)
        points=((1-grid[:,None,None])*base[None]+grid[:,None,None]*z[None]).flatten(0,1)
        direct=valid(current,points).view(21,len(base)).all(0)
        d=torch.linalg.vector_norm(z-base,dim=1); intrinsic=torch.where(direct,d,lam*fallback)
        ext=torch.linalg.vector_norm(z-source,dim=1)
        rows.append({'lambda':lam,'C_int':float((basecost+intrinsic).mean()),'C_L2':float(ext.mean()),
                     'intrinsic_fallback_fraction':float((~direct).float().mean())})
    return pd.DataFrame(rows)

def proxy_profile(method,base,dest,models,costs,b,bref):
    rows=[]
    for lam in LAMS:
        z=(1-lam)*base+lam*dest;vs=[];risks=[]
        for model in models:
            p=score(model,z);vs.append(float((p>=TAU).float().mean()));risks.append(float(torch.relu(TAU-p).mean()))
        c=costs[costs['lambda']==lam].iloc[0]
        rows.append({'method':method,'b':b,'b_over_b_ref':b/bref,'lambda':lam,
                     'proxy_worst_validity':min(vs),'proxy_invalidity':max(risks),
                     'C_int':c.C_int,'C_L2':c.C_L2,'intrinsic_fallback_fraction':c.intrinsic_fallback_fraction})
    return pd.DataFrame(rows)

def select_depth_legacy(profile):
    """Legacy 95%-threshold selector retained only for post-hoc audits."""
    feasible=profile[profile.proxy_worst_validity>=V_TARGET]
    return (feasible.sort_values(['C_int','lambda']).iloc[0] if len(feasible)
            else profile.sort_values(['proxy_worst_validity','C_int'],ascending=[False,True]).iloc[0])

def pareto_tradeoff_profile(profile,eps=1e-12):
    """Annotate the cost/margin-invalidity Pareto frontier per fixed b."""
    g=profile.copy().sort_values('lambda').reset_index(drop=True)
    cost=g.C_int.to_numpy(float);risk=g.proxy_invalidity.to_numpy(float);keep=np.ones(len(g),dtype=bool)
    for i in range(len(g)):
        dominated=((cost<=cost[i]+eps)&(risk<=risk[i]+eps)&
                   ((cost<cost[i]-eps)|(risk<risk[i]-eps))).any()
        keep[i]=not dominated
    g['pareto_efficient']=keep
    p=g.loc[keep];cmin,cmax=float(p.C_int.min()),float(p.C_int.max());rmin,rmax=float(p.proxy_invalidity.min()),float(p.proxy_invalidity.max())
    g['C_norm']=(g.C_int-cmin)/max(cmax-cmin,eps)
    g['R_norm']=(g.proxy_invalidity-rmin)/max(rmax-rmin,eps)
    g['ideal_score']=np.sqrt(g.C_norm**2+g.R_norm**2)
    g['equal_weight_score']=g.C_norm+g.R_norm
    return g

def select_depth_ideal(profile,tie_tol=1e-10):
    """Pareto-filtered normalized ideal-point diagnostic selector."""
    g=pareto_tradeoff_profile(profile);p=g[g.pareto_efficient]
    best=float(p.ideal_score.min());ties=p[p.ideal_score<=best+tie_tol]
    return ties.sort_values(['C_int','lambda']).iloc[0]

def select_depth(profile,tie_tol=1e-10):
    """Normalized Pareto chord-knee selector (the deployed selector)."""
    g=pareto_tradeoff_profile(profile);p=g[g.pareto_efficient].sort_values('C_norm')
    if len(p)<3:return p.sort_values(['C_int','lambda']).iloc[0]
    x=p.C_norm.to_numpy(float);y=p.R_norm.to_numpy(float)
    x1,y1=x[0],y[0];x2,y2=x[-1],y[-1]
    den=max(float(np.hypot(y2-y1,x2-x1)),1e-12)
    distance=np.abs((y2-y1)*x-(x2-x1)*y+x2*y1-y2*x1)/den
    best=float(distance.max());ties=p[distance>=best-tie_tol]
    return ties.sort_values(['C_int','lambda']).iloc[0]

def evaluate_path(method,source,base,dest,current,futures,fav,lof,fallback):
    costs=path_costs(current,source,base,dest,fallback);rows=[]
    for lam in LAMS:
        z=(1-lam)*base+lam*dest; vals=[];meta=[]
        for env,sev,k,rg,fm in futures:
            v=float(valid(fm,z).float().mean());vals.append(v);meta.append((env,rg,v))
        near=[v for _,rg,v in meta if rg in ('in_range','near_range')];l2=torch.linalg.vector_norm(z-source,dim=1)
        row={'method':method,'lambda':lam,'current_validity':float(valid(current,z).float().mean()),
             'mean_future_validity':float(np.mean(vals)),'near_in_range_worst':float(np.min(near)),
             'generic_worst':min(v for e,_,v in meta if e=='generic_population_shift'),
             'target_side_worst':min(v for e,_,v in meta if e=='target_side_shift'),
             'mixed_worst':min(v for e,_,v in meta if e=='mixed_shift'),
             'mean_L2_cost':float(l2.mean()),'median_L2_cost':float(l2.median()),'p90_L2_cost':float(torch.quantile(l2,.9)),
             'NN_target_distance':float(torch.cdist(z,fav).min(1).values.mean()),'LOF':float(lof.score_samples(z.numpy()).mean()),
             'population_target_distance':w2_sinkhorn(z,fav)}
        row.update(costs[costs['lambda']==lam].iloc[0][['C_int','intrinsic_fallback_fraction']].to_dict());rows.append(row)
    return pd.DataFrame(rows)

def baseline_frontier(front):
    out=front[front.method.isin(['base','highconf','trex'])].copy();out['method']=out.method.map({'base':'Base-Wachter','highconf':'HighConf','trex':'T-Rex'})
    return out.rename(columns={'native_parameter':'lambda','mean_l2_cost':'mean_L2_cost','mean_nn_target_distance':'NN_target_distance','mean_lof':'LOF','common_mean_future_validity':'mean_future_validity','common_near_range_worst':'near_in_range_worst'})

def matched_cost(paths,baselines):
    pool=pd.concat([paths,baselines],ignore_index=True,sort=False);methods=['Geo-LIFT','Uniform-LIFT','T-Rex','HighConf','Base-Wachter'];swept=[m for m in methods if m!='Base-Wachter']
    lo=max(pool[pool.method==m].mean_L2_cost.min() for m in swept);hi=min(pool[pool.method==m].mean_L2_cost.max() for m in swept);rows=[]
    for budget in np.linspace(lo,hi,20) if hi>=lo else []:
        for method in methods:
            q=pool[(pool.method==method)&(pool.mean_L2_cost<=budget+1e-9)]
            if q.empty:continue
            r=q.sort_values(['near_in_range_worst','mean_L2_cost'],ascending=[False,True]).iloc[0]
            rows.append({'cost_budget':budget,'method':method,'actual_L2_cost':r.mean_L2_cost,'lambda':r.get('lambda',np.nan),
                         **{c:r.get(c,np.nan) for c in ('current_validity','mean_future_validity','near_in_range_worst','NN_target_distance','LOF','population_target_distance')}})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-home',required=True);ap.add_argument('--current-root',required=True);ap.add_argument('--future-root',required=True);ap.add_argument('--frontier-root',required=True);ap.add_argument('--geo-root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--epochs',type=int,default=8);a=ap.parse_args()
    start=time.perf_counter();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    cfg={'source':'openml','openml_name':'heloc','openml_version':1,'missing_values':[-7,-8,-9]};d=load_tabular_three_way(cfg,a.data_home,.6,.2,0);tx,ty,cx,cy,ex,ey=(d[k] for k in ('train_x','train_y','cal_x','cal_y','test_x','test_y'))
    current=load_model(Path(a.current_root)/'models/current.pt',tx.shape[1]);cf=torch.load(Path(a.current_root)/'counterfactuals.pt',map_location='cpu',weights_only=True);source,source_ids,base=cf['sources'],cf['source_ids'].numpy(),cf['base_cf'];assert len(base)==128
    fav=tx[valid(current,tx)];dest,fallback,geo_audit=build_geometry(current,base,fav);costs={m:path_costs(current,source,base,z,fallback[m]) for m,z in dest.items()}
    # Reuse the established, validated b_cal scale. Held-out calibration is
    # reserved for choosing among the coupled configurations below.
    bref=float(json.loads((Path(a.current_root)/'metadata.json').read_text())['b_cal']);multipliers=(.5,.75,1.,1.25,1.5);B=[bref*q for q in multipliers]
    coarse=torch.cat([((1-l)*base+l*dest['Geo-LIFT']) for l in (0,.25,.5,.75,1.)]);proxy_rows=[];profiles=[];models_by_b={};prev=[];tproxy=time.perf_counter()
    for bi,b in enumerate(B):
        vectors,unit=adverse_directions(current,coarse,b,M_PRIMARY,9300+bi,prev);prev=unit[:1];models=[]
        for m,v in enumerate(vectors):
            shifted=tx+v;realized=float(torch.linalg.vector_norm(shifted-tx,dim=1).square().mean().sqrt());ts=time.perf_counter();model=fine_tune(current,shifted,ty,9400+100*bi+m,a.epochs);runtime=time.perf_counter()-ts;models.append(model);torch.save(model.state_dict(),out/'models'/f'b{bi}_m{m}.pt')
            p=score(model,coarse);proxy_rows.append({'b_index':bi,'b':b,'b_over_b_ref':b/bref,'proxy_model_id':m,'realized_budget':realized,'aggregate_coarse_fragility':float(torch.relu(TAU-p).mean()),'coarse_validity':float((p>=TAU).float().mean()),'fine_tune_runtime_seconds':runtime,'epochs':a.epochs,'learning_rate':2e-4,'weight_decay':1e-4,'initialization':'f0','direction_type':'diverse coarse-path fragility gradient'})
        models_by_b[bi]=models
        for method in ('Geo-LIFT','Uniform-LIFT'):profiles.append(proxy_profile(method,base,dest[method],models,costs[method],b,bref).assign(b_index=bi))
    proxy_seconds=time.perf_counter()-tproxy;profile=pd.concat(profiles,ignore_index=True);profile.to_csv(out/'heloc_b_lambda_profile.csv',index=False);pd.DataFrame(proxy_rows).to_csv(out/'heloc_proxy_models.csv',index=False)
    # Training-derived lambda_b for each geometry.
    configs=[]
    for (method,bi),g in profile.groupby(['method','b_index']):
        r=select_depth(g);configs.append({'method':method,'b_index':bi,'b':r.b,'b_over_b_ref':r.b_over_b_ref,'lambda_b':r['lambda'],'worst_proxy_validity':r.proxy_worst_validity,'proxy_invalidity':r.proxy_invalidity,'C_int':r.C_int,'C_L2':r.C_L2,'v_target':V_TARGET,'feasible_target':bool(r.proxy_worst_validity>=V_TARGET)})
    configs=pd.DataFrame(configs)
    # Fresh held-out calibration CFs and method-specific geometry, never used for proxy fitting.
    neg=(score(current,cx)<TAU).nonzero().flatten();cal_source=cx[neg[:min(128,len(neg))]];cal_base=generate_wachter_ce(current,cal_source,.5,300,.03,.05,20.)
    cdest,cfallback,_=build_geometry(current,cal_base,fav);cal_rows=[]
    for r in configs.itertuples():
        z=(1-r.lambda_b)*cal_base+r.lambda_b*cdest[r.method];vals=[];loss=[]
        for model in models_by_b[r.b_index]:p=score(model,z);vals.append(float((p>=TAU).float().mean()));loss.append(float(torch.relu(TAU-p).mean()))
        cc=path_costs(current,cal_source,cal_base,cdest[r.method],cfallback[r.method]);ci=float(cc[cc['lambda']==r.lambda_b].C_int.iloc[0]);cal_rows.append({'method':r.method,'b_index':r.b_index,'b':r.b,'b_over_b_ref':r.b_over_b_ref,'lambda_b':r.lambda_b,'V_cal':min(vals),'L_cal':max(loss),'calibration_C_int':ci,'calibration_sources':len(cal_source)})
    cal=pd.DataFrame(cal_rows);selected=[]
    for method,g in cal.groupby('method'):
        r=g.sort_values(['V_cal','calibration_C_int'],ascending=[False,True]).iloc[0];selected.append(r)
    selected=pd.DataFrame(selected);cal=cal.merge(selected[['method','b_index']].assign(selected_b=1),on=['method','b_index'],how='left');cal.selected_b=cal.selected_b.fillna(0).astype(int);cal.to_csv(out/'heloc_calibration_selection.csv',index=False)
    configs=configs.merge(cal[['method','b_index','V_cal','L_cal','selected_b']],on=['method','b_index']);configs.to_csv(out/'heloc_coupled_configs.csv',index=False)
    # Frozen full HELOC benchmark: generic, target-side, and mixed families.
    future_root=Path(a.future_root);sanity=pd.read_csv(future_root/'shift_scale_audit.csv');futures=[]
    for r in sanity.itertuples():futures.append((r.environment,float(r.severity),int(r.future_model_id),r.range_group,load_model(future_root/'models'/f'{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt',tx.shape[1])))
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(fav.numpy());paths=[]
    for method in ('Geo-LIFT','Uniform-LIFT'):paths.append(evaluate_path(method,source,base,dest[method],current,futures,fav,lof,fallback[method]))
    paths=pd.concat(paths,ignore_index=True);front=pd.read_csv(Path(a.frontier_root)/'common_candidate_frontier.csv');baselines=baseline_frontier(front)
    finals=[]
    for r in selected.itertuples():
        q=paths[(paths.method==r.method)&(paths['lambda']==r.lambda_b)].iloc[0].to_dict();q.update({'selection':'coupled','b_star':r.b,'b_over_b_ref':r.b_over_b_ref});finals.append(q)
    # Frozen previous direct-lambda calibration for the ablation.
    old=pd.read_csv(Path(a.geo_root)/'geo_uniform_results.csv');old=old[(old.setting=='heloc__baseline_mlp')&(old.calibrated_flag==1)&old.method.isin(['Geo-LIFT','Uniform-LIFT'])]
    for _,r in old.iterrows():
        q=paths[(paths.method==r.method)&(paths['lambda']==round(float(r['lambda']),2))].iloc[0].to_dict();q.update({'selection':'previous_direct_lambda','b_star':np.nan,'b_over_b_ref':np.nan});finals.append(q)
    # Existing selected baselines, with corrected external results and frozen plausibility.
    for method in ('Base-Wachter','HighConf','T-Rex'):
        q=baselines[(baselines.method==method)&(baselines.calibrated_flag==1)];q=q.iloc[0] if len(q) else baselines[baselines.method==method].iloc[-1];finals.append({'method':method,'selection':'existing_calibration',**q.to_dict()})
    final=pd.DataFrame(finals);final.to_csv(out/'heloc_final_results.csv',index=False)
    matched=matched_cost(paths,baselines);matched.to_csv(out/'heloc_matched_cost_trex.csv',index=False)
    ab=final[final.method.isin(['Geo-LIFT','Uniform-LIFT'])].copy();ab.to_csv(out/'heloc_coupled_ablation.csv',index=False)
    runtime=pd.DataFrame([{'stage':'proxy_model_construction','runs':len(B)*M_PRIMARY,'seconds':proxy_seconds,'scaling':'|B| x M'}, {'stage':'101-lambda_sweep_all_b_methods','runs':len(B)*2*len(LAMS),'seconds':np.nan,'scaling':'cheap inference'}, {'stage':'total_coupled_calibration','runs':len(B)*M_PRIMARY,'seconds':time.perf_counter()-start,'scaling':'|B| x M, not |B| x |Lambda| x M'}, {'stage':'hypothetical_per_b_lambda','runs':len(B)*len(LAMS)*M_PRIMARY,'seconds':np.nan,'scaling':'|B| x |Lambda| x M'}]);runtime.to_csv(out/'heloc_runtime.csv',index=False)
    # Central figures.
    for y,name in [('lambda_b','b_to_lambda'),('C_int','b_to_intrinsic_cost'),('worst_proxy_validity','b_to_proxy_robustness')]:
        fig,ax=plt.subplots(figsize=(5.5,3.6))
        for method,g in configs.groupby('method'):ax.plot(g.b_over_b_ref,g[y],marker='o',label=method)
        ax.set(xlabel='b / b_ref',ylabel=y);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=180);plt.close(fig)
    torch.save({'source_ids':torch.as_tensor(source_ids),'source':source,'base':base,'favorable':fav,'destinations':dest},out/'artifacts.pt')
    meta={'b_ref':bref,'B':B,'M':M_PRIMARY,'lambda_grid':'0.00:0.01:1.00','v_target':V_TARGET,'k_geo':K_GEO,'fine_tune':{'initialization':'f0','optimizer':'AdamW','lr':2e-4,'epochs':a.epochs,'weight_decay':1e-4},'future_test_used_for_tuning':False,'geo_audit':geo_audit,'immutable_policy':'existing HELOC benchmark has no non-perturbable-coordinate mask; all standardized coordinates are perturbed'};(out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(final[['method','selection','current_validity','mean_future_validity','near_in_range_worst','mean_L2_cost']].to_string(index=False));print(selected.to_string(index=False))

if __name__=='__main__':main()
