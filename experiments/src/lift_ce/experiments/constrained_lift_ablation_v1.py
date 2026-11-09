"""Current-classifier admissible-coupling ablation for LIFT."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from sklearn.neighbors import LocalOutlierFactor

from .path_lift_decision_compatible_ot_v1 import (
    setting_artifacts, pair_violations, realized_metrics, score,
)
from .stability_weighted_anchor_v1 import weighted_sinkhorn
from .heloc_benchmark_milestone import valid
from .heloc_debug_diagnostics import w2_sinkhorn, markdown_table


def transport_constraints(n: int, m: int):
    cols=np.arange(n*m)
    rows=np.concatenate((np.repeat(np.arange(n),m),n+np.tile(np.arange(m),n)))
    cc=np.concatenate((cols,cols)); data=np.ones(2*n*m)
    return coo_matrix((data,(rows,cc)),shape=(n+m,n*m)).tocsr()


def masked_weighted_sinkhorn(cost: np.ndarray, target: np.ndarray, mask: np.ndarray,
                             reg: float=.1, iterations: int=5000, tol: float=1e-9):
    """The original entropic OT solver with an exact coupling-support mask."""
    c=torch.as_tensor(cost,dtype=torch.float64);allow=torch.as_tensor(mask,dtype=torch.bool)
    n=len(c);a=torch.full((n,),1/n,dtype=c.dtype);b=torch.as_tensor(target,dtype=c.dtype);b=b/b.sum()
    positive=c[c>0];scale=positive.median() if positive.numel() else c.new_tensor(1.)
    logk=-c/scale.clamp_min(1e-12)/reg;logk=logk.masked_fill(~allow,-torch.inf)
    lu=torch.zeros_like(a);lv=torch.zeros_like(b)
    for step in range(iterations):
        lu=a.log()-torch.logsumexp(logk+lv[None],dim=1)
        lv=b.log()-torch.logsumexp(logk+lu[:,None],dim=0)
        if step%20==19:
            q=torch.exp(lu[:,None]+logk+lv[None])
            if max((q.sum(1)-a).abs().max(),(q.sum(0)-b).abs().max())<tol:break
    q=torch.exp(lu[:,None]+logk+lv[None]);q=q/q.sum()
    return q.numpy()


def exact_constrained_plan(cost: np.ndarray, admissible: np.ndarray,
                           target: np.ndarray):
    """Lexicographically minimize inadmissible mass and then Euclidean cost."""
    n,m=cost.shape; src=np.full(n,1/n,dtype=np.float64);target=np.asarray(target,dtype=np.float64);target=target/target.sum();b=np.r_[src,target]
    Aeq=transport_constraints(n,m)
    bad=(~admissible).astype(float).ravel()
    first=linprog(bad,A_eq=Aeq,b_eq=b,bounds=(0,None),method='highs')
    if not first.success:
        raise RuntimeError(f'minimum-slack transport failed: {first.message}')
    eta=float(first.fun)
    if eta>=1-1e-10:
        plan=masked_weighted_sinkhorn(cost,target,np.ones_like(admissible,dtype=bool))
        return plan,eta,first,None
    slack_edges=first.x.reshape(n,m)>1e-12
    support=admissible if eta<=1e-10 else (admissible|slack_edges)
    plan=masked_weighted_sinkhorn(cost,target,support)
    realized_eta=float(plan[~admissible].sum())
    if abs(realized_eta-eta)>max(1e-4,eta*1e-4):
        raise RuntimeError(f'masked Sinkhorn changed minimum slack: LP={eta}, plan={realized_eta}')
    return plan,eta,first,None


def evaluate_destination(ds,setting,source,base,fav,current,proxies,futures,
                         destination,method,eta,admissible_fraction):
    mech=realized_metrics(current,base,destination); rows=[]
    lof=LocalOutlierFactor(n_neighbors=min(20,len(fav)-1),novelty=True).fit(fav.numpy())
    for lam in np.linspace(0,1,101):
        lam=round(float(lam),2); z=(1-lam)*base+lam*destination
        risks=[]; pvalid=[]
        for proxy in proxies:
            p=score(proxy,z); risks.append(float(torch.relu(.5-p).mean()));pvalid.append(float((p>=.5).float().mean()))
        ev=[]
        for env,sev,k,rg,fm in futures:
            ev.append({'environment':env,'range_group':rg,'validity':float(valid(fm,z).float().mean())})
        ev=pd.DataFrame(ev); meaningful=ev[ev.range_group.isin(['in_range','near_range'])]
        l2=torch.linalg.vector_norm(z-source,dim=1); nn=torch.cdist(z,fav).min(1).values
        row={'setting':f'{ds}__{setting}','dataset':ds,'classifier_setting':setting,
             'method':method,'lambda':lam,'current_validity':float(valid(current,z).float().mean()),
             'calibration_risk':max(risks),'proxy_worst_validity':min(pvalid),
             'mean_future_validity':float(ev.validity.mean()),
             'near_in_range_worst':float(meaningful.validity.min()),
             'mean_L2_cost':float(l2.mean()),'median_L2_cost':float(l2.median()),
             'p90_L2_cost':float(torch.quantile(l2,.9)),'NN_target_distance':float(nn.mean()),
             'LOF':float(lof.score_samples(z.numpy()).mean()),
             'population_target_distance':w2_sinkhorn(z,fav),'eta_star':eta,
             'pairwise_admissible_fraction':admissible_fraction,**mech}
        for env in ev.environment.unique(): row[f'worst__{env}']=float(ev[ev.environment==env].validity.min())
        tq=meaningful[meaningful.environment=='target_side_shift']
        row['target_side_near_range_worst']=float(tq.validity.min()) if len(tq) else np.nan
        rows.append(row)
    out=pd.DataFrame(rows);g=out.sort_values('lambda');r0,r1=g.iloc[0].calibration_risk,g.iloc[-1].calibration_risk
    gain=(r0-g.calibration_risk)/max(r0-r1,1e-12)-g['lambda'];out['calibrated_flag']=0;out.loc[gain.idxmax(),'calibrated_flag']=1
    return out


def run_setting(ds,setting,home,roots,prior):
    d,source,source_ids,base,current,proxies,futures,front,tensors,trex=setting_artifacts(ds,setting,home,roots)
    fav=d['train_x'][valid(current,d['train_x'])];fs=score(current,fav);raw=torch.minimum(torch.relu(fs-.5),torch.tensor(.25))+1e-6;w=(raw/raw.sum()).numpy().astype(np.float64);w=w/w.sum()
    eu=torch.cdist(base,fav).square();vint,vmax,vfrac,mmin=pair_violations(current,base,fav);adm=(mmin>=-1e-10).numpy();label=f'{ds}__{setting}'
    plan,eta,first,second=exact_constrained_plan(eu.numpy(),adm,w)
    p=torch.tensor(plan,dtype=base.dtype);dest=(p/p.sum(1,keepdim=True).clamp_min(1e-30))@fav
    margin=weighted_sinkhorn(eu,torch.tensor(w),iterations=1000,tol=1e-8);dm=margin.argmax(1).numpy();dc=plan.argmax(1)
    # Full pair table is intentionally retained: it is the support definition.
    ii,jj=np.indices(adm.shape)
    pairs=pd.DataFrame({'setting':label,'source_id':np.asarray(source_ids)[ii.ravel()],
        'target_id':jj.ravel(),'euclidean_cost':eu.numpy().ravel(),
        'min_path_margin':mmin.numpy().ravel(),'admissible':adm.ravel()})
    row_adm=adm.sum(1);col_adm=adm.sum(0)
    feasibility={'setting':label,'n_sources':len(base),'n_targets':len(fav),
        'admissible_edge_fraction':float(adm.mean()),'mean_admissible_targets_per_source':float(row_adm.mean()),
        'median_admissible_targets_per_source':float(np.median(row_adm)),
        'fraction_sources_zero_admissible':float((row_adm==0).mean()),
        'fraction_targets_zero_incoming':float((col_adm==0).mean()),'strict_feasible':bool(eta<=1e-9),'eta_star':eta}
    coupling=pd.DataFrame({'setting':label,'source_id':np.asarray(source_ids)[ii.ravel()],
        'target_id':jj.ravel(),'transported_mass':plan.ravel(),'admissible':adm.ravel(),
        'euclidean_cost':eu.numpy().ravel()})
    coupling=coupling[coupling.transported_mass>1e-14].reset_index(drop=True)
    coupling['inadmissible_transported_mass']=coupling.transported_mass*(~coupling.admissible)
    coupling_l1=float(np.abs(plan-margin.numpy()).sum());changed=float((dm!=dc).mean())
    entropy=float(np.mean(-(plan*np.log(np.maximum(plan,1e-30))).sum(1)))
    feasibility.update({'total_admissible_mass':float(plan[adm].sum()),'total_inadmissible_mass':float(plan[~adm].sum()),
        'margin_lift_inadmissible_mass':float(margin.numpy()[~adm].sum()),
        'dominant_target_change_fraction':changed,'coupling_L1_vs_Margin':coupling_l1,'mean_source_entropy':entropy,
        'max_source_marginal_error':float(np.abs(plan.sum(1)-1/len(base)).max()),
        'max_target_marginal_error':float(np.abs(plan.sum(0)-w).max())})
    constrained=evaluate_destination(ds,setting,source,base,fav,current,proxies,futures,dest,
        'Constrained-LIFT' if eta<=1e-9 else 'Minimal-Slack Constrained-LIFT',eta,float(adm.mean()))
    old=prior[prior.setting==label].copy();old=old[old.method.isin(['Margin-LIFT','Path-LIFT'])]
    allpaths=pd.concat((old,constrained),ignore_index=True,sort=False)
    # Source-level realized path audit for all three methods.
    audit=[]
    for method,g in allpaths.groupby('method'):
        if method.startswith('Constrained') or method.startswith('Minimal'):
            target=dest
        else:
            # Rebuild prior destinations exactly.
            if method=='Margin-LIFT': q=margin
            else:
                pos=vint[vint>0];beta=.1*float(eu.median())/float(pos.median()) if len(pos) else 0
                q=weighted_sinkhorn(eu+beta*vint,torch.tensor(w),iterations=1000,tol=1e-8)
            target=(q/q.sum(1,keepdim=True).clamp_min(1e-30))@fav
        grid=torch.linspace(0,1,101);z=(1-grid[:,None,None])*base[None]+grid[:,None,None]*target[None];pr=score(current,z.flatten(0,1)).view(101,len(base));mn=(pr-.5).min(0).values;vi=torch.relu(.5-pr).max(0).values
        for i in range(len(base)):audit.append({'setting':label,'method':method,'source_id':int(source_ids[i]),'minimum_realized_path_margin':float(mn[i]),'realized_path_violation':bool(mn[i]<-1e-8),'maximum_violation_magnitude':float(vi[i])})
    return pairs,pd.DataFrame([feasibility]),coupling,pd.DataFrame(audit),allpaths,front


def matched_cost(allpaths,fronts,roots):
    rows=[]
    for setting,g in allpaths.groupby('setting'):
        ds=setting.split('__')[0];setting_type=setting.split('__')[1];front=fronts[setting]
        if setting_type=='baseline_mlp':
            t=front[front.method=='T-Rex'].rename(columns={'mean_l2_cost':'mean_L2_cost'})
            audit_path=Path('/mnt/big_disk/xu.zhao/robust_ce_results/heloc_prior_balanced_audit_seed0/common_candidate_frontier.csv') if ds=='heloc' else roots['diabetes_current']/'diabetes_candidate_frontier.csv'
            aud=pd.read_csv(audit_path);aud=aud[aud.method=='trex'].set_index('candidate_id')
            tr=[]
            for r in t.itertuples():
                a=aud.loc[int(r.candidate_id)];tr.append({'method':'T-Rex','lambda':r.native_parameter,'mean_L2_cost':r.mean_L2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':a.mean_nn_target_distance,'LOF':a.mean_lof,'population_target_distance':a.population_target_distance})
            t=pd.DataFrame(tr)
        else:t=front[front.method=='T-Rex'].rename(columns={'native_parameter':'lambda'})
        pool=pd.concat((g,t),ignore_index=True,sort=False);methods=['T-Rex','Margin-LIFT','Path-LIFT',g[g.method.str.contains('Constrained')].method.iloc[0]]
        lo=max(pool[pool.method==m].mean_L2_cost.min() for m in methods);hi=min(pool[pool.method==m].mean_L2_cost.max() for m in methods)
        for budget in np.linspace(lo,hi,20):
            for method in methods:
                q=pool[(pool.method==method)&(pool.mean_L2_cost<=budget+1e-9)].sort_values(['near_in_range_worst','mean_L2_cost'],ascending=[False,True]);r=q.iloc[0]
                rows.append({'setting':setting,'cost_budget':budget,'method':method,'native_parameter':r.get('lambda',np.nan),'actual_L2_cost':r.mean_L2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':r.NN_target_distance,'LOF':r.LOF,'population_target_distance':r.population_target_distance})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    for x in ('heloc-data-home','diabetes-data-home','corrected-root','margin-root','logistic-root','heloc-current-root','diabetes-current-root','path-root','output'):p.add_argument('--'+x,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    roots={'corrected':Path(a.corrected_root),'margin':Path(a.margin_root),'logistic':Path(a.logistic_root),'heloc_current':Path(a.heloc_current_root),'diabetes_current':Path(a.diabetes_current_root)}
    prior=pd.read_csv(Path(a.path_root)/'path_lift_path_results.csv');res=[]
    for ds,setting,home in [('heloc','baseline_mlp',a.heloc_data_home),('diabetes','baseline_mlp',a.diabetes_data_home),('heloc','logistic',a.heloc_data_home)]:res.append(run_setting(ds,setting,home,roots,prior))
    names=['constrained_lift_pair_admissibility','constrained_lift_feasibility','constrained_lift_coupling','constrained_lift_realized_path_audit','constrained_lift_results']
    frames=[]
    for i,n in enumerate(names):f=pd.concat([r[i] for r in res],ignore_index=True);f.to_csv(out/f'{n}.csv',index=False);frames.append(f)
    pairs,feas,coupling,audit,results=frames;fronts={f'{ds}__{st}':r[5] for (ds,st,_),r in zip([('heloc','baseline_mlp',a.heloc_data_home),('diabetes','baseline_mlp',a.diabetes_data_home),('heloc','logistic',a.heloc_data_home)],res)}
    matched=matched_cost(results,fronts,roots);matched.to_csv(out/'constrained_lift_matched_cost.csv',index=False)
    selected=results[results.calibrated_flag==1];agg=audit.groupby(['setting','method']).agg(realized_PV=('realized_path_violation','mean'),Gamma=('maximum_violation_magnitude','mean'),mean_min_margin=('minimum_realized_path_margin','mean'),median_min_margin=('minimum_realized_path_margin','median')).reset_index();summary=selected.merge(agg,on=['setting','method'],how='left').merge(feas,on='setting',how='left')
    lines=['# Constrained-LIFT admissible-coupling ablation v1','','## Calibrated results','',markdown_table(summary),'','## Feasibility','',markdown_table(feas),'','The support mask is a diagnostic coupling constraint, not a Wasserstein metric. No paper theory is changed.']
    (out/'constrained_lift_ablation_v1.md').write_text('\n'.join(lines))
    fl=[]
    for r in feas.itertuples():fl += [f'## {r.setting}',f'- Strict feasible: {r.strict_feasible}',f'- eta_star: {r.eta_star:.10g}',f'- Admissible edges: {r.admissible_edge_fraction:.4%}',f'- Dominant targets changed: {r.dominant_target_change_fraction:.4%}',f'- Coupling L1 difference: {r.coupling_L1_vs_Margin:.6f}','']
    (out/'constrained_lift_feasibility_audit.md').write_text('# Constrained-LIFT feasibility audit\n\n'+'\n'.join(fl))
    print(summary[['setting','method','lambda','current_validity','mean_future_validity','near_in_range_worst','mean_L2_cost','realized_PV','Gamma']].to_string(index=False))
if __name__=='__main__':main()
