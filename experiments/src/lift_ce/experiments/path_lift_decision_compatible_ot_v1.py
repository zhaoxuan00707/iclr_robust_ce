"""Minimal current-decision-compatible OT cost ablation for LIFT."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.neighbors import LocalOutlierFactor
from .stability_weighted_anchor_v1 import reconstruct,weighted_sinkhorn,load_model
from .heloc_benchmark_milestone import valid,bootstrap_budget
from .heloc_method_bootstrap import calibration_stress
from .heloc_debug_diagnostics import w2_sinkhorn,markdown_table
from .logistic_margin_anchor_benchmark import Logistic,train_logistic
import lift_ce.experiments.logistic_margin_anchor_benchmark as logistic_module
from ..data.tabular import load_tabular_three_way

MULT=(0.,.5,1.,2.,5.)

def score(m,x):
    with torch.no_grad():return torch.sigmoid(m(x))

def pair_violations(model,base,target,batch=32768):
    n,m=len(base),len(target);vint=torch.empty(n,m);vmax=torch.empty(n,m);vfrac=torch.empty(n,m);mmin=torch.empty(n,m);grid=torch.linspace(0,1,21)
    pairs=torch.cartesian_prod(torch.arange(n),torch.arange(m))
    for a in range(0,len(pairs),batch):
        ij=pairs[a:a+batch];z=base[ij[:,0]];y=target[ij[:,1]];x=((1-grid[:,None,None])*z[None]+grid[:,None,None]*y[None]).flatten(0,1);p=score(model,x).view(21,-1);vi=torch.relu(.5-p);ii=ij[:,0];jj=ij[:,1];vint[ii,jj]=vi.mean(0);vmax[ii,jj]=vi.max(0).values;vfrac[ii,jj]=(p<.5).float().mean(0);mmin[ii,jj]=(p-.5).min(0).values
    return vint,vmax,vfrac,mmin

def setting_artifacts(ds,setting,data_home,roots):
    cfg={'source':'openml','openml_name':'heloc','openml_version':1,'missing_values':[-7,-8,-9]} if ds=='heloc' else {'source':'openml','openml_id':37,'positive_label':'tested_negative'};d=load_tabular_three_way(cfg,data_home,.6,.2,0);trainx,trainy,calx,caly,testx,testy=(d[k] for k in ('train_x','train_y','cal_x','cal_y','test_x','test_y'))
    if setting=='baseline_mlp':
        root=roots['heloc_current'] if ds=='heloc' else roots['diabetes_current'];train_x,train_y,source,source_ids,base,current,proxies,_,baseline,trex=reconstruct(ds,data_home,root);corrected=roots['corrected'];sa=pd.read_csv(corrected/'corrected_future_classifier_sanity.csv');sa=sa[sa.dataset==ds];futures=[(r.environment,float(r.severity),int(r.future_model_id),r.range_group,load_model(corrected/ds/'models'/f'{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt',trainx.shape[1])) for r in sa.itertuples()];front=pd.read_csv(corrected/'corrected_method_results.csv');front=front[front.dataset==ds];tensors=(torch.load('/mnt/big_disk/xu.zhao/robust_ce_results/heloc_prior_balanced_audit_seed0/candidate_counterfactuals.pt',map_location='cpu',weights_only=True) if ds=='heloc' else torch.load(root/'diabetes_candidate_counterfactuals.pt',map_location='cpu',weights_only=True));return d,source,source_ids,base,current,proxies,futures,front,tensors,trex
    root=roots['logistic']/ds;current=Logistic(trainx.shape[1]);current.load_state_dict(torch.load(root/'models/current.pt',weights_only=True));current.eval();neg=(score(current,testx)<.5).nonzero().flatten();loc=neg[torch.randperm(len(neg),generator=torch.Generator().manual_seed(2))[:min(128,len(neg))]];source=testx[loc];source_ids=np.asarray(d['test_ids'])[loc.numpy()];front=pd.read_csv(root/f'{ds}_logistic_frontier.csv');tensors=torch.load(root/f'{ds}_logistic_counterfactuals.pt',map_location='cpu',weights_only=True);base=tensors[int(front[front.method=='Base'].iloc[0].candidate_id)];bcal,_=bootstrap_budget(calx,20,min(128,len(calx)),4);logistic_module.MODEL_KIND='logistic';logistic_module.WEIGHT_DECAY=1e-4;proxies=[]
    for j in range(5):stressed,_=calibration_stress(calx,bcal,10_000+j);proxies.append(train_logistic(torch.cat((trainx,stressed)),torch.cat((trainy,caly)),30,1000+j))
    sa=pd.read_csv(root/f'{ds}_logistic_future_sanity.csv');futures=[]
    for r in sa.itertuples():m=Logistic(trainx.shape[1]);m.load_state_dict(torch.load(root/'models'/f'{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt',weights_only=True));m.eval();futures.append((r.environment,float(r.severity),int(r.future_model_id),r.range_group,m))
    tr=front[(front.method=='T-Rex')&(front.calibrated_flag==1)].iloc[0];trex=tensors[int(tr.candidate_id)];return d,source,source_ids,base,current,proxies,futures,front,tensors,trex

def realized_metrics(model,base,dest):
    grid=torch.linspace(0,1,101);z=(1-grid[:,None,None])*base[None]+grid[:,None,None]*dest[None];p=score(model,z.flatten(0,1)).view(101,len(base));endpoint=(p[0]>=.5)&(p[-1]>=.5);vi=torch.relu(.5-p).max(0).values;minimum=(p-.5).min(0).values;q=endpoint
    return {'actual_path_violation_rate':float((minimum[q]<-1e-8).float().mean()) if q.any() else np.nan,'actual_Gamma':float(vi[q].mean()) if q.any() else np.nan,'mean_minimum_path_margin':float(minimum[q].mean()) if q.any() else np.nan,'median_minimum_path_margin':float(minimum[q].median()) if q.any() else np.nan,'endpoint_valid_sources':int(q.sum())}

def evaluate(ds,setting,home,roots,out):
    d,source,source_ids,base,current,proxies,futures,baseline,tensors,trex=setting_artifacts(ds,setting,home,roots);trainx=d['train_x'];fav=trainx[valid(current,trainx)];fs=score(current,fav);margin=torch.relu(fs-.5);raw=torch.minimum(margin,torch.tensor(.25))+1e-6;w=raw/raw.sum();eu=torch.cdist(base,fav).square();vint,vmax,vfrac,mmin=pair_violations(current,base,fav);positive=vint[vint>0];beta_primary=.1*float(eu.median())/float(positive.median()) if len(positive) else 0.;label=f'{ds}__{setting}';pairrows=[]
    for i in range(min(16,len(base))):
      for j in range(min(256,len(fav))):pairrows.append({'setting':label,'source_id':int(source_ids[i]),'target_id':j,'euclidean_cost':float(eu[i,j]),'Vmax':float(vmax[i,j]),'Vint':float(vint[i,j]),'Vfrac':float(vfrac[i,j]),'Mmin':float(mmin[i,j])})
    plans={};dest={};coupling=[]
    uniform_w=torch.full((len(fav),),1/len(fav));uniform_plan=weighted_sinkhorn(eu,uniform_w,iterations=1000,tol=1e-8);plans['Uniform-LIFT']=uniform_plan;dest['Uniform-LIFT']=(uniform_plan/uniform_plan.sum(1,keepdim=True).clamp_min(1e-30))@fav
    for mult in MULT:
        name='Margin-LIFT' if mult==0 else 'Path-LIFT' if mult==1 else f'Path-LIFT-beta-x{mult:g}';cost=eu+(mult*beta_primary)*vint;plan=weighted_sinkhorn(cost,w,iterations=1000,tol=1e-8);plans[name]=plan;dest[name]=(plan/plan.sum(1,keepdim=True).clamp_min(1e-30))@fav
    hardcost=eu.clone();penalty=float(eu.max()+1000*eu.median());hardcost[mmin<0]=penalty;hard=weighted_sinkhorn(hardcost,w,iterations=1000,tol=1e-8);plans['HardPath-LIFT']=hard;dest['HardPath-LIFT']=(hard/hard.sum(1,keepdim=True).clamp_min(1e-30))@fav
    pm,pp=plans['Margin-LIFT'],plans['Path-LIFT'];im,ip=pm.argmax(1),pp.argmax(1)
    for i in range(len(base)):coupling.append({'setting':label,'source_id':int(source_ids[i]),'dominant_target_margin':int(im[i]),'dominant_target_path':int(ip[i]),'dominant_target_changed':bool(im[i]!=ip[i]),'euclidean_cost_path_target':float(eu[i,ip[i]]),'path_violation_path_target':float(vint[i,ip[i]]),'path_aware_cost_path_target':float(eu[i,ip[i]]+beta_primary*vint[i,ip[i]]),'source_marginal_error':float((pp.sum(1)-1/len(base)).abs().max()),'target_marginal_error':float((pp.sum(0)-w).abs().max())})
    lof=LocalOutlierFactor(n_neighbors=min(20,len(fav)-1),novelty=True).fit(fav.numpy());pathrows=[];sensitivity=[]
    for name,destination in dest.items():
        mechanism=realized_metrics(current,base,destination)
        for lam in np.linspace(0,1,101):
            lam=round(float(lam),2);z=(1-lam)*base+lam*destination;vals=[];risks=[]
            for proxy in proxies:p=score(proxy,z);vals.append(float((p>=.5).float().mean()));risks.append(float(torch.relu(.5-p).mean()))
            l2=torch.linalg.vector_norm(z-source,dim=1);nn=torch.cdist(z,fav).min(1).values;local=[]
            for env,sev,k,rg,fm in futures:local.append({'environment':env,'severity':sev,'future_model_id':k,'range_group':rg,'validity':float(valid(fm,z).float().mean())})
            ev=pd.DataFrame(local);meaning=ev[ev.range_group.isin(['in_range','near_range'])];beta_mult=(0 if name in ('Margin-LIFT','Uniform-LIFT') else 1 if name=='Path-LIFT' else np.nan if name=='HardPath-LIFT' else float(name.rsplit('x',1)[1]));row={'setting':label,'dataset':ds,'classifier_setting':setting,'method':name,'beta_multiplier':beta_mult,'beta_primary':beta_primary,'lambda':lam,'current_validity':float(valid(current,z).float().mean()),'calibration_risk':max(risks),'proxy_worst_validity':min(vals),'mean_future_validity':float(ev.validity.mean()),'near_in_range_worst':float(meaning.validity.min()),'mean_L2_cost':float(l2.mean()),'median_L2_cost':float(l2.median()),'p90_L2_cost':float(torch.quantile(l2,.9)),'NN_target_distance':float(nn.mean()),'LOF':float(lof.score_samples(z.numpy()).mean()),'population_target_distance':w2_sinkhorn(z,fav),**mechanism}
            for env in ev.environment.unique():row[f'worst__{env}']=float(ev[ev.environment==env].validity.min())
            tq=meaning[meaning.environment=='target_side_shift'];row['target_side_near_range_worst']=float(tq.validity.min()) if len(tq) else np.nan;pathrows.append(row)
    path=pd.DataFrame(pathrows);selected=[]
    for name,g in path.groupby('method'):
        g=g.sort_values('lambda');r0,r1=g.iloc[0].calibration_risk,g.iloc[-1].calibration_risk;gain=(r0-g.calibration_risk)/max(r0-r1,1e-12)-g['lambda'];idx=gain.idxmax();path.loc[idx,'calibrated_flag']=1;selected.append(path.loc[idx])
    path['calibrated_flag']=path.calibrated_flag.fillna(0).astype(int);selected=pd.DataFrame(selected);sensitivity=selected.copy()
    # Matched L2 cost with frozen T-Rex frontier.
    old=[]
    if setting=='baseline_mlp':
      corr=baseline
      audit_path=(Path('/mnt/big_disk/xu.zhao/robust_ce_results/heloc_prior_balanced_audit_seed0/common_candidate_frontier.csv') if ds=='heloc' else roots['diabetes_current']/'diabetes_candidate_frontier.csv')
      audit=pd.read_csv(audit_path);audit=audit[audit.method=='trex'].set_index('candidate_id')
      for _,r in corr[corr.method=='T-Rex'].iterrows():
        arow=audit.loc[int(r.candidate_id)]
        old.append({'method':'T-Rex','native_parameter':r.native_parameter,'mean_L2_cost':r.mean_l2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':arow.mean_nn_target_distance,'LOF':arow.mean_lof,'population_target_distance':arow.population_target_distance})
    else:
      for _,r in baseline[baseline.method=='T-Rex'].iterrows():old.append({'method':'T-Rex','native_parameter':r.native_parameter,'mean_L2_cost':r.mean_L2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':r.NN_target_distance,'LOF':r.LOF,'population_target_distance':r.population_target_distance})
    lp=path[path.method.isin(['Margin-LIFT','Path-LIFT'])].rename(columns={'lambda':'native_parameter'});pool=pd.concat([pd.DataFrame(old),lp],ignore_index=True,sort=False);methods=['T-Rex','Margin-LIFT','Path-LIFT'];lo=max(pool[pool.method==m].mean_L2_cost.min() for m in methods);hi=min(pool[pool.method==m].mean_L2_cost.max() for m in methods);matched=[]
    for budget in np.linspace(lo,hi,20):
      for method in methods:
        q=pool[(pool.method==method)&(pool.mean_L2_cost<=budget+1e-9)];r=q.sort_values(['near_in_range_worst','mean_L2_cost'],ascending=[False,True]).iloc[0];matched.append({'setting':label,'dataset':ds,'classifier_setting':setting,'cost_budget':budget,'method':method,'native_parameter':r.native_parameter,'actual_L2_cost':r.mean_L2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':r.NN_target_distance,'LOF':r.LOF,'population_target_distance':r.population_target_distance})
    # Direction cosines.
    dirs=[];md=dest['Margin-LIFT']-base;pdirection=dest['Path-LIFT']-base;td=trex-base
    for name,a,b in [('Path_vs_Margin',pdirection,md),('Path_vs_T-Rex',pdirection,td),('Margin_vs_T-Rex',md,td)]:
        c=torch.nn.functional.cosine_similarity(a,b,dim=1,eps=1e-12);dirs.append({'setting':label,'comparison':name,'mean':float(c.mean()),'median':float(c.median()),'positive_fraction':float((c>0).float().mean())})
    return pd.DataFrame(pairrows),pd.DataFrame(coupling),path,pd.DataFrame(matched),sensitivity,pd.DataFrame(dirs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--heloc-data-home',required=True);p.add_argument('--diabetes-data-home',required=True);p.add_argument('--corrected-root',required=True);p.add_argument('--margin-root',required=True);p.add_argument('--logistic-root',required=True);p.add_argument('--heloc-current-root',required=True);p.add_argument('--diabetes-current-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);roots={'corrected':Path(a.corrected_root),'margin':Path(a.margin_root),'logistic':Path(a.logistic_root),'heloc_current':Path(a.heloc_current_root),'diabetes_current':Path(a.diabetes_current_root)};res=[]
    for ds,setting,home in [('heloc','baseline_mlp',a.heloc_data_home),('diabetes','baseline_mlp',a.diabetes_data_home),('heloc','logistic',a.heloc_data_home)]:res.append(evaluate(ds,setting,home,roots,out))
    names=('path_lift_pairwise_violation_diagnostic','path_lift_ot_coupling_diagnostic','path_lift_path_results','path_lift_matched_cost','path_lift_beta_sensitivity','path_lift_direction_diagnostic');frames=[]
    for i,name in enumerate(names):f=pd.concat([r[i] for r in res],ignore_index=True);f.to_csv(out/f'{name}.csv',index=False);frames.append(f)
    pair,couple,path,matched,sens,dirs=frames;primary=sens[sens.method.isin(['Margin-LIFT','Path-LIFT','HardPath-LIFT'])];changes=couple.groupby('setting').dominant_target_changed.mean().reset_index(name='dominant_target_change_fraction');summary=primary.merge(changes,on='setting',how='left')
    report=['# Decision-compatible Path-LIFT OT ablation v1','','The additive path penalty is a computational diagnostic, not a metric, and is not inserted into any Wasserstein theorem. Beta is set exclusively by current-data scale matching.','', '## Calibrated results','',markdown_table(summary),'','## Direction diagnostics','',markdown_table(dirs),'','## Matched-cost results','',markdown_table(matched)]
    (out/'path_lift_decision_compatible_ot_v1.md').write_text('\n'.join(report));print(summary[['setting','method','beta_primary','lambda','current_validity','mean_future_validity','near_in_range_worst','mean_L2_cost','actual_path_violation_rate','actual_Gamma','dominant_target_change_fraction']].to_string(index=False))
if __name__=='__main__':main()
