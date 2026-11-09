"""Prior-balanced Diabetes benchmark: Base, HighConf, Extended T-Rex, original LIFT."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from .german_credit_strict_audit import solve_wachter,trex_trajectory
from .heloc_benchmark_milestone import bootstrap_budget,transport_destination,valid
from .heloc_debug_diagnostics import markdown_table,w2_sinkhorn
from .heloc_method_bootstrap import calibration_stress
from ..data.tabular import load_tabular_three_way
from ..models.tabular import TabularMLP,train_model

ENVS=("generic_population_shift","target_side_shift","mixed_shift");SEV=(.5,1.,2.5)

def load_model(path,dim):m=TabularMLP(dim,[64,32]);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));m.eval();return m
def model_metrics(m,x,y):
    with torch.no_grad():p=torch.sigmoid(m(x));return float(((p>=.5)==y.bool()).float().mean()),float(roc_auc_score(y.numpy(),p.numpy())),float((p>=.5).float().mean())
def shift(env,x,y,severity,k,seed,bcal,features):
    z=x.clone();labels=y.clone();g=torch.Generator().manual_seed(seed);active=(y==1).nonzero().flatten() if env=="target_side_shift" else torch.arange(len(x));ids=features;frac=len(active)/len(x);kind=("mean","scale","tail")[k%3];target=severity*bcal
    if kind=="mean":z[active[:,None],ids]+=target/np.sqrt(len(ids)*frac)*torch.where(torch.rand(len(ids),generator=g)>.5,1.,-1.)
    elif kind=="scale":
        b=z[active[:,None],ids];c=b.mean(0);radius=torch.sqrt((b-c).square().sum(1).mean()).clamp_min(1e-6);z[active[:,None],ids]=c+(1+target/(np.sqrt(frac)*float(radius)))*(b-c)
    else:
        b=z[active];score=b[:,ids[0]];cut=torch.quantile(score,.5);lo=(score<=cut).nonzero().flatten();hi=(score>cut).nonzero().flatten();f=min(.5+.18*severity,.95);nl=int(len(active)*f);pick=torch.cat((lo[torch.randint(len(lo),(nl,),generator=g)],hi[torch.randint(len(hi),(len(active)-nl,),generator=g)]));z[active]=b[pick]
    if env=="mixed_shift":keep=max(96,int(len(z)*(1-min(.04*severity,.2))));idx=torch.randperm(len(z),generator=g)[:keep].sort().values;z,labels=z[idx],labels[idx]
    return z,labels,kind

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-home",required=True);ap.add_argument("--output",required=True);ap.add_argument("--seed",type=int,default=0);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"models").mkdir(exist_ok=True)
    d=load_tabular_three_way({"source":"openml","openml_id":37,"positive_label":"tested_negative"},a.data_home,.6,.2,a.seed);train_x,train_y,cal_x,cal_y,test_x,test_y=(d[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y"));names=d["feature_names"]
    current,_=train_model(train_x,train_y,cal_x,cal_y,[64,32],40,1e-3,a.seed);torch.save(current.state_dict(),out/"models/current.pt");tr,ca,te=model_metrics(current,train_x,train_y),model_metrics(current,cal_x,cal_y),model_metrics(current,test_x,test_y)
    with torch.no_grad():score=torch.sigmoid(current(test_x));negative=(score<.5).nonzero().flatten();gen=torch.Generator().manual_seed(a.seed+2);loc=negative[torch.randperm(len(negative),generator=gen)[:min(128,len(negative))]]
    source=test_x[loc];source_ids=np.asarray(d["test_ids"])[loc.numpy()];groups={};protos={};thresholds=[.5]+[round(float(x),2) for x in np.arange(.55,.951,.05)];solved={};solverrows=[]
    for p in thresholds:
        z,rows=solve_wachter(current,source,p,groups,protos,5,500,a.seed+90_000);solved[p]=z
        for r in rows:r["source_id"]=int(source_ids[r.pop("source_index")])
        solverrows.extend(rows)
    pd.DataFrame(solverrows).to_csv(out/"diabetes_wachter_solver_audit.csv",index=False);base=solved[.5];favorable=train_x[valid(current,train_x)];destination=transport_destination(base,favorable)
    bcal,_=bootstrap_budget(cal_x,20,min(128,len(cal_x)),a.seed+4);proxies=[];stress=[]
    for m in range(5):stressed,b=calibration_stress(cal_x,bcal,a.seed+10_000+m);pm,_=train_model(torch.cat((train_x,stressed)),torch.cat((train_y,cal_y)),test_x,test_y,[64,32],40,1e-3,a.seed+1000+m);proxies.append(pm);stress.append({"proxy":m,"realized_shift":b,"b_cal":bcal})
    candidates=[("base",0.,base,"fixed")]+[("highconf",p,solved[p],"equal-budget") for p in thresholds[1:]]+[("lift",float(l),(1-l)*base+l*destination,"original barycentric") for l in np.linspace(0,1,101)]
    cache=out/"trex_trajectories.pt"
    if cache.exists():trajectories=torch.load(cache,map_location="cpu",weights_only=True)
    else:
        trajectories={}
        for lr in (.005,.01,.02):t,s=trex_trajectory(current,base,lr,groups,protos,a.seed+50_000);trajectories[f"lr_{lr:g}"]={"trajectory":t,"scores":s}
        torch.save(trajectories,cache)
    for lr in (.005,.01,.02):
      t,s=trajectories[f"lr_{lr:g}"]["trajectory"],trajectories[f"lr_{lr:g}"]["scores"]
      for steps in (50,100,200):
       ss=s[:steps]
       for tau in np.arange(.5,.951,.05):ok=ss>=tau;idx=ok.float().argmax(0);reach=ok.any(0);idx[~reach]=steps;candidates.append(("trex",float(tau),t[idx,torch.arange(len(base))],f"lr={lr};steps={steps};reach={float(reach.float().mean()):.3f}"))
    ranges=(train_x.max(0).values-train_x.min(0).values).clamp_min(1e-6);weights=1/(len(names)*ranges.square());pd.DataFrame({"feature":names,"type":"continuous_or_ordinal","training_range":ranges.numpy(),"weight":weights.numpy(),"maximum_contribution":1/len(names)}).to_csv(out/"diabetes_semantic_cost_schema.csv",index=False)
    lof=LocalOutlierFactor(n_neighbors=min(20,len(favorable)-1),novelty=True).fit(favorable.numpy());rows=[];proxyrows=[];tensors={}
    for cid,(method,param,z,detail) in enumerate(candidates):
        vals=[];risks=[]
        with torch.no_grad():
          for j,pm in enumerate(proxies):p=torch.sigmoid(pm(z));vals.append(float((p>=.5).float().mean()));risks.append(float(torch.relu(torch.tensor(.5)-p).mean()));proxyrows.append({"candidate_id":cid,"method":method,"parameter":param,"proxy":j,"validity":vals[-1],"risk":risks[-1]})
        l2=torch.linalg.vector_norm(z-source,dim=1);mix=((z-source).square()*weights).sum(1);nn=torch.cdist(z,favorable).min(1).values;rows.append({"candidate_id":cid,"method":method,"native_parameter":param,"detail":detail,"current_validity":float(valid(current,z).float().mean()),"mean_l2_cost":float(l2.mean()),"median_l2_cost":float(l2.median()),"p90_l2_cost":float(torch.quantile(l2,.9)),"mean_mixed_cost":float(mix.mean()),"median_mixed_cost":float(mix.median()),"p90_mixed_cost":float(torch.quantile(mix,.9)),"mean_nn_target_distance":float(nn.mean()),"mean_lof":float(lof.score_samples(z.numpy()).mean()),"population_target_distance":w2_sinkhorn(z,favorable),"proxy_mean_validity":np.mean(vals),"proxy_worst_validity":np.min(vals),"calibration_risk":max(risks)});tensors[str(cid)]=z
    cand=pd.DataFrame(rows);cand["calibrated_flag"]=0;cand["selection_status"]="not_selected";cand.loc[cand.method=="base",["calibrated_flag","selection_status"]]=[1,"fixed"]
    for method in ("highconf","trex"):
        q=cand[cand.method==method];f=q[q.proxy_mean_validity>=.95];r=(f.sort_values("mean_l2_cost").iloc[0] if len(f) else q.sort_values(["proxy_mean_validity","mean_l2_cost"],ascending=[False,True]).iloc[0]);cand.loc[r.name,["calibrated_flag","selection_status"]]=[1,"properly_calibrated" if len(f) else "fallback_target_infeasible"]
    q=cand[cand.method=="lift"].sort_values("native_parameter");r0,r1=q.iloc[0].calibration_risk,q.iloc[-1].calibration_risk;gain=(r0-q.calibration_risk)/max(r0-r1,1e-12)-q.native_parameter;cand.loc[gain.idxmax(),["calibrated_flag","selection_status"]]=[1,"gain_over_transport"];pd.DataFrame(proxyrows).to_csv(out/"diabetes_calibration_proxy_scores.csv",index=False);pd.DataFrame(stress).to_csv(out/"diabetes_calibration_stresses.csv",index=False)
    # Training-only predictive features for common shifts.
    cor=[]
    for j in range(train_x.shape[1]):cor.append(abs(float(np.corrcoef(train_x[:,j].numpy(),train_y.numpy())[0,1])))
    features=torch.as_tensor(np.argsort(cor)[-3:][::-1].copy());protocol=[];shifted={}
    for env in ENVS:
      for sev in SEV:
       for k in range(5):
        sx,sy,kind=shift(env,train_x,train_y,sev,k,a.seed+20_000+1000*k,bcal,features);n=min(len(sx),len(train_x));dist=float(torch.sqrt((sx[:n]-train_x[:n]).square().sum(1).mean()));ratio=dist/bcal;protocol.append({"environment":env,"severity":sev,"future_model_id":k,"kind":kind,"empirical_shift_distance":dist,"b_cal":bcal,"relative_shift":ratio,"range_group":"in_range" if ratio<.75 else "near_range" if ratio<=1 else "moderate_out_of_range" if ratio<=2 else "strong_out_of_range"});shifted[(env,sev,k)]=(sx,sy)
    pd.DataFrame(protocol).to_csv(out/"diabetes_shift_scale_audit.csv",index=False);future={}
    for r in protocol:
        e,s,k=r["environment"],r["severity"],r["future_model_id"];path=out/"models"/f"{e}_s{s:g}_k{k}.pt"
        if not path.exists():sx,sy=shifted[(e,s,k)];m,_=train_model(sx,sy,test_x,test_y,[64,32],40,1e-3,a.seed+30_000+10_000*ENVS.index(e)+100*int(s*10)+k);torch.save(m.state_dict(),path)
        future[(e,s,k)]=load_model(path,train_x.shape[1])
    ev=[];ranges_lookup={(r["environment"],r["severity"],r["future_model_id"]):r["range_group"] for r in protocol}
    for _,r in cand.iterrows():
      z=tensors[str(int(r.candidate_id))]
      for (e,s,k),m in future.items():ev.append({"candidate_id":int(r.candidate_id),"method":r.method,"native_parameter":r.native_parameter,"environment":e,"severity":s,"future_model_id":k,"range_group":ranges_lookup[(e,s,k)],"validity":float(valid(m,z).float().mean())})
    ev=pd.DataFrame(ev);ev.to_csv(out/"diabetes_future_evaluation.csv",index=False);per=ev.groupby(["candidate_id","environment"]).validity.min().unstack();cand["mean_future_validity"]=cand.candidate_id.map(ev.groupby("candidate_id").validity.mean());
    for e in ENVS:cand[f"worst__{e}"]=cand.candidate_id.map(per[e])
    for rg in ("in_range","near_range","moderate_out_of_range","strong_out_of_range"):cand[f"common_{rg}_worst"]=cand.candidate_id.map(ev[ev.range_group==rg].groupby("candidate_id").validity.min())
    cand["common_global_worst"]=cand[[f"worst__{e}" for e in ENVS]].min(1);cand.to_csv(out/"diabetes_candidate_frontier.csv",index=False);torch.save(tensors,out/"diabetes_candidate_counterfactuals.pt")
    # Matched budgets under both cost definitions.
    methods=("base","highconf","trex","lift");matched=[]
    for metric in ("mean_l2_cost","mean_mixed_cost"):
        swept=methods[1:];lo=max(cand[cand.method==m][metric].min() for m in swept);hi=min(cand[cand.method==m][metric].max() for m in swept);budgets=np.linspace(lo,hi,15) if hi>=lo else []
        for b in budgets:
          for method in methods:
            g=cand[cand.method==method];qq=g[g[metric]<=b+1e-9]
            if qq.empty or (method!="base" and b>g[metric].max()+1e-9):continue
            r=qq.sort_values(["common_near_range_worst",metric],ascending=[False,True]).iloc[0];matched.append({"budget_metric":metric,"cost_budget":b,"method":method,"parameter":r.native_parameter,"actual_l2_cost":r.mean_l2_cost,"actual_mixed_cost":r.mean_mixed_cost,"current_validity":r.current_validity,"mean_future_validity":r.mean_future_validity,"near_range_worst":r.common_near_range_worst,"generic_worst":r.worst__generic_population_shift,"target_side_worst":r.worst__target_side_shift,"mixed_worst":r.worst__mixed_shift,"nearest_target_distance":r.mean_nn_target_distance,"mean_lof":r.mean_lof,"population_target_distance":r.population_target_distance})
    matched=pd.DataFrame(matched);matched.to_csv(out/"diabetes_matched_cost.csv",index=False);selected=cand[cand.calibrated_flag==1];selected.to_csv(out/"diabetes_selected_results.csv",index=False)
    report=["# Diabetes original-LIFT comparison","",f"Favorable outcome: tested_negative. Distinct predicted-unfavorable test sources: {len(source)}.","","## Current classifier",markdown_table(pd.DataFrame([{"train_accuracy":tr[0],"train_auc":tr[1],"cal_accuracy":ca[0],"cal_auc":ca[1],"test_accuracy":te[0],"test_auc":te[1],"favorable_train_fraction":train_y.mean()}])),"","## Calibration-selected results","",markdown_table(selected),"","## Matched L2 and semantic-cost comparisons","",markdown_table(matched),"","All methods share the same model, sources, equal-budget Base/HighConf solver, calibration proxies, and future classifiers. Original LIFT uses its soft barycentric OT destination. No final future result enters parameter selection."];(out/"diabetes_original_lift_comparison.md").write_text("\n".join(report));print(json.dumps({"sources":len(source),"b_cal":bcal,"selected":selected[["method","native_parameter","selection_status","current_validity","mean_l2_cost","mean_mixed_cost","mean_future_validity","common_near_range_worst"]].to_dict("records"),"matched_rows":len(matched)},indent=2))

if __name__=="__main__":main()
