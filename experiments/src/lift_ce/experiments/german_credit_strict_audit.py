"""Strict solver, categorical, and future-model audit for German Credit."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.neighbors import LocalOutlierFactor

from .german_credit_benchmark_milestone_v1 import ENVS, GEOMETRY, SEVERITIES, load_model, masked_stress, plaus
from .german_credit_unconstrained_rerun import global_destination
from .heloc_benchmark_milestone import valid
from .heloc_debug_diagnostics import markdown_table
from ..data.tabular import load_tabular_three_way
from ..models.tabular import train_model


def category_groups(names):
    numeric={"duration","credit_amount","installment_commitment","residence_since","age","existing_credits","num_dependents"}
    roots=[]
    for name in names:
        if name in numeric: continue
        root=name.rsplit("_",1)[0]
        if root not in roots: roots.append(root)
    # Robustly recover roots by matching original German field prefixes.
    fields=("checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker")
    return {f:[i for i,n in enumerate(names) if n.startswith(f+"_")] for f in fields}


def prototypes(train_x, groups):
    return {name:torch.unique(train_x[:,ids],dim=0) for name,ids in groups.items()}


def project(x, groups, protos):
    z=x.clone()
    for name,ids in groups.items():
        p=protos[name]
        distance=z[:,ids].square().sum(1,keepdim=True)+p.square().sum(1).unsqueeze(0)-2*z[:,ids]@p.T
        z[:,ids]=p[distance.argmin(1)]
    return z


def straight_through_project(x,groups,protos):
    px=project(x,groups,protos)
    return x+(px-x).detach()


def invalid_counts(x,groups,protos,tol=1e-6):
    bad=torch.zeros(len(x),dtype=torch.bool); fractional={}
    for name,ids in groups.items():
        dist=(x[:,None,ids]-protos[name][None]).abs().amax(2).min(1).values
        this=dist>tol; bad|=this; fractional[name]=int(this.sum())
    return int(bad.sum()),fractional


def solve_wachter(model,source,threshold,groups,protos,restarts=5,steps=500,seed=0):
    records=[];batch=[]
    for restart in range(restarts):
        gen=torch.Generator().manual_seed(seed+restart);jitter=torch.zeros_like(source) if restart==0 else .02*torch.randn(source.shape,generator=gen);batch.append(source+jitter)
    expanded_source=source.repeat(restarts,1);x=torch.cat(batch).requires_grad_(True);opt=torch.optim.Adam([x],lr=.03);best=torch.full((len(x),),float("inf"));best_x=project(expanded_source,groups,protos);best_step=torch.full((len(x),),-1,dtype=torch.long)
    for step in range(steps):
        opt.zero_grad();xp=straight_through_project(x,groups,protos);score=torch.sigmoid(model(xp));loss=(20*torch.relu(threshold-score)+.05*(xp-expanded_source).square().mean(1)).mean();loss.backward();opt.step()
        with torch.no_grad():
            xp=project(x,groups,protos);score=torch.sigmoid(model(xp));cost=torch.linalg.vector_norm(xp-expanded_source,dim=1);use=(score>=threshold)&(cost<best);best[use]=cost[use];best_x[use]=xp[use];best_step[use]=step+1
    with torch.no_grad():score=torch.sigmoid(model(best_x));ok=score>=threshold
    solutions=[]
    for restart in range(restarts):
        sl=slice(restart*len(source),(restart+1)*len(source));solutions.append((best_x[sl].clone(),best[sl].clone(),ok[sl].clone(),best_step[sl].clone(),score[sl].clone()))
        for i in range(len(source)):records.append({"source_index":i,"confidence":threshold,"current_valid":bool(ok[sl][i]),"l2_cost":float(best[sl][i]) if ok[sl][i] else np.nan,"optimization_steps":int(best_step[sl][i]),"restart_id":restart,"final_classifier_score":float(score[sl][i]),"selected":False})
    # Per source, keep minimum valid restart, otherwise highest score.
    chosen=[]
    for i in range(len(source)):
        candidates=[(float(sol[1][i]),r) for r,sol in enumerate(solutions) if bool(sol[2][i])]
        r=min(candidates)[1] if candidates else max(range(restarts),key=lambda j:float(solutions[j][4][i]))
        chosen.append(solutions[r][0][i]);records[r*len(source)+i]["selected"]=True
    return torch.stack(chosen),records


def trex_trajectory(model,base,lr,groups,protos,seed):
    x=base.clone();traj=[x.clone()];scores=[]
    for step in range(200):
        x=x.detach().requires_grad_(True);xp=straight_through_project(x,groups,protos);clean=torch.sigmoid(model(xp));total=torch.zeros_like(clean);gen=torch.Generator().manual_seed(seed+step)
        for _ in range(10):
            noise=.1*torch.randn((100,)+x.shape,generator=gen);noisy=straight_through_project((x.unsqueeze(0)+noise).flatten(0,1),groups,protos);p=torch.sigmoid(model(noisy)).view(100,len(x));total+=(p-(clean.unsqueeze(0)-p).abs()).sum(0)
        score=total/1000;score.sum().backward();x=project((x+lr*x.grad).detach(),groups,protos);traj.append(x);scores.append(score.detach())
        if (step+1)%50==0:print(f"strict T-Rex lr={lr:g} step={step+1} stability={score.mean():.4f}",flush=True)
    return torch.stack(traj),torch.stack(scores)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-home",required=True);ap.add_argument("--previous",required=True);ap.add_argument("--output",required=True);ap.add_argument("--seed",type=int,default=0);a=ap.parse_args()
    previous,out=Path(a.previous),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    d=load_tabular_three_way({"source":"openml","openml_id":31,"positive_label":"good"},a.data_home,.6,.2,a.seed);train_x,train_y,cal_x,cal_y,test_x,test_y=(d[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y"));names=d["feature_names"]
    current=load_model(previous/"models/current.pt",train_x.shape[1]);art=torch.load(previous/"core_artifacts.pt",map_location="cpu",weights_only=True);source=art["source"]
    source_ids=np.load(previous/"german_split_source_ids.npz")["sources"]
    groups=category_groups(names);protos=prototypes(train_x,groups)

    # 1. Equal-budget solver audit.
    thresholds=[.5]+[round(float(x),2) for x in np.arange(.55,.951,.05)];solver_rows=[];solved={}
    for p in thresholds:
        sol,rows=solve_wachter(current,source,p,groups,protos,5,500,a.seed+90_000)
        solved[p]=sol
        for row in rows:row["source_id"]=int(source_ids[row.pop("source_index")])
        solver_rows.extend(rows)
    solver=pd.DataFrame(solver_rows);solver.to_csv(out/"german_wachter_solver_audit.csv",index=False)
    picked=solver[solver.selected].pivot(index="source_id",columns="confidence",values="l2_cost");cheaper=float((picked[.95]<picked[.5]).mean())
    summary=solver[solver.selected].groupby("confidence").l2_cost.agg(["mean","median","count"]).reset_index()
    fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(summary.confidence,summary["mean"],"o-",label="mean");ax.plot(summary.confidence,summary["median"],"s-",label="median");ax.set(xlabel="target confidence",ylabel="L2 cost");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/"german_wachter_confidence_vs_cost.png",dpi=220);plt.close(fig)
    (out/"german_wachter_solver_audit.md").write_text("# German Wachter solver audit\n\nBoth Base and HighConf use five paired initializations, Adam(lr=.03), 500 steps, and identical objectives except target confidence.\n\n"+markdown_table(summary)+f"\n\nFraction with cost(0.95) < cost(0.50): **{cheaper:.1%}**.\n")

    base=solved[.5];favorable=train_x[valid(current,train_x)];dest=global_destination(base,favorable);bcal=float(json.loads((previous/"metadata.json").read_text())["b_cal"])
    proxies=[]
    for m in range(5):
        stressed,_=masked_stress(cal_x,bcal,a.seed+10_000+m);proxy,_=train_model(torch.cat((train_x,stressed)),torch.cat((train_y,cal_y)),test_x,test_y,[64,32],60,1e-3,a.seed+1000+m);proxies.append(proxy)
    candidates=[("base",0.,base,"equal-budget")]
    for p in thresholds[1:]:candidates.append(("highconf",p,solved[p],"equal-budget"))
    for lam in np.linspace(0,1,101):candidates.append(("lift_global",float(lam),project((1-lam)*base+lam*dest,groups,protos),"projected categorical"))
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
            for tau in np.arange(.5,.951,.05):
                ok=ss>=tau;idx=ok.float().argmax(0);reach=ok.any(0);idx[~reach]=steps;candidates.append(("trex",float(tau),project(t[idx,torch.arange(len(base))],groups,protos),f"lr={lr};steps={steps};reach={float(reach.float().mean()):.3f}"))

    # 2. Categorical audit before/after realization; all evaluated tensors are projected.
    raw_previous=torch.load(previous.parent/"german_credit_benchmark_milestone_v1_unconstrained_seed0"/"candidate_counterfactuals.pt",map_location="cpu",weights_only=True)
    prev_front=pd.read_csv(previous.parent/"german_credit_benchmark_milestone_v1_unconstrained_seed0"/"german_candidate_frontier.csv")
    catrows=[]
    for method in ("base","highconf","trex","lift_global"):
        q=prev_front[prev_front.method==method];raw=torch.cat([raw_previous[str(int(i))] for i in q.candidate_id]);bad,detail=invalid_counts(raw,groups,protos);fixed=project(raw,groups,protos);bad2,_=invalid_counts(fixed,groups,protos)
        catrows.append({"method":method,"number_of_cfs":len(raw),"invalid_before_realization":bad,"invalid_fraction_before":bad/len(raw),"invalid_categorical_count":bad2,"invalid_categorical_fraction":bad2/len(raw),"group_failures_before":json.dumps(detail)})
    cat=pd.DataFrame(catrows);cat.to_csv(out/"german_categorical_validity_audit.csv",index=False)
    (out/"german_categorical_validity_audit.md").write_text("# German categorical validity audit\n\nEvery categorical block is projected to its nearest legal train-observed standardized one-hot prototype. Projection occurs before classifier evaluation, cost, and plausibility measurement. This enforces categorical legality only and does not impose protected-context compatibility.\n\n"+markdown_table(cat))

    lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(favorable[:,list(GEOMETRY)].numpy());rows=[];proxyrows=[];tensors={}
    for cid,(method,param,x,detail) in enumerate(candidates):
        x=project(x,groups,protos);vals=[];risks=[]
        with torch.no_grad():
            for j,pm in enumerate(proxies):pp=torch.sigmoid(pm(x));vals.append(float((pp>=.5).float().mean()));risks.append(float(torch.relu(torch.tensor(.5)-pp).mean()));proxyrows.append({"candidate_id":cid,"method":method,"native_parameter":param,"proxy":j,"validity":vals[-1],"risk":risks[-1]})
        rows.append({"candidate_id":cid,"method":method,"native_parameter":param,"detail":detail,**plaus(x,source,current,favorable,lof),"proxy_mean_validity":np.mean(vals),"proxy_worst_validity":np.min(vals),"calibration_risk":np.max(risks)});tensors[str(cid)]=x
    cand=pd.DataFrame(rows);cand["calibrated_flag"]=0;cand["selection_status"]="not_selected";cand.loc[cand.method=="base",["calibrated_flag","selection_status"]]=[1,"fixed"]
    for method in ("highconf","trex"):
        q=cand[cand.method==method];f=q[q.proxy_mean_validity>=.95];chosen=(f.sort_values("mean_l2_cost").iloc[0] if len(f) else q.sort_values(["proxy_mean_validity","mean_l2_cost"],ascending=[False,True]).iloc[0]);cand.loc[chosen.name,["calibrated_flag","selection_status"]]=[1,"properly_calibrated" if len(f) else "fallback_target_infeasible"]
    q=cand[cand.method=="lift_global"].sort_values("native_parameter");r0,r1=float(q.iloc[0].calibration_risk),float(q.iloc[-1].calibration_risk);score=(r0-q.calibration_risk)/max(r0-r1,1e-12)-q.native_parameter;cand.loc[score.idxmax(),["calibrated_flag","selection_status"]]=[1,"gain_over_transport"]
    pd.DataFrame(proxyrows).to_csv(out/"german_lift_calibration.csv",index=False)

    # 3. Future-model quality audit and exact culprit identification.
    sanity=[];future_models={}
    f0pred=valid(current,test_x);held_fav=test_x[(test_y==1)]
    for env in ENVS:
        for sev in SEVERITIES:
            for k in range(5):
                m=load_model(previous/"models"/f"{env}_s{sev:g}_k{k}.pt",train_x.shape[1]);future_models[(env,sev,k)]=m
                with torch.no_grad():score=torch.sigmoid(m(test_x));pred=score>=.5
                acc=float((pred==test_y.bool()).float().mean());auc=float(roc_auc_score(test_y.numpy(),score.numpy()));rec=float(recall_score(test_y.numpy(),pred.numpy()));rate=float(pred.float().mean());dis=float((pred!=f0pred).float().mean());support=float(valid(m,held_fav).float().mean());catastrophic=bool(acc<.55 or auc<.55 or rec<.5 or support<.5)
                sanity.append({"environment":env,"severity":sev,"seed":k,"accuracy":acc,"ROC_AUC":auc,"favorable_class_recall":rec,"favorable_prediction_rate":rate,"disagreement_with_f0":dis,"favorable_support_validity":support,"catastrophic":catastrophic})
    sanity=pd.DataFrame(sanity);sanity.to_csv(out/"german_future_model_sanity.csv",index=False)

    ev=[]
    for _,r in cand.iterrows():
        x=tensors[str(int(r.candidate_id))]
        for (env,sev,k),m in future_models.items():ev.append({"candidate_id":int(r.candidate_id),"method":r.method,"native_parameter":r.native_parameter,"environment":env,"severity":sev,"seed":k,"validity":float(valid(m,x).float().mean())})
    ev=pd.DataFrame(ev);ev.to_csv(out/"german_future_evaluation.csv",index=False)
    per=ev.groupby(["candidate_id","environment"],as_index=False).agg(mean=("validity","mean"),worst=("validity","min"));w=per.pivot(index="candidate_id",columns="environment",values="worst");cand["mean_future_validity"]=cand.candidate_id.map(ev.groupby("candidate_id").validity.mean());cand["common_global_worst"]=cand.candidate_id.map(w.min(axis=1));
    for env in ENVS:cand[f"worst__{env}"]=cand.candidate_id.map(w[env])
    # Meaningful in/near-range excludes catastrophic models and uses prior shift audit ratios <=1.
    audit=pd.read_csv(previous/"german_shift_scale_audit.csv")[["environment","severity","future_model_id","relative_shift","range_group"]].rename(columns={"future_model_id":"seed"});ev2=ev.merge(audit,on=["environment","severity","seed"]).merge(sanity[["environment","severity","seed","catastrophic"]],on=["environment","severity","seed"]);meaning=ev2[(ev2.relative_shift<=1)&(~ev2.catastrophic)];cand["in_range_mean_future_validity"]=cand.candidate_id.map(meaning.groupby("candidate_id").validity.mean());cand["in_range_worst_future_validity"]=cand.candidate_id.map(meaning.groupby("candidate_id").validity.min());cand.to_csv(out/"german_candidate_frontier_audited.csv",index=False)
    selected=cand[cand.calibrated_flag==1]
    # Models attaining selected-method minima.
    culprit=[]
    for _,r in selected.iterrows():
        qev=ev[ev.candidate_id==r.candidate_id];mn=qev.validity.min()
        for _,z in qev[qev.validity==mn].iterrows():culprit.append({"method":r.method,"minimum_cf_validity":mn,"environment":z.environment,"severity":z.severity,"seed":z.seed})
    culprit=pd.DataFrame(culprit).merge(sanity,on=["environment","severity","seed"],how="left");culprit.to_csv(out/"german_worst_model_culprits.csv",index=False)
    (out/"german_future_model_sanity.md").write_text("# German future-model sanity audit\n\nCatastrophic is predefined as accuracy < .55, AUC < .55, favorable recall < .50, or favorable-support validity < .50. Models are retained but separated from headline in/near-range interpretation.\n\n## Flagged models\n\n"+markdown_table(sanity[sanity.catastrophic])+"\n\n## Exact selected-CF worst-model culprits\n\n"+markdown_table(culprit))
    report=["# German Credit strict audited rerun","","## Selected points after equal-budget solver and categorical realization","",markdown_table(selected[["method","native_parameter","selection_status","current_validity","mean_future_validity","common_global_worst","in_range_mean_future_validity","in_range_worst_future_validity","worst__generic_population","worst__target_side","worst__mixed","mean_l2_cost","mean_nn_target_distance","mean_lof","population_target_distance"]]),"","## Audit conclusions","",f"HighConf p=.95 is cheaper than Base p=.50 for {cheaper:.1%} of paired sources after identical-budget optimization.",f"Categorical projection leaves {int(cat.invalid_categorical_count.sum())} invalid evaluated CFs across the audited populations.",f"{int(sanity.catastrophic.sum())} of {len(sanity)} future models meet the predefined catastrophic criterion.","","Method-level conclusions should use the audited in-range columns and environment-specific results; the unfiltered common worst is retained only for transparency."]
    (out/"german_credit_strict_audit_report.md").write_text("\n".join(report));torch.save(tensors,out/"candidate_counterfactuals_audited.pt")
    print(json.dumps({"highconf95_cheaper_fraction":cheaper,"invalid_after_projection":int(cat.invalid_categorical_count.sum()),"catastrophic_models":int(sanity.catastrophic.sum()),"selected":selected[["method","native_parameter","current_validity","mean_l2_cost","mean_future_validity","common_global_worst","in_range_worst_future_validity"]].to_dict("records")},indent=2))

if __name__=="__main__":main()
