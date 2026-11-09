"""HELOC robust-CF benchmark milestone: Base, HighConf, T-Rex, and LIFT."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from .adult_trex_comparison import trex_trajectory
from .heloc_method_bootstrap import calibration_stress
from ..counterfactuals.tabular import generate_wachter_ce
from ..data.tabular import Shift, apply_shift, load_tabular_three_way
from ..models.tabular import train_model
from ..ot import sinkhorn_coupling


def valid(model,x):
    with torch.no_grad(): return torch.sigmoid(model(x))>=.5


def exact_w2_support(x,y):
    c=torch.cdist(x,y).square().numpy();r,k=linear_sum_assignment(c)
    return float(np.sqrt(c[r,k].mean()))


def bootstrap_budget(cal_x,B,support,seed):
    g=torch.Generator().manual_seed(seed);x=cal_x[torch.randperm(len(cal_x),generator=g)[:support]]
    rng=np.random.default_rng(seed+1); vals=[]
    for _ in range(B):
        idx=torch.as_tensor(rng.integers(0,len(x),len(x)));vals.append(exact_w2_support(x[idx],x))
    return float(np.quantile(vals,.95)),vals


def transport_destination(base,target):
    cost=torch.cdist(base,target).square()
    plan=sinkhorn_coupling(cost,regularization=.1,iterations=200)
    weights=plan/plan.sum(1,keepdim=True).clamp_min(1e-12)
    return weights@target


def calibration_risks(models,candidates):
    rows=[]
    for name,param,x in candidates:
        losses=[];validities=[]
        with torch.no_grad():
            for model in models:
                p=torch.sigmoid(model(x));losses.append(float(torch.relu(torch.tensor(.5)-p).mean()));validities.append(float((p>=.5).float().mean()))
        rows.append({"method":name,"parameter":param,"risk":max(losses),"proxy_mean_validity":float(np.mean(validities)),"proxy_worst_validity":min(validities)})
    return rows


def pick_min_cost_feasible(rows,candidates,floor=.95):
    lookup={(m,p):x for m,p,x in candidates}; eligible=[r for r in rows if r["proxy_mean_validity"]>=floor]
    pool=eligible or rows
    return min(pool,key=lambda r:float(torch.linalg.vector_norm(lookup[(r['method'],r['parameter'])],ord=2,dim=1).mean()))


def make_future_data(family,severity,k,train_x,train_y,test_x,test_y,seed):
    if severity==0:return train_x,train_y,test_x,test_y
    if family=="multiplicity":
        g=torch.Generator().manual_seed(seed)
        # Fixed permutation across severity gives nested leave-p-out sets.
        keep=int(len(train_x)*(1-.05*severity));idx=torch.randperm(len(train_x),generator=g)[:keep]
        return train_x[idx],train_y[idx],test_x,test_y
    kind=("mean_shift","scale_shift","adverse_tail")[k%3]
    strength={"mean_shift":.15*severity,"scale_shift":1+.15*severity,"adverse_tail":.25*severity}[kind]
    sx,sy=apply_shift(train_x,train_y,Shift(kind,strength),seed);tx,ty=apply_shift(test_x,test_y,Shift(kind,strength),seed+1)
    if family=="mixed":
        g=torch.Generator().manual_seed(seed+2);keep=int(len(sx)*(1-.05*severity));idx=torch.randperm(len(sx),generator=g)[:keep];sx,sy=sx[idx],sy[idx]
    return sx,sy,tx,ty


def main():
    p=argparse.ArgumentParser();p.add_argument("--data-home",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--epochs",type=int,default=30);p.add_argument("--sources",type=int,default=128)
    p.add_argument("--K",type=int,default=5);p.add_argument("--B",type=int,default=20);p.add_argument("--M",type=int,default=5)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/"models").mkdir(exist_ok=True)
    cfg={"source":"openml","openml_name":"heloc","openml_version":1,"missing_values":[-7,-8,-9]}
    d=load_tabular_three_way(cfg,args.data_home,.6,.2,args.seed)
    train_x,train_y,cal_x,cal_y,test_x,test_y=(d[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y"))
    model,acc=train_model(train_x,train_y,test_x,test_y,[64,32],args.epochs,1e-3,args.seed);torch.save(model.state_dict(),out/"models/current.pt")
    with torch.no_grad():score=torch.sigmoid(model(test_x));auc=roc_auc_score(test_y.numpy(),score.numpy())
    negative=(score<.5).nonzero().flatten();g=torch.Generator().manual_seed(args.seed+2);local_ids=negative[torch.randperm(len(negative),generator=g)[:args.sources]]
    source=test_x[local_ids];source_ids=np.asarray(d["test_ids"])[local_ids.numpy()]
    base=generate_wachter_ce(model,source,.5,300,.03,.05,20.)
    favorable=train_x[valid(model,train_x)];destination=transport_destination(base,favorable)
    lift=[("lift",float(lam),(1-float(lam))*base+float(lam)*destination) for lam in np.linspace(0,1,21)]
    high=[("highconf",float(t),generate_wachter_ce(model,source,float(t),300,.03,.05,20.)) for t in (.55,.65,.75,.85,.95)]
    traj,tr_scores=trex_trajectory(model,base,1000,.1,.01,50,args.seed+50_000)
    trex=[]
    for tau in np.linspace(.5,.9,17):
        ok=tr_scores>=tau;idx=ok.float().argmax(0);idx[~ok.any(0)]=len(traj)-1
        trex.append(("trex",float(tau),traj[idx,torch.arange(len(base))]))
    candidates=[("base",0.,base)]+high+trex+lift

    bcal,boot=bootstrap_budget(cal_x,args.B,min(128,len(cal_x)),args.seed+4);proxies=[];stress_budgets=[]
    for m in range(args.M):
        stressed,budget=calibration_stress(cal_x,bcal,args.seed+10_000+m);stress_budgets.append(budget)
        proxy,_=train_model(torch.cat((train_x,stressed)),torch.cat((train_y,cal_y)),test_x,test_y,[64,32],args.epochs,1e-3,args.seed+1000+m);proxies.append(proxy)
    calrows=calibration_risks(proxies,candidates)
    bymethod={m:[r for r in calrows if r["method"]==m] for m in ("base","highconf","trex","lift")}
    selected={"base":0.}
    for m in ("highconf","trex"):selected[m]=pick_min_cost_feasible(bymethod[m],[x for x in candidates if x[0]==m])["parameter"]
    lr=sorted(bymethod["lift"],key=lambda r:r["parameter"]);r0,r1=lr[0]["risk"],lr[-1]["risk"];den=max(r0-r1,1e-12)
    selected["lift"]=max(lr,key=lambda r:(r0-r["risk"])/den-r["parameter"])["parameter"]
    with (out/"calibration_curve.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=calrows[0]);w.writeheader();w.writerows(calrows)

    futures=[]
    for family in ("multiplicity","population","mixed"):
        for severity in (0.,1.,2.):
            for k in range(args.K):
                # The same seed fixes directions/subsets across severity.
                sx,sy,tx,ty=make_future_data(family,severity,k,train_x,train_y,test_x,test_y,args.seed+20_000+1000*k)
                fm,facc=train_model(sx,sy,tx,ty,[64,32],args.epochs,1e-3,args.seed+30_000+10_000*("multiplicity","population","mixed").index(family)+100*int(severity)+k)
                key=f"{family}_s{severity:g}_k{k}";torch.save(fm.state_dict(),out/f"models/{key}.pt")
                disagreement=float((valid(fm,test_x)!=valid(model,test_x)).float().mean());futures.append((family,severity,k,fm,facc,disagreement));print(key,flush=True)

    lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(favorable.numpy());tidy=[];summary=[]
    for method,param,x in candidates:
        cur=valid(model,x);percost=torch.linalg.vector_norm(x-source,ord=2,dim=1);nearest=torch.cdist(x,favorable).min(1).values;lofs=lof.score_samples(x.numpy())
        for family,severity,k,fm,facc,disagreement in futures:
            fv=valid(fm,x)
            for i in range(len(x)):
                tidy.append({"dataset":"heloc","seed":args.seed,"method":method,"native_parameter":param,"calibrated":int(abs(param-selected[method])<1e-8),"environment":family,"severity":severity,"future_model_id":k,"source_id":int(source_ids[i]),"current_valid":int(cur[i]),"future_valid":int(fv[i]),"l2_cost":float(percost[i]),"nearest_target_distance":float(nearest[i]),"lof_score":float(lofs[i])})
        vals=[r for r in tidy if r["method"]==method and r["native_parameter"]==param]
        summary.append({"method":method,"native_parameter":param,"calibrated":int(abs(param-selected[method])<1e-8),"current_validity":float(cur.float().mean()),"mean_l2_cost":float(percost.mean()),"median_l2_cost":float(percost.median()),"p90_l2_cost":float(torch.quantile(percost,.9)),"mean_future_validity":float(np.mean([r["future_valid"] for r in vals]))})
    with (out/"results_tidy.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=tidy[0]);w.writeheader();w.writerows(tidy)
    with (out/"candidate_summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    torch.save({"source_ids":torch.as_tensor(source_ids),"sources":source,"base_cf":base,"destination":destination,"candidates":{f"{m}_{p:g}":x for m,p,x in candidates}},out/"counterfactuals.pt")
    np.savez(out/"split_ids.npz",train=d["train_ids"],calibration=d["cal_ids"],test=d["test_ids"],sources=source_ids)
    meta={"split":{"train":.6,"calibration":.2,"test":.2},"seed":args.seed,"K":args.K,"B":args.B,"M":args.M,"b_cal":bcal,"stress_budgets":stress_budgets,"selected":selected,"current":{"accuracy":acc,"auc":auc,"class_balance":float(train_y.mean()),"favorable_rate":float(valid(model,test_x).float().mean())},"no_future_used_for_calibration":True}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2))

    # Aggregate future validity and produce reproducible first-pass figures.
    import pandas as pd
    df=pd.DataFrame(tidy);agg=df.groupby(["method","native_parameter","calibrated","environment","severity","future_model_id"],as_index=False).agg(validity=("future_valid","mean"),cost=("l2_cost","mean"))
    env=agg.groupby(["method","native_parameter","calibrated","environment","severity"],as_index=False).agg(mean_validity=("validity","mean"),worst_validity=("validity","min"),cost=("cost","mean"));env.to_csv(out/"environment_summary.csv",index=False)
    chosen=env[env.calibrated==1]
    fig,ax=plt.subplots(figsize=(7,4));
    for m,gp in env.groupby("method"):ax.scatter(gp.cost,gp.worst_validity,s=15,alpha=.45,label=m)
    ax.set(xlabel="mean L2 cost",ylabel="worst future validity",title="HELOC robustness-cost candidates");ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/"A_pareto.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for (m,e),gp in chosen.groupby(["method","environment"]):ax.plot(gp.severity,gp.worst_validity,marker="o",label=f"{m}/{e}")
    ax.set(xlabel="severity",ylabel="worst future validity",title="Calibrated graceful degradation");ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/"B_severity.png",dpi=220);plt.close(fig)
    liftcal={r["parameter"]:r["risk"] for r in bymethod["lift"]};lift_env=env[env.method=="lift"].groupby("native_parameter",as_index=False).agg(worst=("worst_validity","mean"),cost=("cost","mean"))
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(lift_env.native_parameter,lift_env.cost,label="cost");ax.plot(lift_env.native_parameter,[liftcal[x] for x in lift_env.native_parameter],label="calibration risk");ax.plot(lift_env.native_parameter,lift_env.worst,label="future worst");ax.axvline(selected["lift"],ls="--",c="k");ax.legend();ax.grid(alpha=.25);ax.set_xlabel("lambda");fig.tight_layout();fig.savefig(out/"D_lift_diagnostic.png",dpi=220);plt.close(fig)
    print(json.dumps(meta,indent=2))

if __name__=="__main__":main()
