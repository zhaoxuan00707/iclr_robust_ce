"""Controlled stability-weighted target-anchor ablation for HELOC and Diabetes."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor

from .heloc_benchmark_milestone import bootstrap_budget, valid
from .heloc_debug_diagnostics import markdown_table, w2_sinkhorn
from .heloc_method_bootstrap import calibration_stress
from ..data.tabular import load_tabular_three_way
from ..models.tabular import TabularMLP, train_model


ANCHORS = ("LIFT-UniformAnchor", "LIFT-WorstScoreAnchor", "LIFT-MarginStableAnchor")


def load_model(path: Path, dim: int) -> TabularMLP:
    model = TabularMLP(dim, [64, 32])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def weighted_sinkhorn(cost: torch.Tensor, target_weight: torch.Tensor, reg: float = .1,
                      iterations: int = 1000, tol: float = 1e-7) -> torch.Tensor:
    """Entropic balanced OT with a uniform source and prescribed target marginal."""
    n = len(cost)
    a = torch.full((n,), 1. / n, dtype=cost.dtype, device=cost.device)
    b = target_weight.to(cost).clamp_min(1e-30)
    b = b / b.sum()
    positive = cost[cost > 0]
    scale = positive.median() if positive.numel() else cost.new_tensor(1.)
    log_k = -cost / scale.clamp_min(1e-12) / reg
    log_a, log_b = a.log(), b.log()
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for step in range(iterations):
        log_u = log_a - torch.logsumexp(log_k + log_v[None], dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u[:, None], dim=0)
        if step % 20 == 19:
            p = torch.exp(log_u[:, None] + log_k + log_v[None])
            if max((p.sum(1)-a).abs().max(), (p.sum(0)-b).abs().max()) < tol:
                break
    p = torch.exp(log_u[:, None] + log_k + log_v[None])
    return p / p.sum().clamp_min(1e-30)


def reconstruct(dataset: str, data_home: str, root: Path):
    if dataset == "heloc":
        cfg = {"source":"openml", "openml_name":"heloc", "openml_version":1,
               "missing_values":[-7,-8,-9]}
        d = load_tabular_three_way(cfg, data_home, .6, .2, 0)
        train_x, train_y, cal_x, cal_y, test_x, test_y = (
            d[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y"))
        current = load_model(root/"models/current.pt", train_x.shape[1])
        cf = torch.load(root/"counterfactuals.pt", map_location="cpu", weights_only=True)
        source, source_ids, base = cf["sources"], cf["source_ids"].numpy(), cf["base_cf"]
        bcal = json.loads((root/"metadata.json").read_text())["b_cal"]
        proxies = []
        for m in range(5):
            stressed, _ = calibration_stress(cal_x, bcal, 10_000+m)
            proxy, _ = train_model(torch.cat((train_x,stressed)), torch.cat((train_y,cal_y)),
                                   test_x,test_y,[64,32],30,1e-3,1000+m)
            proxies.append(proxy)
        prior=Path("/mnt/big_disk/xu.zhao/robust_ce_results/heloc_prior_balanced_audit_seed0")
        audit=pd.read_csv(prior/"shift_scale_audit.csv");future=[]
        for r in audit.itertuples():
            future.append((r.environment,float(r.severity),int(r.future_model_id),r.range_group,
                           load_model(prior/"models"/f"{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt",train_x.shape[1])))
        baseline=pd.read_csv(prior/"common_candidate_frontier.csv")
        # Use the corrected extended T-Rex deployment point (tau=.90, lr=.02,
        # max_steps=200), reconstructed from its frozen trajectory.
        extroot=Path("/mnt/big_disk/xu.zhao/robust_ce_results/heloc_fairness_diagnostic_v3_seed0")
        trajectory=torch.load(extroot/"trex_trajectories.pt",map_location="cpu",weights_only=True)["lr_0.02"]
        t,s=trajectory["trajectory"],trajectory["scores"][:200];ok=s>=.9;idx=ok.float().argmax(0);idx[~ok.any(0)]=200
        trex=t[idx,torch.arange(len(base))]
        selected=baseline[(baseline.method=="trex")&(baseline.calibrated_flag==1)].iloc[0]
        trex=torch.load(prior/"candidate_counterfactuals.pt",map_location="cpu",weights_only=True)[str(int(selected.candidate_id))]
        return train_x,train_y,source,source_ids,base,current,proxies,future,baseline,trex
    cfg = {"source":"openml","openml_id":37,"positive_label":"tested_negative"}
    d = load_tabular_three_way(cfg,data_home,.6,.2,0)
    train_x,train_y,cal_x,cal_y,test_x,test_y = (
        d[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y"))
    current = load_model(root/"models/current.pt",train_x.shape[1])
    with torch.no_grad():
        neg=(torch.sigmoid(current(test_x))<.5).nonzero().flatten()
        loc=neg[torch.randperm(len(neg),generator=torch.Generator().manual_seed(2))[:min(128,len(neg))]]
    source, source_ids = test_x[loc], np.asarray(d["test_ids"])[loc.numpy()]
    frontier=pd.read_csv(root/"diabetes_candidate_frontier.csv")
    tensors=torch.load(root/"diabetes_candidate_counterfactuals.pt",map_location="cpu",weights_only=True)
    base=tensors[str(int(frontier[(frontier.method=="base")].iloc[0].candidate_id))]
    bcal=float(pd.read_csv(root/"diabetes_calibration_stresses.csv").b_cal.iloc[0])
    proxies=[]
    for m in range(5):
        stressed,_=calibration_stress(cal_x,bcal,10_000+m)
        proxy,_=train_model(torch.cat((train_x,stressed)),torch.cat((train_y,cal_y)),
                            test_x,test_y,[64,32],40,1e-3,1000+m)
        proxies.append(proxy)
    audit=pd.read_csv(root/"diabetes_shift_scale_audit.csv")
    future=[]
    for r in audit.itertuples():
        p=root/"models"/f"{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt"
        future.append((r.environment,float(r.severity),int(r.future_model_id),r.range_group,
                       load_model(p,train_x.shape[1])))
    selected=frontier[(frontier.method=="trex")&(frontier.calibrated_flag==1)].iloc[0]
    trex=tensors[str(int(selected.candidate_id))]
    return train_x,train_y,source,source_ids,base,current,proxies,future,frontier,trex


def anchor_weights(scores: torch.Tensor, tau: float=.5):
    worst=scores.min(1).values; mean=scores.mean(1)
    raw={ANCHORS[0]:torch.ones_like(worst), ANCHORS[1]:worst.clamp(0,1),
         ANCHORS[2]:torch.relu(worst-tau)+1e-6}
    return worst,mean,{k:v/v.sum() for k,v in raw.items()}


def evaluate_dataset(dataset: str, data_home: str, root: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    train_x,train_y,source,source_ids,base,current,proxies,futures,baseline,trex = reconstruct(dataset,data_home,root)
    favorable=train_x[valid(current,train_x)]
    with torch.no_grad():
        current_scores=torch.sigmoid(current(favorable))
        proxy_scores=torch.stack([torch.sigmoid(m(favorable)) for m in proxies],1)
    worst,mean,weights=anchor_weights(proxy_scores)
    score_rows=[]
    for j in range(len(favorable)):
        score_rows.append({"dataset":dataset,"target_id":j,"current_model_score":float(current_scores[j]),
            "worst_proxy_score":float(worst[j]),"mean_proxy_score":float(mean[j]),
            "worst_proxy_margin":float(worst[j]-.5),"uniform_weight":float(weights[ANCHORS[0]][j]),
            "worst_score_weight":float(weights[ANCHORS[1]][j]),
            "margin_stability_weight":float(weights[ANCHORS[2]][j])})
    pd.DataFrame(score_rows).to_csv(out/"target_anchor_stability_scores.csv",index=False)
    weight_rows=[]
    for name,w in weights.items():
        top=max(1,int(np.ceil(.1*len(w))));order=torch.argsort(w,descending=True)
        weight_rows.append({"dataset":dataset,"anchor_type":name,"ESS":float(1/(w.square().sum())),
            "maximum_weight":float(w.max()),"median_weight":float(w.median()),
            "top_10pct_mass":float(w[order[:top]].sum()),"mass_on_worst_score_below_tau":float(w[worst<.5].sum()),
            "fraction_positive_margin":float((worst>.5).float().mean()),
            "fraction_epsilon_only":float((worst<=.5).float().mean()) if name==ANCHORS[2] else np.nan})
    pd.DataFrame(weight_rows).to_csv(out/"target_anchor_weight_diagnostic.csv",index=False)

    ground=torch.cdist(base,favorable).square(); couplings={};destinations={};coupling_rows=[];ot_rows=[]
    for name,w in weights.items():
        plan=weighted_sinkhorn(ground,w);couplings[name]=plan
        destination=(plan/plan.sum(1,keepdim=True).clamp_min(1e-30))@favorable;destinations[name]=destination
        transported_worst=float((plan*worst[None]).sum());transported_current=float((plan*current_scores[None]).sum())
        ot_rows.append({"dataset":dataset,"anchor_type":name,"mean_transported_worst_proxy_score":transported_worst,
                        "mean_transported_current_score":transported_current,"mean_source_target_ot_distance":float(torch.sqrt((plan*ground).sum()))})
        nz=(plan>1e-12).nonzero()
        for i,j in nz.tolist(): coupling_rows.append({"dataset":dataset,"anchor_type":name,"source_id":int(source_ids[i]),
            "target_id":j,"transported_mass":float(plan[i,j]),"source_target_distance":float(torch.sqrt(ground[i,j])),
            "target_worst_proxy_score":float(worst[j]),"target_weight":float(w[j])})
    pd.DataFrame(coupling_rows).to_csv(out/"weighted_anchor_ot_coupling.csv",index=False)

    lof=LocalOutlierFactor(n_neighbors=min(20,len(favorable)-1),novelty=True).fit(favorable.numpy())
    path_rows=[];proxy_rows=[];eval_rows=[];tensors={}
    for name,destination in destinations.items():
        d2=float(torch.sqrt((couplings[name]*ground).sum()))
        for lam in np.linspace(0,1,101):
            lam=round(float(lam),2);z=(1-lam)*base+lam*destination;tensors[(name,lam)]=z
            vals=[];risks=[]
            with torch.no_grad():
                for m,proxy in enumerate(proxies):
                    p=torch.sigmoid(proxy(z));v=float((p>=.5).float().mean());risk=float(torch.relu(torch.tensor(.5)-p).mean());vals.append(v);risks.append(risk)
                    proxy_rows.append({"dataset":dataset,"anchor_type":name,"lambda":lam,"proxy_model_id":m,"validity":v,"risk":risk})
            l2=torch.linalg.vector_norm(z-source,dim=1);nn=torch.cdist(z,favorable).min(1).values
            row={"dataset":dataset,"anchor_type":name,"lambda":lam,"current_validity":float(valid(current,z).float().mean()),
                 "calibration_risk":max(risks),"proxy_worst_validity":min(vals),"proxy_mean_validity":float(np.mean(vals)),
                 "mean_L2_cost":float(l2.mean()),"median_L2_cost":float(l2.median()),"p90_L2_cost":float(torch.quantile(l2,.9)),
                 "NN_target_distance":float(nn.mean()),"LOF":float(lof.score_samples(z.numpy()).mean()),
                 "population_target_distance":w2_sinkhorn(z,favorable),
                 "W2_to_PCF":float(torch.sqrt(((z-base).square().sum(1)).mean())),
                 "W2_to_weighted_target_anchor":(1-lam)*d2}
            vals_future=[]
            for env,sev,k,rg,fm in futures:
                v=float(valid(fm,z).float().mean());vals_future.append(v)
                eval_rows.append({"dataset":dataset,"anchor_type":name,"lambda":lam,"environment":env,"severity":sev,"future_model_id":k,"range_group":rg,"validity":v})
            ev=pd.DataFrame(eval_rows[-len(futures):]);row["mean_future_validity"]=float(np.mean(vals_future))
            for rg in ("in_range","near_range"):
                q=ev[ev.range_group==rg];row[f"{rg}_worst_validity"]=float(q.validity.min()) if len(q) else np.nan
            for env in ev.environment.unique():row[f"worst__{env}"]=float(ev[ev.environment==env].validity.min())
            tq=ev[(ev.environment=="target_side_shift")&(ev.range_group.isin(["in_range","near_range"]))]
            row["target_side_in_near_worst_validity"]=float(tq.validity.min()) if len(tq) else np.nan
            path_rows.append(row)
    path=pd.DataFrame(path_rows);proxy_df=pd.DataFrame(proxy_rows);ev=pd.DataFrame(eval_rows)
    selections=[]
    for name,g in path.groupby("anchor_type"):
        g=g.sort_values("lambda");r0,r1=float(g.iloc[0].calibration_risk),float(g.iloc[-1].calibration_risk)
        gain=(r0-g.calibration_risk)/max(r0-r1,1e-12)-g["lambda"];idx=gain.idxmax();path.loc[idx,"calibrated_flag"]=1
        chosen=path.loc[idx]; wr=next(x for x in weight_rows if x["anchor_type"]==name)
        selections.append({"dataset":dataset,"anchor_type":name,"selected_lambda":chosen["lambda"],"calibration_risk":chosen.calibration_risk,
                           "current_validity":chosen.current_validity,"mean_L2_cost":chosen.mean_L2_cost,
                           "proxy_worst_validity":chosen.proxy_worst_validity,"ESS":wr["ESS"]})
    path["calibrated_flag"]=path.calibrated_flag.fillna(0).astype(int)
    diagnostic=[]
    for name,g in path.groupby("anchor_type"):
        points={"calibrated":g[g.calibrated_flag==1].iloc[0]}
        for floor,label in ((.95,"lambda_95"),(.99,"lambda_99")):
            q=g[g.current_validity>=floor];points[label]=q.sort_values("lambda").iloc[-1] if len(q) else None
        for label,r in points.items():
            if r is not None:diagnostic.append({"dataset":dataset,"anchor_type":name,"point":label,**r.to_dict()})
    path.to_csv(out/"stable_anchor_lift_path.csv",index=False);proxy_df.to_csv(out/"stable_anchor_proxy_scores.csv",index=False)
    pd.DataFrame(eval_rows).to_csv(out/"stable_anchor_future_evaluation.csv",index=False)
    pd.DataFrame(selections).to_csv(out/"stable_anchor_calibration_results.csv",index=False)
    pd.DataFrame(diagnostic).to_csv(out/"stable_anchor_validity_points.csv",index=False)
    pd.DataFrame(ot_rows).to_csv(out/"weighted_anchor_ot_diagnostic.csv",index=False)
    torch.save({f"{k[0]}__{k[1]:.2f}":v for k,v in tensors.items()},out/"stable_anchor_counterfactuals.pt")

    # Directions relative to the frozen selected T-Rex point.
    directions=[];td=trex-base
    for name in (ANCHORS[0],ANCHORS[2]):
        ld=destinations[name]-base;cos=torch.nn.functional.cosine_similarity(ld,td,dim=1,eps=1e-12)
        for i,c in enumerate(cos):directions.append({"dataset":dataset,"comparison":f"{name}_vs_T-Rex","source_id":int(source_ids[i]),"cosine":float(c)})
    a=destinations[ANCHORS[0]]-base;b=destinations[ANCHORS[2]]-base
    for i,c in enumerate(torch.nn.functional.cosine_similarity(a,b,dim=1,eps=1e-12)):
        directions.append({"dataset":dataset,"comparison":"UniformAnchor_vs_MarginStableAnchor","source_id":int(source_ids[i]),"cosine":float(c)})
    pd.DataFrame(directions).to_csv(out/"stable_anchor_direction_diagnostic.csv",index=False)

    # Combine old baseline frontier with anchor paths at common observed L2 budgets.
    old=[]
    if dataset=="diabetes":
        for _,r in baseline.iterrows():
            old.append({"method":{"trex":"T-Rex","highconf":"HighConf"}.get(r.method,r.method),"parameter":r.native_parameter,
                        "mean_L2_cost":r.mean_l2_cost,"current_validity":r.current_validity,"mean_future_validity":r.mean_future_validity,
                        "in_range_worst_validity":r.get("common_in_range_worst",np.nan),"near_range_worst_validity":r.get("common_near_range_worst",np.nan),
                        "NN_target_distance":r.mean_nn_target_distance,"LOF":r.mean_lof,"population_target_distance":r.population_target_distance,
                        **{f"worst__{e}":r.get(f"worst__{e}",np.nan) for e in ("generic_population_shift","target_side_shift","mixed_shift")}})
    else:
        for _,r in baseline[baseline.method.isin(["trex","highconf"])].iterrows():
            old.append({"method":{"trex":"T-Rex","highconf":"HighConf"}[r.method],"parameter":r.native_parameter,"mean_L2_cost":r.mean_l2_cost,
                        "current_validity":r.current_validity,"mean_future_validity":r.get("common_mean_future_validity",r.get("mean_future_validity",np.nan)),
                        "in_range_worst_validity":r.get("common_in_range_worst",np.nan),"near_range_worst_validity":r.get("common_near_range_worst",np.nan),
                        "NN_target_distance":r.mean_nn_target_distance,"LOF":r.mean_lof,"population_target_distance":r.population_target_distance,
                        "worst__generic_population_shift":r.get("worst__generic_population_shift",np.nan),
                        "worst__target_side_shift":r.get("worst__target_side_shift",np.nan),"worst__mixed_shift":r.get("worst__mixed_shift",np.nan)})
    compare=pd.concat([pd.DataFrame(old),path.rename(columns={"anchor_type":"method","lambda":"parameter"})],ignore_index=True,sort=False)
    swept=["T-Rex","HighConf",*ANCHORS];lo=max(compare[compare.method==m].mean_L2_cost.min() for m in swept);hi=min(compare[compare.method==m].mean_L2_cost.max() for m in swept)
    matched=[]
    for budget in np.linspace(lo,hi,15) if hi>=lo else []:
        for method in swept:
            q=compare[(compare.method==method)&(compare.mean_L2_cost<=budget+1e-9)]
            if q.empty:continue
            metric="near_range_worst_validity";r=q.sort_values([metric,"mean_L2_cost"],ascending=[False,True]).iloc[0]
            matched.append({"dataset":dataset,"cost_budget":budget,"method":method,"native_parameter":r.parameter,"actual_cost":r.mean_L2_cost,
                            "mean_future_validity":r.mean_future_validity,"in_range_worst_validity":r.get("in_range_worst_validity",np.nan),
                            "near_range_worst_validity":r.get("near_range_worst_validity",np.nan),"current_validity":r.current_validity,
                            "NN_target_distance":r.NN_target_distance,"LOF":r.LOF,"population_target_distance":r.population_target_distance,
                            **{c:r.get(c,np.nan) for c in compare.columns if c.startswith("worst__")}})
    pd.DataFrame(matched).to_csv(out/"stable_anchor_matched_cost.csv",index=False)

    # Plausibility deltas at calibrated, lambda95, and matched-cost selections.
    diag=pd.DataFrame(diagnostic);deltas=[]
    for point in ("calibrated","lambda_95"):
        q=diag[diag.point==point];u=q[q.anchor_type==ANCHORS[0]]
        if u.empty:continue
        u=u.iloc[0]
        for _,r in q.iterrows():deltas.append({"dataset":dataset,"comparison_point":point,"anchor_type":r.anchor_type,
            "Delta_NN_target_distance":r.NN_target_distance-u.NN_target_distance,"Delta_LOF":r.LOF-u.LOF,
            "Delta_population_target_distance":r.population_target_distance-u.population_target_distance})
    pd.DataFrame(deltas).to_csv(out/"stable_anchor_plausibility_retention.csv",index=False)
    return path,pd.DataFrame(selections),pd.DataFrame(weight_rows),pd.DataFrame(directions),pd.DataFrame(matched),pd.DataFrame(ot_rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-home");ap.add_argument("--heloc-data-home");ap.add_argument("--diabetes-data-home");ap.add_argument("--heloc-root",required=True);ap.add_argument("--diabetes-root",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True);all_results=[]
    homes={"heloc":a.heloc_data_home or a.data_home,"diabetes":a.diabetes_data_home or a.data_home}
    for dataset,root in (("heloc",Path(a.heloc_root)),("diabetes",Path(a.diabetes_root))):
        result=evaluate_dataset(dataset,homes[dataset],root,output/dataset);all_results.append((dataset,*result))
    weights=pd.concat([x[3] for x in all_results]);sels=pd.concat([x[2] for x in all_results]);dirs=pd.concat([x[4] for x in all_results]);matched=pd.concat([x[5] for x in all_results]);ots=pd.concat([x[6] for x in all_results])
    direction_summary=dirs.groupby(["dataset","comparison"],as_index=False).agg(mean=("cosine","mean"),median=("cosine","median"),percentage_positive=("cosine",lambda x:100*(x>0).mean()))
    selected=[]
    for dataset,path,*_ in all_results:
        selected.append(path[path.calibrated_flag==1])
    selected=pd.concat(selected)
    lines=["# LIFT stability-weighted target anchor v1","","This is a controlled target-marginal ablation. Splits, base CFs, proxy construction, future checkpoints, metrics, and baseline methods are unchanged. No final future model is used for anchor weights or lambda selection.","","## Anchor concentration","",markdown_table(weights),"","## Calibration-selected points","",markdown_table(selected[["dataset","anchor_type","lambda","current_validity","mean_L2_cost","mean_future_validity","in_range_worst_validity","near_range_worst_validity","NN_target_distance","LOF","population_target_distance"]]),"","## Transported target stability","",markdown_table(ots),"","## Direction diagnostics","",markdown_table(direction_summary),"","## Calibration audit","",markdown_table(sels),"","## Matched-cost results","",markdown_table(matched),"","## Interpretation","","The tables above are generated directly from the frozen evaluations. MarginStable uses the existing validity threshold tau=0.5 and epsilon=1e-6, with no temperature or test-tuned parameter. Anchor variants are reported as parallel ablations and are not selected using test results."]
    (output/"lift_stability_weighted_anchor_v1.md").write_text("\n".join(lines))
    print(json.dumps({"selected":sels.to_dict("records"),"direction_summary":direction_summary.to_dict("records")},indent=2))


if __name__=="__main__":main()
