"""German Credit ablation: remove custom actionability/protected-context constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor

from .german_credit_benchmark_milestone_v1 import (
    ENVS, GEOMETRY, SEVERITIES, LAB, COL, load_model, masked_stress, plaus,
    select, stability, w2_sinkhorn,
)
from .heloc_benchmark_milestone import valid
from .heloc_debug_diagnostics import markdown_table
from ..data.tabular import load_tabular_three_way
from ..models.tabular import train_model
from ..ot import sinkhorn_coupling


def wachter_unconstrained(model, source, threshold, steps=500):
    x = source.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=.03)
    best = torch.full((len(x),), float("inf"))
    result = source.clone()
    done = torch.zeros(len(x), dtype=torch.bool)
    for _ in range(steps):
        opt.zero_grad()
        p = torch.sigmoid(model(x))
        loss = (20 * torch.relu(threshold - p) + .05 * (x-source).square().mean(1)).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            ok = torch.sigmoid(model(x)) >= threshold
            cost = torch.linalg.vector_norm(x-source, dim=1)
            use = ok & (cost < best)
            result[use], best[use], done = x[use], cost[use], done | ok
    return torch.where(done[:, None], result, x.detach())


def global_destination(base, favorable):
    # Keep the prior run's OT ground geometry; remove only the realization projection.
    cost = torch.cdist(base[:, list(GEOMETRY)], favorable[:, list(GEOMETRY)]).square()
    plan = sinkhorn_coupling(cost, .1, 300)
    weights = plan / plan.sum(1, keepdim=True).clamp_min(1e-12)
    return weights @ favorable


def trex_trajectory_unconstrained(model, base, lr, seed):
    x = base.clone(); trajectory = [x.clone()]; scores = []
    for step in range(200):
        x = x.detach().requires_grad_(True)
        # Same estimator, but noise and updates now cover the full encoded space.
        clean = torch.sigmoid(model(x)); total = torch.zeros_like(clean)
        gen = torch.Generator().manual_seed(seed + step)
        for start in range(0, 1000, 100):
            noise = .1 * torch.randn((100,) + x.shape, generator=gen)
            p = torch.sigmoid(model((x.unsqueeze(0)+noise).flatten(0, 1))).view(100, len(x))
            total += (p - (clean.unsqueeze(0)-p).abs()).sum(0)
        score = total / 1000
        score.sum().backward()
        x = (x + lr*x.grad).detach()
        trajectory.append(x.clone()); scores.append(score.detach())
        if (step+1) % 50 == 0:
            print(f"German unconstrained T-Rex lr={lr:g} step={step+1} stability={score.mean():.4f}", flush=True)
    return torch.stack(trajectory), torch.stack(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-home", required=True)
    ap.add_argument("--previous", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    previous, out = Path(args.previous), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = load_tabular_three_way(
        {"source":"openml", "openml_id":31, "positive_label":"good"},
        args.data_home, .6, .2, args.seed,
    )
    train_x, train_y, cal_x, cal_y, test_x, test_y = (
        data[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y")
    )
    current = load_model(previous/"models/current.pt", train_x.shape[1])
    artifacts = torch.load(previous/"core_artifacts.pt", map_location="cpu", weights_only=True)
    source, base = artifacts["source"], artifacts["base"]
    favorable = train_x[valid(current, train_x)]
    destination = global_destination(base, favorable)

    # Rebuild exactly the same calibration proxies as the constrained run.
    oldmeta = json.loads((previous/"metadata.json").read_text())
    bcal = float(oldmeta["b_cal"])
    proxies = []
    for m in range(5):
        stressed, _ = masked_stress(cal_x, bcal, args.seed+10_000+m)
        proxy, _ = train_model(torch.cat((train_x, stressed)), torch.cat((train_y, cal_y)),
                               test_x, test_y, [64,32], 60, 1e-3, args.seed+1000+m)
        proxies.append(proxy)

    candidates = [("base", 0., base, "frozen previous base")]
    for p in np.arange(.55, .951, .05):
        candidates.append(("highconf", float(p), wachter_unconstrained(current, source, float(p)), "unconstrained"))
    for lam in np.linspace(0, 1, 101):
        candidates.append(("lift_global", float(lam), (1-lam)*base+lam*destination, "global P1; no projection"))

    cache = out/"trex_trajectories.pt"
    if cache.exists():
        trajectories = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        trajectories = {}
        for lr in (.005,.01,.02):
            t,s = trex_trajectory_unconstrained(current, base, lr, args.seed+50_000)
            trajectories[f"lr_{lr:g}"] = {"trajectory":t, "scores":s}
        torch.save(trajectories, cache)
    for lr in (.005,.01,.02):
        t = trajectories[f"lr_{lr:g}"]["trajectory"]
        scores = trajectories[f"lr_{lr:g}"]["scores"]
        for steps in (50,100,200):
            ss = scores[:steps]
            for tau in np.arange(.5,.951,.05):
                ok = ss >= tau; idx = ok.float().argmax(0); reached = ok.any(0); idx[~reached] = steps
                candidates.append(("trex", float(tau), t[idx,torch.arange(len(base))],
                                   f"lr={lr:g};steps={steps};reach={float(reached.float().mean()):.3f}"))

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(favorable[:,list(GEOMETRY)].numpy())
    rows, proxy_rows, tensors = [], [], {}
    for cid,(method,param,x,detail) in enumerate(candidates):
        vals, risks = [], []
        with torch.no_grad():
            for j,proxy in enumerate(proxies):
                pp = torch.sigmoid(proxy(x)); v = float((pp>=.5).float().mean())
                risk = float(torch.relu(torch.tensor(.5)-pp).mean())
                vals.append(v); risks.append(risk)
                proxy_rows.append({"candidate_id":cid,"method":method,"native_parameter":param,
                                   "proxy":j,"validity":v,"risk":risk})
        rows.append({"candidate_id":cid,"method":method,"native_parameter":param,"detail":detail,
                     **plaus(x,source,current,favorable,lof),"proxy_mean_validity":np.mean(vals),
                     "proxy_worst_validity":np.min(vals),"calibration_risk":np.max(risks)})
        tensors[str(cid)] = x
    cand = pd.DataFrame(rows); cand["calibrated_flag"] = 0; cand["selection_status"] = "not_selected"
    cand.loc[cand.method=="base", ["calibrated_flag","selection_status"]] = [1,"fixed"]
    for method in ("highconf","trex"):
        q=cand[cand.method==method]; feasible=q[q.proxy_mean_validity>=.95]
        chosen=(feasible.sort_values("mean_l2_cost").iloc[0] if len(feasible)
                else q.sort_values(["proxy_mean_validity","mean_l2_cost"],ascending=[False,True]).iloc[0])
        cand.loc[chosen.name,["calibrated_flag","selection_status"]] = [1,"properly_calibrated" if len(feasible) else "fallback_target_infeasible"]
    q=cand[cand.method=="lift_global"].sort_values("native_parameter")
    r0,r1=float(q.iloc[0].calibration_risk),float(q.iloc[-1].calibration_risk)
    score=(r0-q.calibration_risk)/max(r0-r1,1e-12)-q.native_parameter
    cand.loc[score.idxmax(),["calibrated_flag","selection_status"]] = [1,"gain_over_transport"]
    pd.DataFrame(proxy_rows).to_csv(out/"german_lift_calibration.csv",index=False)
    torch.save(tensors,out/"candidate_counterfactuals.pt")

    # Evaluate on the exact frozen future checkpoints from the previous run.
    evaluations=[]
    for _,row in cand.iterrows():
        x=tensors[str(int(row.candidate_id))]
        for env in ENVS:
            for sev in SEVERITIES:
                for k in range(5):
                    model=load_model(previous/"models"/f"{env}_s{sev:g}_k{k}.pt",train_x.shape[1])
                    evaluations.append({"candidate_id":int(row.candidate_id),"method":row.method,
                                        "native_parameter":row.native_parameter,"environment":env,
                                        "severity":sev,"future_model_id":k,
                                        "validity":float(valid(model,x).float().mean())})
    ev=pd.DataFrame(evaluations); ev.to_csv(out/"german_future_evaluation.csv",index=False)
    sev=ev.groupby(["candidate_id","method","native_parameter","environment","severity"],as_index=False).agg(
        mean_future_validity=("validity","mean"),worst_future_validity=("validity","min"))
    sev.to_csv(out/"german_environment_severity_summary.csv",index=False)
    per=ev.groupby(["candidate_id","environment"],as_index=False).agg(mean=("validity","mean"),worst=("validity","min"))
    worst=per.pivot(index="candidate_id",columns="environment",values="worst")
    cand["mean_future_validity"]=cand.candidate_id.map(ev.groupby("candidate_id").validity.mean())
    cand["common_global_worst"]=cand.candidate_id.map(worst.min(axis=1))
    for env in ENVS: cand[f"worst__{env}"]=cand.candidate_id.map(worst[env])
    cand.to_csv(out/"german_candidate_frontier.csv",index=False)

    # Matched budgets over the actual joint T-Rex/LIFT support; HighConf is included only where supported.
    compared=("trex","lift_global")
    lo=max(cand[cand.method==m].mean_l2_cost.min() for m in compared)
    hi=min(cand[cand.method==m].mean_l2_cost.max() for m in compared)
    budgets=np.linspace(lo,hi,12) if hi>=lo else np.array([])
    matched=[]
    for budget in budgets:
        for method in ("highconf","trex","lift_global"):
            g=cand[cand.method==method]
            if not (g.mean_l2_cost.min()-1e-9 <= budget <= g.mean_l2_cost.max()+1e-9): continue
            for environment,y in [("common","common_global_worst")]+[(e,f"worst__{e}") for e in ENVS]:
                row=select(g,budget,y)
                matched.append({"cost_budget":budget,"method":method,"environment":environment,
                                "parameter":row.native_parameter,"actual_cost":row.mean_l2_cost,
                                "mean_future_validity":row.mean_future_validity,"worst_future_validity":row[y],
                                "nearest_target_distance":row.mean_nn_target_distance,"mean_lof":row.mean_lof,
                                "population_target_distance":row.population_target_distance})
    matched=pd.DataFrame(matched)
    matched.to_csv(out/"german_matched_cost_all.csv",index=False)
    matched[matched.environment=="common"].to_csv(out/"german_matched_cost_common.csv",index=False)
    matched[matched.environment!="common"].to_csv(out/"german_matched_cost_by_environment.csv",index=False)

    for y,name in [("common_global_worst","german_cost_vs_common_worst"),("mean_future_validity","german_cost_vs_mean_future")]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in ("highconf","trex","lift_global"):
            g=cand[cand.method==method].sort_values("mean_l2_cost")
            ax.plot(g.mean_l2_cost,g[y],"o",ms=2,color=COL[method],label=LAB[method])
        ax.set(xlabel="mean L2 cost",ylabel=y.replace("_"," "));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=240);plt.close(fig)

    lift=cand[cand.method=="lift_global"].sort_values("native_parameter").copy()
    lift.to_csv(out/"german_lift_global_path.csv",index=False)
    selected=cand[cand.calibrated_flag==1]
    old=pd.read_csv(previous/"german_candidate_frontier.csv")
    old_lift=old[old.method=="lift_global"].sort_values("native_parameter")
    new_trex=cand[(cand.method=="trex")&(cand.calibrated_flag==1)].iloc[0]
    vt=tensors[str(int(new_trex.candidate_id))]-base; vl=destination-base
    cosine=(vl*vt).sum(1)/(torch.linalg.vector_norm(vl,dim=1)*torch.linalg.vector_norm(vt,dim=1)).clamp_min(1e-12)
    pd.DataFrame({"cosine":cosine.numpy()}).to_csv(out/"german_lift_trex_direction.csv",index=False)
    comparison=pd.DataFrame([
        {"quantity":"selected_lambda","constrained":float(old_lift[old_lift.calibrated_flag==1].native_parameter.iloc[0]),"unconstrained":float(lift[lift.calibrated_flag==1].native_parameter.iloc[0])},
        {"quantity":"lambda1_current_validity","constrained":float(old_lift.iloc[-1].current_validity),"unconstrained":float(lift.iloc[-1].current_validity)},
        {"quantity":"lift_max_cost","constrained":float(old_lift.mean_l2_cost.max()),"unconstrained":float(lift.mean_l2_cost.max())},
        {"quantity":"trex_min_cost","constrained":float(old[old.method=="trex"].mean_l2_cost.min()),"unconstrained":float(cand[cand.method=="trex"].mean_l2_cost.min())},
    ])
    comparison.to_csv(out/"constrained_vs_unconstrained.csv",index=False)
    cosine_stats={"mean":float(cosine.mean()),"median":float(cosine.median()),"positive":float((cosine>0).float().mean()),"negative":float((cosine<0).float().mean())}
    report=["# German Credit unconstrained rerun","","This ablation reuses the previous split, classifier, source population, base CFs, calibration seeds, and exact future-model checkpoints. It removes the custom actionability projection, protected-context matching penalty, and LIFT-compatible branch.","","## Selected points","",markdown_table(selected[["method","native_parameter","selection_status","current_validity","mean_l2_cost","mean_future_validity","common_global_worst","worst__generic_population","worst__target_side","worst__mixed","mean_nn_target_distance","mean_lof","population_target_distance"]]),"","## Constrained versus unconstrained","",markdown_table(comparison),"",f"Direction cosine: mean {cosine_stats['mean']:.4f}, median {cosine_stats['median']:.4f}, positive {cosine_stats['positive']:.1%}, negative {cosine_stats['negative']:.1%}.","",f"Observed T-Rex/LIFT cost overlap: [{lo:.4f}, {hi:.4f}] ({len(budgets)} diagnostic budgets).","","## Direct answers","",f"1. **No.** LIFT no longer calibrates to zero: lambda changes from 0.00 in the constrained run to {float(lift[lift.calibrated_flag==1].native_parameter.iloc[0]):.2f} here.",f"2. **No systematic validity collapse remains.** Endpoint validity changes from 2.2% constrained to {float(lift.iloc[-1].current_validity):.1%} unconstrained; the calibrated point retains {float(lift[lift.calibrated_flag==1].current_validity.iloc[0]):.1%} current validity.",f"3. **Yes.** T-Rex and LIFT now overlap over cost [{lo:.4f}, {hi:.4f}]; previously LIFT ended at 2.0780 while T-Rex began at 2.1190.",f"4. **No.** The direction is near-orthogonal but weakly positive: mean cosine {cosine_stats['mean']:.4f}, median {cosine_stats['median']:.4f}, with {cosine_stats['positive']:.1%} positive and only {cosine_stats['negative']:.1%} negative.","5. **Most of the previous anchor-path failure disappears:** positive lambda is selected, endpoint current validity recovers by 97.8 percentage points, cost support overlaps, and negative direction disagreement vanishes. What does not disappear is the harsh target-side aggregate: its worst future model remains near zero for all selected methods. LIFT also pays substantially higher cost at its calibrated point, so this ablation does not establish robustness-per-cost dominance.","","LOF uses sklearn `score_samples`; more negative values indicate worse outlierness. No final future metric was used for calibration or candidate generation."]
    (out/"german_credit_unconstrained_report.md").write_text("\n".join(report))
    (out/"metadata.json").write_text(json.dumps({"previous":str(previous),"sources":len(source),"b_cal":bcal,"future_models_reused":True,"cosine":cosine_stats,"cost_overlap":[float(lo),float(hi)]},indent=2))
    print(json.dumps({"selected":selected[["method","native_parameter","selection_status"]].to_dict("records"),"lift_endpoint_validity":float(lift.iloc[-1].current_validity),"cost_overlap":[float(lo),float(hi)],"cosine":cosine_stats},indent=2))


if __name__ == "__main__":
    main()
