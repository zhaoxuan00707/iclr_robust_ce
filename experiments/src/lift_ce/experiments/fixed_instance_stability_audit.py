"""Fixed-problem-instance stability audit from corrected conditional attacks.

Uses completed per-(r, lambda, restart) artifacts only.  No model/data
regeneration or production selection is performed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("/home/xu.zhao/robust_ce_results/heloc_diabetes_wro_lambda_v2")
OUT=Path("/home/xu.zhao/robust_ce_results/fixed_instance_conditional_stability_v1")
INSTANCES={"heloc":1,"diabetes":0}
EPS=0.005

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=[]; summaries=[]
    for ds,seed in INSTANCES.items():
        sd=ROOT/ds/f"seed{seed}"
        path=sd/"lift_conditional_attack_evaluation.csv"
        if not path.exists():
            # Older corrected aggregates were written before the descriptive
            # filename was introduced; their contents are conditional after
            # the production fix.
            path=sd/"lift_pooled_attack_cross_evaluation.csv"
        d=pd.read_csv(path)
        d=d[np.isclose(d.attack_source_lambda,d.target_lambda)].copy()
        d=d.rename(columns={"r_over_b_ref":"r","target_lambda":"lambda",
          "true_hinge":"full_update_true_hinge","budget_fraction":"budget_fraction"})
        d["dataset"]=ds; d["instance_seed"]=seed; d["solver_seed"]=d.restart
        d["search_surrogate_value"]=np.nan; d["truncated_update_hinge"]=np.nan
        # Attach final search traces where available (same restart/target).
        for i,row in d.iterrows():
            tp=sd/"attacks"/"lift"/f"r{row.r:g}"/f"lambda{row['lambda']:.2f}"/f"restart{int(row.restart)}_trace.csv"
            if tp.exists():
                tr=pd.read_csv(tp)
                d.loc[i,"search_surrogate_value"]=float(tr.search_objective_value.iloc[-1])
                d.loc[i,"truncated_update_hinge"]=float(tr.search_time_true_hinge.iloc[-1])
        d.to_csv(OUT/f"{ds}_solver_seed_curves.csv",index=False)
        all_rows.append(d)
        for (r,solver_seed),g in d.groupby(["r","solver_seed"]):
            g=g.sort_values("lambda"); best=g.loc[g.full_update_true_hinge.idxmin()]
            m=float(best.full_update_true_hinge); near=g[g.full_update_true_hinge<=m+EPS]
            p0=float(g.loc[np.isclose(g['lambda'],0),'full_update_true_hinge'].iloc[0])
            p1=float(g.loc[np.isclose(g['lambda'],1),'full_update_true_hinge'].iloc[0])
            summaries.append({"dataset":ds,"instance_seed":seed,"r":float(r),"solver_seed":int(solver_seed),
             "lambda_hat":float(best['lambda']),"min_phi":m,"near_optimal_eps":EPS,
             "near_optimal_lambda_min":float(near['lambda'].min()),"near_optimal_lambda_max":float(near['lambda'].max()),
             "phi_0":p0,"phi_1":p1,"gap_to_lambda1":p1-m,
             "budget_fraction_at_min":float(best.budget_fraction),"selected_restart":int(best.restart)})
        # overlay and mean/std
        fig,ax=plt.subplots(figsize=(6,4)); piv=d.pivot_table(index="lambda",columns="solver_seed",values="full_update_true_hinge",aggfunc="first").sort_index()
        for c in piv: ax.plot(piv.index,piv[c],alpha=.45,label=f"solver seed {int(c)}")
        ax.plot(piv.index,piv.mean(1),color="black",lw=2,label="mean")
        ax.fill_between(piv.index,piv.mean(1)-piv.std(1,ddof=0),piv.mean(1)+piv.std(1,ddof=0),color="black",alpha=.15)
        ax.set(xlabel="lambda",ylabel="full-update true hinge",title=f"{ds.upper()} fixed instance, r curves"); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(OUT/f"{ds}_phi_solver_seeds.png",dpi=180); plt.close(fig)
    curves=pd.concat(all_rows,ignore_index=True); sums=pd.DataFrame(summaries)
    curves.to_csv(OUT/"all_solver_seed_curves.csv",index=False); sums.to_csv(OUT/"solver_seed_minima.csv",index=False)
    stats=(sums.groupby(["dataset","r"],as_index=False).agg(mean_lambda=("lambda_hat","mean"),median_lambda=("lambda_hat","median"),std_lambda=("lambda_hat","std"),min_lambda=("lambda_hat","min"),max_lambda=("lambda_hat","max"),fraction_lambda_1=("lambda_hat",lambda x:np.mean(np.isclose(x,1))),fraction_ge_0p9=("lambda_hat",lambda x:np.mean(x>=.9)),mean_gap_to_lambda1=("gap_to_lambda1","mean")))
    stats.to_csv(OUT/"solver_seed_stability_summary.csv",index=False)
    # Fixed-instance metadata confirms frozen objects; hashes are represented
    # by the single instance directory and source artifact paths.
    meta={"instances":INSTANCES,"replication_role":"adversarial-search/solver seeds (restarts), not data folds","epsilon":EPS,"conditional_filter":"attack_source_lambda == target_lambda","pooling_used_for_selection":False,"f0_cf_path_p1_bref_frozen":True,"future_shifts_used_for_selection":False,"lambda_grid":"existing production grid 0.00..1.00 step .05"}
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2))
    # Baseline/final comparison from fixed instance final shards.
    rows=[]
    for ds,seed in INSTANCES.items():
      sd=ROOT/ds/f"seed{seed}"
      for method in ("lift","trex","roar","proplace","ellice"):
        p=sd/f"{method}_final_results.csv"
        if p.exists():
          x=pd.read_csv(p); x=x[x.operating_point.isin(["robustness_only","lambda_0","lambda_1"])];
          if len(x): rows.append(x)
    if rows: pd.concat(rows,ignore_index=True).to_csv(OUT/"fixed_instance_baseline_comparison.csv",index=False)
    print(sums.to_string(index=False)); print("\\nSUMMARY\\n",stats.to_string(index=False)); print("\\nWrote",OUT)
if __name__=="__main__": main()
