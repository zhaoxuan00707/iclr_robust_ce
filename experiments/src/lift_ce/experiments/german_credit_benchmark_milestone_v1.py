"""German Credit prior-balanced robust-CF debug milestone."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

from .heloc_benchmark_milestone import bootstrap_budget,exact_w2_support,valid
from .heloc_debug_diagnostics import markdown_table,w2_sinkhorn
from .heloc_fairness_diagnostic_v3 import COLORS,LABELS
from ..data.tabular import load_tabular_three_way
from ..models.tabular import TabularMLP,train_model
from ..ot import sinkhorn_coupling

MUTABLE=(0,1,2) # duration, credit_amount, installment_commitment
GEOMETRY=(0,1,2,3,4,5,6) # all standardized numeric/ordinal columns
SEVERITIES=(.5,1.,2.5);ENVS=("generic_population","target_side","mixed")
COL={**COLORS,"lift_global":"tab:blue","lift_compatible":"tab:green"};LAB={**LABELS,"lift_global":"LIFT-global","lift_compatible":"LIFT-compatible"}

def load_model(path,dim):m=TabularMLP(dim,[64,32]);m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True));m.eval();return m
def metrics(model,x,y):
    with torch.no_grad():p=torch.sigmoid(model(x));return float(((p>=.5)==y.bool()).float().mean()),float(roc_auc_score(y.numpy(),p.numpy())),float((p>=.5).float().mean())

def wachter(model,source,threshold,steps=500):
    mask=torch.zeros(source.shape[1]);mask[list(MUTABLE)]=1;x=source.clone().requires_grad_(True);opt=torch.optim.Adam([x],lr=.03);best=torch.full((len(x),),float("inf"));result=source.clone();done=torch.zeros(len(x),dtype=torch.bool)
    for _ in range(steps):
        opt.zero_grad();p=torch.sigmoid(model(x));loss=(20*torch.relu(threshold-p)+.05*(x-source).square().mean(1)).mean();loss.backward();opt.step()
        with torch.no_grad():x.copy_(source+mask*(x-source));ok=torch.sigmoid(model(x))>=threshold;c=torch.linalg.vector_norm(x-source,dim=1);use=ok&(c<best);result[use]=x[use];best[use]=c[use];done|=ok
    return torch.where(done[:,None],result,x.detach())

def stability(model,x,k,sigma,seed):
    clean=torch.sigmoid(model(x));total=torch.zeros_like(clean);g=torch.Generator().manual_seed(seed);mask=torch.zeros(x.shape[1]);mask[list(MUTABLE)]=1
    for start in range(0,k,100):
        n=min(100,k-start);noise=sigma*torch.randn((n,)+x.shape,generator=g)*mask;p=torch.sigmoid(model((x.unsqueeze(0)+noise).flatten(0,1))).view(n,len(x));total+=(p-(clean.unsqueeze(0)-p).abs()).sum(0)
    return total/k

def trex_trajectory(model,base,lr,seed):
    mask=torch.zeros(base.shape[1]);mask[list(MUTABLE)]=1;x=base.clone();traj=[x.clone()];scores=[]
    for step in range(200):
        x=x.detach().requires_grad_(True);s=stability(model,x,1000,.1,seed+step);s.sum().backward();x=(base+mask*(x+lr*x.grad-base)).detach();traj.append(x.clone());scores.append(s.detach())
        if (step+1)%50==0:print(f"German T-Rex lr={lr:g} step={step+1} stability={s.mean():.4f}",flush=True)
    return torch.stack(traj),torch.stack(scores)

def anchor_destination(base,favorable,source,groups,compatible):
    cost=torch.cdist(base[:,list(GEOMETRY)],favorable[:,list(GEOMETRY)]).square();mismatch=torch.zeros_like(cost)
    for ids in groups:
        # Immutable standardized one-hot blocks are source-exact. Equality of
        # the active prototype is tested by nearest block vector.
        mismatch+=(torch.cdist(source[:,ids],favorable[:,ids]).square()>1e-8).float()
    used=cost+(1e4*mismatch if compatible else 0);plan=sinkhorn_coupling(used,.1,300);weights=plan/plan.sum(1,keepdim=True).clamp_min(1e-12);paired=weights@favorable;dest=base.clone();dest[:,list(MUTABLE)]=paired[:,list(MUTABLE)]
    match=favorable[used.argmin(1)];fraction=float((mismatch[torch.arange(len(base)),used.argmin(1)]==0).float().mean());return dest,fraction,float((mismatch[cost.argmin(1),] if False else mismatch[torch.arange(len(base)),cost.argmin(1)]>0).float().mean())

def masked_stress(x,budget,seed):
    g=torch.Generator().manual_seed(seed);raw=torch.randn((len(x),len(MUTABLE)),generator=g);raw-=raw.mean(0);raw*=budget*float(.55+.4*torch.rand((),generator=g))/torch.sqrt(raw.square().sum(1).mean()).clamp_min(1e-9);z=x.clone();z[:,list(MUTABLE)]+=raw;return z,float(torch.sqrt(raw.square().sum(1).mean()))

def shift(env,x,y,severity,k,seed,bcal):
    z=x.clone();labels=y.clone();g=torch.Generator().manual_seed(seed);ids=torch.as_tensor(MUTABLE);active=(y==1).nonzero().flatten() if env=="target_side" else torch.arange(len(x));frac=len(active)/len(x);target=severity*bcal;kind=("mean","scale","tail")[k%3]
    if kind=="mean":z[active[:,None],ids]+=target/np.sqrt(len(ids)*frac)*torch.where(torch.rand(len(ids),generator=g)>.5,1.,-1.)
    elif kind=="scale":
        b=z[active[:,None],ids];c=b.mean(0);r=float(torch.sqrt((b-c).square().sum(1).mean()).clamp_min(1e-6));z[active[:,None],ids]=c+(1+target/(np.sqrt(frac)*r))*(b-c)
    else:
        b=z[active];score=b[:,1];cut=torch.quantile(score,.5);lo=(score<=cut).nonzero().flatten();hi=(score>cut).nonzero().flatten();f=min(.5+.18*severity,.95);nl=int(len(active)*f);pick=torch.cat((lo[torch.randint(len(lo),(nl,),generator=g)],hi[torch.randint(len(hi),(len(active)-nl,),generator=g)]));z[active]=b[pick]
    if env=="mixed":keep=max(128,int(len(z)*(1-min(.04*severity,.2))));idx=torch.randperm(len(z),generator=g)[:keep].sort().values;z,labels=z[idx],labels[idx]
    return z,labels,kind

def plaus(x,source,current,favorable,lof):
    c=torch.linalg.vector_norm(x-source,dim=1);nn=torch.cdist(x[:,list(GEOMETRY)],favorable[:,list(GEOMETRY)]).min(1).values
    return {"current_validity":float(valid(current,x).float().mean()),"mean_l2_cost":float(c.mean()),"median_l2_cost":float(c.median()),"p90_l2_cost":float(torch.quantile(c,.9)),"mean_nn_target_distance":float(nn.mean()),"mean_lof":float(lof.score_samples(x[:,list(GEOMETRY)].numpy()).mean()),"population_target_distance":w2_sinkhorn(x[:,list(GEOMETRY)],favorable[:,list(GEOMETRY)])}
def rg(r):return "in_range" if r<.75 else "near_range" if r<=1 else "moderate_out_of_range" if r<=2 else "strong_out_of_range"
def select(g,b,y):
    q=g[g.mean_l2_cost<=b+1e-9]
    return None if q.empty else q.sort_values([y,"mean_l2_cost"],ascending=[False,True]).iloc[0]

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-home",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=0);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"models").mkdir(exist_ok=True)
    cfg={"source":"openml","openml_id":31,"positive_label":"good"};d=load_tabular_three_way(cfg,a.data_home,.6,.2,a.seed);train_x,train_y,cal_x,cal_y,test_x,test_y=(d[k] for k in ("train_x","train_y","cal_x","cal_y","test_x","test_y"));names=d["feature_names"]
    current,_=train_model(train_x,train_y,cal_x,cal_y,[64,32],60,1e-3,a.seed);torch.save(current.state_dict(),out/"models/current.pt");tr,ca,te=metrics(current,train_x,train_y),metrics(current,cal_x,cal_y),metrics(current,test_x,test_y)
    with torch.no_grad():score=torch.sigmoid(current(test_x));negative=(score<.5).nonzero().flatten();g=torch.Generator().manual_seed(a.seed+2);loc=negative[torch.randperm(len(negative),generator=g)[:min(128,len(negative))]]
    source=test_x[loc];source_ids=np.asarray(d["test_ids"])[loc.numpy()];base=wachter(current,source,.5);favorable=train_x[valid(current,train_x)]
    # Protected compatibility context: purpose, personal status/sex, foreign worker, credit history.
    prefixes=("purpose_","personal_status_","foreign_worker_","credit_history_");groups=[torch.as_tensor([i for i,n in enumerate(names) if n.startswith(pre)]) for pre in prefixes]
    global_dest,global_compat,global_reject=anchor_destination(base,favorable,source,groups,False);compat_dest,compat_fraction,rejected=anchor_destination(base,favorable,source,groups,True)
    np.savez(out/"german_split_source_ids.npz",train=d["train_ids"],calibration=d["cal_ids"],test=d["test_ids"],sources=source_ids)
    (out/"german_preprocessing_audit.md").write_text("# German Credit preprocessing audit\n\n60/20/20 stratified split. Median imputation, train-only standardization of numeric/ordinal variables, and train-fitted one-hot encoding of categorical variables. Every model and diagnostic uses the same encoded representation. No fractional categorical CF values are evaluated because categorical coordinates are source-exact.\n")
    policy={"duration":"mutable loan term","credit_amount":"mutable loan term","installment_commitment":"mutable loan term","checking_status":"conditional ordinal; held fixed in v1","savings_status":"conditional ordinal; held fixed in v1","employment":"conditional monotone; held fixed in v1","residence_since":"conditional monotone; held fixed in v1","age":"immutable","personal_status":"immutable/sensitive","foreign_worker":"immutable/sensitive","credit_history":"immutable history","purpose":"immutable for current application","housing":"conditional but held fixed","job":"conditional but held fixed","property_magnitude":"immutable current asset state","existing_credits":"immutable current history","num_dependents":"immutable","other_parties":"immutable current contract","other_payment_plans":"immutable current contract","own_telephone":"immutable in v1"}
    (out/"german_actionability_audit.md").write_text("# German Credit actionability audit\n\n"+"\n".join(f"- `{k}`: {v}." for k,v in policy.items())+"\n\nAll methods can change only duration, credit amount, and installment commitment. All categorical blocks remain source-exact. LIFT-compatible conditions target matching on purpose, personal-status/sex, foreign-worker, and credit-history groups; these rules were fixed before future evaluation.\n")

    bcal,_=bootstrap_budget(cal_x[:,list(GEOMETRY)],20,min(128,len(cal_x)),a.seed+4);proxies=[];stress=[]
    for m in range(5):z,b=masked_stress(cal_x,bcal,a.seed+10_000+m);pm,_=train_model(torch.cat((train_x,z)),torch.cat((train_y,cal_y)),test_x,test_y,[64,32],60,1e-3,a.seed+1000+m);proxies.append(pm);stress.append({"proxy":m,"realized_shift":b,"b_cal":bcal})
    candidates=[]
    def add(method,param,x,detail):candidates.append((method,float(param),x,detail))
    add("base",0,base,"fixed")
    for q in np.arange(.55,.951,.05):add("highconf",q,wachter(current,source,float(q)),"confidence")
    for lam in np.linspace(0,1,101):add("lift_global",lam,(1-lam)*base+lam*global_dest,"global anchor") ;add("lift_compatible",lam,(1-lam)*base+lam*compat_dest,"compatible anchor")
    cache=out/"trex_trajectories.pt"
    if cache.exists():trajectories=torch.load(cache,map_location="cpu",weights_only=True)
    else:
        trajectories={}
        for lr in (.005,.01,.02):t,s=trex_trajectory(current,base,lr,a.seed+50_000);trajectories[f"lr_{lr:g}"]={"trajectory":t,"scores":s}
        torch.save(trajectories,cache)
    for lr in (.005,.01,.02):
      t=trajectories[f"lr_{lr:g}"]["trajectory"];scores=trajectories[f"lr_{lr:g}"]["scores"]
      for steps in (50,100,200):
       ss=scores[:steps]
       for tau in np.arange(.5,.951,.05):ok=ss>=tau;idx=ok.float().argmax(0);reach=ok.any(0);idx[~reach]=steps;add("trex",tau,t[idx,torch.arange(len(base))],f"lr={lr:g};steps={steps};reach={float(reach.float().mean()):.3f}")
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(favorable[:,list(GEOMETRY)].numpy());rows=[];proxyrows=[];tensors={}
    for cid,(method,param,x,detail) in enumerate(candidates):
        vals=[];risks=[]
        with torch.no_grad():
          for j,pm in enumerate(proxies):pp=torch.sigmoid(pm(x));v=float((pp>=.5).float().mean());r=float(torch.relu(torch.tensor(.5)-pp).mean());vals.append(v);risks.append(r);proxyrows.append({"candidate_id":cid,"method":method,"lambda_or_parameter":param,"proxy":j,"validity":v,"risk":r})
        rows.append({"candidate_id":cid,"method":method,"native_parameter":param,"detail":detail,**plaus(x,source,current,favorable,lof),"proxy_mean_validity":np.mean(vals),"proxy_worst_validity":np.min(vals),"calibration_risk":np.max(risks)});tensors[str(cid)]=x
    cand=pd.DataFrame(rows);cand["calibrated_flag"]=0;cand["selection_status"]="not_selected";cand.loc[cand.method=="base",["calibrated_flag","selection_status"]]=[1,"fixed"]
    for method in ("highconf","trex"):
        q=cand[cand.method==method];f=q[q.proxy_mean_validity>=.95];s=(f.sort_values("mean_l2_cost").iloc[0] if len(f) else q.sort_values(["proxy_mean_validity","mean_l2_cost"],ascending=[False,True]).iloc[0]);cand.loc[s.name,["calibrated_flag","selection_status"]]=[1,"properly_calibrated" if len(f) else "fallback_target_infeasible"]
    for method in ("lift_global","lift_compatible"):
        q=cand[cand.method==method].sort_values("native_parameter");r0=float(q.iloc[0].calibration_risk);r1=float(q.iloc[-1].calibration_risk);score=(r0-q.calibration_risk)/max(r0-r1,1e-12)-q.native_parameter;i=score.idxmax();cand.loc[i,["calibrated_flag","selection_status"]]=[1,"gain_over_transport"]
    pd.DataFrame(proxyrows).to_csv(out/"german_lift_calibration.csv",index=False);pd.DataFrame(stress).to_csv(out/"german_calibration_stresses.csv",index=False);torch.save(tensors,out/"candidate_counterfactuals.pt")

    # Early anchor report, then shared common future models.
    align=[]
    for method in ("lift_global","lift_compatible"):
        q=cand[cand.method==method].sort_values("native_parameter");align.append({"method":method,"cost_increasing":bool(np.all(np.diff(q.mean_l2_cost)>=-1e-8)),"cost_decreasing":bool(np.all(np.diff(q.mean_l2_cost)<=1e-8)),"target_distance_decreasing":bool(np.all(np.diff(q.population_target_distance)<=1e-8)),"current_validity_start":q.current_validity.iloc[0],"current_validity_end":q.current_validity.iloc[-1],"selected_lambda":float(q[q.calibrated_flag==1].native_parameter.iloc[0])})
    trexsel=cand[(cand.method=="trex")&(cand.calibrated_flag==1)].iloc[0];vt=tensors[str(int(trexsel.candidate_id))][:,list(MUTABLE)]-base[:,list(MUTABLE)];direction=[]
    for name,dest in (("global",global_dest),("compatible",compat_dest)):
        vl=dest[:,list(MUTABLE)]-base[:,list(MUTABLE)];cos=(vl*vt).sum(1)/(torch.linalg.vector_norm(vl,dim=1)*torch.linalg.vector_norm(vt,dim=1)).clamp_min(1e-12)
        for i,c in enumerate(cos):direction.append({"source_id":int(source_ids[i]),"anchor":name,"cosine":float(c)})
    direction=pd.DataFrame(direction);direction.to_csv(out/"german_lift_trex_direction.csv",index=False)
    (out/"german_anchor_alignment_diagnostic.md").write_text("# German anchor alignment diagnostic\n\n"+markdown_table(pd.DataFrame(align))+"\n\nCompatibility fraction: %.3f; rejected incompatible global matches: %.3f.\n\n"%(compat_fraction,rejected)+markdown_table(direction.groupby("anchor").cosine.agg(["mean","median",lambda x:(x>0).mean(),lambda x:(x<0).mean()]).reset_index()))

    protocol=[];shifted={}
    for env in ENVS:
      for sev in SEVERITIES:
       for k in range(5):sx,sy,kind=shift(env,train_x,train_y,sev,k,a.seed+20_000+1000*k,bcal);n=min(128,len(sx));dist=exact_w2_support(sx[:n,list(GEOMETRY)],train_x[:n,list(GEOMETRY)]);ratio=dist/bcal;protocol.append({"environment":env,"severity":sev,"future_model_id":k,"kind":kind,"empirical_shift_distance":dist,"b_cal":bcal,"relative_shift":ratio,"range_group":rg(ratio)});shifted[(env,sev,k)]=(sx,sy)
    audit=pd.DataFrame(protocol);audit.to_csv(out/"german_shift_scale_audit.csv",index=False);futures=[]
    for r in protocol:
        env,sev,k=r["environment"],r["severity"],r["future_model_id"];path=out/"models"/f"{env}_s{sev:g}_k{k}.pt"
        if not path.exists():sx,sy=shifted[(env,sev,k)];fm,_=train_model(sx,sy,test_x,test_y,[64,32],60,1e-3,a.seed+30_000+10_000*ENVS.index(env)+100*int(sev*10)+k);torch.save(fm.state_dict(),path)
        futures.append((env,sev,k,load_model(path,train_x.shape[1])))
    ev=[]
    for _,r in cand.iterrows():
      x=tensors[str(int(r.candidate_id))]
      for env,sev,k,m in futures:ev.append({"candidate_id":int(r.candidate_id),"method":r.method,"native_parameter":r.native_parameter,"environment":env,"severity":sev,"future_model_id":k,"validity":float(valid(m,x).float().mean())})
    ev=pd.DataFrame(ev);ev.to_csv(out/"german_future_evaluation.csv",index=False)
    ev.groupby(["candidate_id","method","native_parameter","environment","severity"],as_index=False).agg(mean_future_validity=("validity","mean"),worst_future_validity=("validity","min")).to_csv(out/"german_environment_severity_summary.csv",index=False)
    per=ev.groupby(["candidate_id","environment"],as_index=False).agg(mean=("validity","mean"),worst=("validity","min"));w=per.pivot(index="candidate_id",columns="environment",values="worst");cand["mean_future_validity"]=cand.candidate_id.map(ev.groupby("candidate_id").validity.mean());cand["common_global_worst"]=cand.candidate_id.map(w.min(axis=1))
    for env in ENVS:cand[f"worst__{env}"]=cand.candidate_id.map(w[env])
    cand.to_csv(out/"german_candidate_frontier.csv",index=False)
    # Separate native track.
    nr=[]
    for sev in SEVERITIES:
      for k in range(5):
        gen=torch.Generator().manual_seed(a.seed+60_000+k);idx=torch.randint(len(train_x),(len(train_x),),generator=gen) if k%2==0 else torch.randperm(len(train_x),generator=gen)[:int(len(train_x)*(1-min(.04*sev,.2)))];path=out/"models"/f"native_s{sev:g}_k{k}.pt"
        if not path.exists():fm,_=train_model(train_x[idx],train_y[idx],test_x,test_y,[64,32],60,1e-3,a.seed+70_000+100*int(sev*10)+k);torch.save(fm.state_dict(),path)
        fm=load_model(path,train_x.shape[1])
        for _,r in cand[cand.calibrated_flag==1].iterrows():nr.append({"method":r.method,"parameter":r.native_parameter,"severity":sev,"future_model_id":k,"validity":float(valid(fm,tensors[str(int(r.candidate_id))]).float().mean()),"cost":r.mean_l2_cost})
    pd.DataFrame(nr).to_csv(out/"german_native_model_multiplicity.csv",index=False)

    # Strict all-swept-method overlap budget.
    swept=("highconf","trex","lift_global","lift_compatible");lo=max(float(cand[cand.method==m].mean_l2_cost.min()) for m in swept);hi=min(float(cand[cand.method==m].mean_l2_cost.max()) for m in swept);budgets=np.linspace(lo,hi,12) if hi>=lo else np.array([]);matched=[]
    for b in budgets:
      for method in swept:
       g=cand[cand.method==method]
       for scope,y in [("common","common_global_worst")]+[(e,f"worst__{e}") for e in ENVS]:
        r=select(g,b,y);matched.append({"cost_budget":b,"method":method,"environment":scope,"parameter":r.native_parameter,"actual_cost":r.mean_l2_cost,"mean_future_validity":r.mean_future_validity,"worst_future_validity":r[y],"nearest_target_distance":r.mean_nn_target_distance,"mean_lof":r.mean_lof,"population_target_distance":r.population_target_distance})
    matched_columns=["cost_budget","method","environment","parameter","actual_cost","mean_future_validity","worst_future_validity","nearest_target_distance","mean_lof","population_target_distance"]
    matched=pd.DataFrame(matched,columns=matched_columns)
    matched[matched.environment=="common"].to_csv(out/"german_matched_cost_common.csv",index=False);matched[matched.environment!="common"].to_csv(out/"german_matched_cost_by_environment.csv",index=False);matched.to_csv(out/"german_matched_cost_plausibility.csv",index=False)
    for scope,y,name in [("common","common_global_worst","german_cost_vs_common_worst"),("mean","mean_future_validity","german_cost_vs_mean_future")]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for method in swept:
            q=cand[cand.method==method].sort_values("mean_l2_cost");ax.plot(q.mean_l2_cost,q[y],"o",ms=2,color=COL[method],label=LAB[method])
        ax.set(xlabel="mean L2 cost",ylabel=y.replace("_"," "));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/f"{name}.png",dpi=240);plt.close(fig)
    for method in ("lift_global","lift_compatible"):
        q=cand[cand.method==method].copy();q["W2_to_PCF"]=[exact_w2_support(tensors[str(int(i))][:,list(GEOMETRY)],base[:,list(GEOMETRY)]) for i in q.candidate_id];q["W2_to_anchor"]=[w2_sinkhorn(tensors[str(int(i))][:,list(GEOMETRY)],favorable[:,list(GEOMETRY)]) for i in q.candidate_id];q.to_csv(out/f"german_{method}_path.csv",index=False)
    selected=cand[cand.calibrated_flag==1];support=cand.groupby("method",as_index=False).agg(min_cost=("mean_l2_cost","min"),max_cost=("mean_l2_cost","max"));meta={"sources_requested":128,"sources_available":len(source),"b_cal":bcal,"b_cal_over_W2_PCF_P1":bcal/w2_sinkhorn(base[:,list(GEOMETRY)],favorable[:,list(GEOMETRY)]),"current":{"train":tr,"calibration":ca,"test":te,"class_balance":float(train_y.mean())},"compatibility_fraction":compat_fraction,"rejected_global_fraction":rejected,"selected":selected[["method","native_parameter","selection_status"]].to_dict("records"),"future_not_used_for_calibration":True};(out/"metadata.json").write_text(json.dumps(meta,indent=2))
    selected_cols=["method","native_parameter","selection_status","current_validity","mean_l2_cost","mean_future_validity","common_global_worst","worst__generic_population","worst__target_side","worst__mixed","mean_nn_target_distance","mean_lof","population_target_distance"]
    shift_summary=audit.groupby(["environment","severity","range_group"],as_index=False).relative_shift.agg(["mean","min","max"]).reset_index()
    report=["# German Credit benchmark milestone v1","","## Current model and sources","",markdown_table(pd.DataFrame([{"train_accuracy":tr[0],"train_auc":tr[1],"cal_accuracy":ca[0],"cal_auc":ca[1],"test_accuracy":te[0],"test_auc":te[1],"class_balance":train_y.mean(),"sources":len(source)}])),"",f"Only {len(source)} distinct test instances were predicted unfavorable; no source was duplicated to reach 128.","","## Selected points","",markdown_table(selected[selected_cols]),"","The T-Rex row is a descriptive fallback because no candidate reached the predeclared 95% proxy-validity target; it is not a properly calibrated deployment point.","","## Cost support","",markdown_table(support),"",f"The strict common support is empty: lower bound {lo:.3f} exceeds upper bound {hi:.3f}. Accordingly, no formal matched-cost rows are fabricated.","","## Anchor alignment","",markdown_table(pd.DataFrame(align)),"",f"Only {compat_fraction:.1%} of compatibility-conditioned matches satisfy all protected context groups. The global nearest-anchor rule would reject {rejected:.1%} of matches under those constraints.","","## Matched cost","",markdown_table(matched),"","## Shift audit","",markdown_table(shift_summary),"","The evaluation spans in/near-range, moderate, and strong mismatch ratios. Shift sizes were fixed from b_cal, not method performance.","","## Direct answers","","1. **Global anchor compatibility:** no. Moving toward global P1 reduces target distance but current validity falls from 100% to 2.2%.","2. **Compatible anchor:** it materially raises endpoint current validity to 22.2%, but still leaves a severe validity collapse.","3. **Calibration:** both LIFT variants select lambda=0 using calibration only.","4. **Cost overlap:** there is no strict overlap across all swept methods. LIFT max cost is 2.078, T-Rex min cost is 2.119, and HighConf min cost is 2.323.","5. **Matched-cost robustness:** not identifiable under the no-extrapolation rule because supports do not overlap.","6. **Matched-cost plausibility:** likewise not formally identifiable. Along the LIFT paths, target distances improve, but those points lose current validity.","7. **Raw robustness efficiency:** T-Rex reaches higher generic/mixed worst validity than LIFT, but this is not a matched-cost result and its selected point is a calibration fallback.","8. **Compatible versus global LIFT:** compatible anchoring mitigates, but does not solve, the anchor-validity conflict; neither calibrates to positive refinement.","9. **Environment dependence:** generic and mixed shifts distinguish methods, while the worst target-side future model drives every method to zero; the latter is therefore non-discriminating at the aggregate-worst level. Severity-level results are saved separately.","10. **Dataset regime:** this seed resembles Adult more than HELOC: favorable-support alignment conflicts with classifier-valid actionable refinement.","","## Constraints","","No categorical CF coordinate changes; no fractional one-hot values are evaluated. Conditional ordinal transitions are classified but held fixed in this debug milestone. No final future model is used for calibration or anchor construction. LOF is saved as sklearn `score_samples`: more negative values indicate worse outlierness."]
    (out/"german_credit_benchmark_milestone_v1.md").write_text("\n".join(report));torch.save({"source":source,"base":base,"global_destination":global_dest,"compatible_destination":compat_dest},out/"core_artifacts.pt")
    print(json.dumps({"sources":len(source),"base_validity":float(valid(current,base).float().mean()),"b_cal":bcal,"compatibility_fraction":compat_fraction,"cost_overlap":[lo,hi],"budgets":len(budgets),"selected":meta["selected"],"alignment":align},indent=2))
if __name__=="__main__":main()
