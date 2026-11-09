"""Finish official ElliCE runs and German mixed-domain Geo-LIFT audit only."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor

from ..baselines.official_adapters import generate_ellice_cf
from ..data.tabular import load_tabular_three_way
from .german_credit_benchmark_milestone_v1 import load_model as load_german, masked_stress
from .geo_uniform_lift_ablation_v1 import geometry
from .path_lift_decision_compatible_ot_v1 import setting_artifacts, pair_violations, score
from .constrained_lift_ablation_v1 import exact_constrained_plan
from .heloc_benchmark_milestone import valid
from .stability_weighted_anchor_v1 import weighted_sinkhorn
from ..models.tabular import train_model


GERMAN_CFG={"source":"openml","openml_id":31,"positive_label":"good"}
ORDINAL={"installment_commitment","residence_since","existing_credits","num_dependents"}
CONTINUOUS={"duration","credit_amount","age"}
GERMAN_CATEGORICAL=("checking_status","credit_history","employment","foreign_worker",
 "housing","job","other_parties","other_payment_plans","own_telephone",
 "personal_status","property_magnitude","purpose","savings_status")


def semantic_groups(names):
    numeric=[i for i,n in enumerate(names) if n in (CONTINUOUS|ORDINAL)]
    grouped={k:[] for k in GERMAN_CATEGORICAL}
    for i,n in enumerate(names):
        if i in numeric: continue
        matches=[k for k in GERMAN_CATEGORICAL if n.startswith(k+"_")]
        if len(matches)!=1: raise ValueError(f"Cannot uniquely assign encoded German column {n!r}")
        grouped[matches[0]].append(i)
    return numeric,grouped


def schema_from_training(train,names):
    numeric,groups=semantic_groups(names); spec={"continuous":[],"ordinal":[],"binary":[],"onehot":[]}
    for i in numeric:
        item={"index":i,"name":names[i],"values":sorted(torch.unique(train[:,i]).tolist()),
              "min":float(train[:,i].min()),"max":float(train[:,i].max())}
        (spec["ordinal"] if names[i] in ORDINAL else spec["continuous"]).append(item)
    for name,idx in groups.items():
        # OpenML German uses multiway one-hot blocks. A two-column block is
        # still represented as a nominal exactly-one group, not independent bits.
        spec["onehot"].append({"name":name,"indices":idx,
          "low":[float(train[:,j].min()) for j in idx],
          "high":[float(train[:,j].max()) for j in idx]})
    return spec


def project_to_legal_domain(z,schema):
    out=z.clone()
    for q in schema["continuous"]: out[:,q["index"]].clamp_(q["min"],q["max"])
    for q in schema["ordinal"]:
        vals=torch.tensor(q["values"],dtype=out.dtype);i=q["index"]
        out[:,i]=vals[(out[:,i,None]-vals).abs().argmin(1)]
    for q in schema["binary"]:
        vals=torch.tensor(q["values"],dtype=out.dtype);i=q["index"]
        out[:,i]=vals[(out[:,i,None]-vals).abs().argmin(1)]
    for q in schema["onehot"]:
        idx=q["indices"];lo=torch.tensor(q["low"],dtype=out.dtype);hi=torch.tensor(q["high"],dtype=out.dtype)
        # Recover the underlying indicator scale before argmax because the
        # benchmark standardizes encoded columns after one-hot encoding.
        indicator=(out[:,idx]-lo)/(hi-lo).clamp_min(1e-12);winner=indicator.argmax(1)
        out[:,idx]=lo
        out[torch.arange(len(out)),torch.tensor(idx)[winner]]=hi[winner]
    return out


def validate_cf(z,schema):
    finite=torch.isfinite(z).all(1);legal=finite.clone();categorical=finite.clone()
    for q in schema["continuous"]:
        x=z[:,q["index"]];legal &= (x>=q["min"]-1e-5)&(x<=q["max"]+1e-5)
    for kind in ("ordinal","binary"):
        for q in schema[kind]:
            vals=torch.tensor(q["values"],dtype=z.dtype);ok=(z[:,q["index"],None]-vals).abs().min(1).values<1e-5
            legal &= ok
            if kind=="binary":categorical &= ok
    for q in schema["onehot"]:
        idx=q["indices"];lo=torch.tensor(q["low"],dtype=z.dtype);hi=torch.tensor(q["high"],dtype=z.dtype)
        islo=(z[:,idx]-lo).abs()<1e-5;ishi=(z[:,idx]-hi).abs()<1e-5
        ok=(islo|ishi).all(1)&(ishi.sum(1)==1);legal &= ok;categorical &= ok
    return legal,categorical


def load_common(ds,home,roots,german_root):
    if ds in ("heloc","diabetes"):
        d,source,ids,base,current,proxies,_,front,tensors,trex=setting_artifacts(ds,"baseline_mlp",home,roots)
        return d,source,np.asarray(ids),base,current,proxies,{"Base":base,"T-Rex":trex}
    d=load_tabular_three_way(GERMAN_CFG,home,.6,.2,0);art=torch.load(german_root/"core_artifacts.pt",map_location="cpu",weights_only=True)
    current=load_german(german_root/"models/current.pt",d["train_x"].shape[1]);source,base=art["source"],art["base"]
    # Recover exact test IDs by nearest equality; sources are frozen test rows.
    near=torch.cdist(source,d["test_x"]).argmin(1);ids=np.asarray(d["test_ids"])[near.numpy()]
    bcal=float(json.loads((german_root/"metadata.json").read_text())["b_cal"]);proxies=[]
    for m in range(5):
        stressed,_=masked_stress(d["cal_x"],bcal,10_000+m)
        pm,_=train_model(torch.cat((d["train_x"],stressed)),torch.cat((d["train_y"],d["cal_y"])),
                         d["test_x"],d["test_y"],[64,32],60,1e-3,1000+m);proxies.append(pm)
    front=pd.read_csv(german_root.parent/"german_credit_strict_audit_seed0"/"german_candidate_frontier_audited.csv")
    tensors=torch.load(german_root.parent/"german_credit_strict_audit_seed0"/"candidate_counterfactuals_audited.pt",map_location="cpu",weights_only=True) if (german_root.parent/"german_credit_strict_audit_seed0"/"candidate_counterfactuals_audited.pt").exists() else None
    methods={"Base":base}
    if tensors is not None:
        row=front[(front.method=="trex")&(front.calibrated_flag==1)].iloc[0];methods["T-Rex"]=tensors[str(int(row.candidate_id))]
    return d,source,ids,base,current,proxies,methods


def future_models(ds,dim,future_root):
    return [load_german(future_root/f"{ds}_future_{k}.pt",dim) for k in range(5)]


def evaluate(z,source,current,futures,fav,lof,failure):
    success=~failure;cur=((score(current,z)>=.5)&success);fv=[]
    for fm in futures:fv.append(float(((score(fm,z)>=.5)&success).float().mean()))
    l1=(z-source).abs().sum(1);l2=torch.linalg.vector_norm(z-source,dim=1)
    return {"current_validity":float(cur.float().mean()),"future_validity":float(np.mean(fv)),
      "future_validity_se":float(np.std(fv,ddof=1)/np.sqrt(len(fv))),"l1_cost":float(l1.mean()),
      "l2_cost":float(l2.mean()),"lof":float(lof.score_samples(z.numpy()).mean()),"failure_rate":float(failure.float().mean())}


def run_ellice(ds,d,source,ids,current,futures,out,schema=None):
    cache=out/f"ellice_{ds}_counterfactuals.pt";train=d["train_x"];names=d["feature_names"]
    if cache.exists(): obj=torch.load(cache,map_location="cpu",weights_only=True);z=obj["counterfactuals"];native_failed=obj["failed"]
    else:
        rows=[];failed=[];groups=[[names[i] for i in q["indices"]] for q in schema["onehot"]] if schema else None
        generator_train=train.clone();generator_source=source.clone();scale=torch.ones(train.shape[1]);shift=torch.zeros(train.shape[1])
        permitted_range=None;allowed_values=None
        if schema:
            permitted_range={q["name"]:[q["min"],q["max"]] for q in schema["continuous"]}
            allowed_values={q["name"]:q["values"] for q in schema["ordinal"]}
            for q in schema["onehot"]:
                idx=q["indices"];lo=torch.tensor(q["low"]);hi=torch.tensor(q["high"]);span=(hi-lo).clamp_min(1e-12)
                generator_train[:,idx]=(train[:,idx]-lo)/span;generator_source[:,idx]=(source[:,idx]-lo)/span
                scale[idx]=span;shift[idx]=lo
        for i,x in enumerate(source):
            try:
                cf=generate_ellice_cf(generator_source[i].numpy(),current,generator_train.numpy(),d["train_y"].numpy(),names,
                    target_class=1,robustness_epsilon=.01,one_hot_groups=groups,
                    model_input_scale=scale.numpy(),model_input_shift=shift.numpy(),
                    permitted_range=permitted_range,allowed_values=allowed_values)
                if cf.empty: rows.append(x.clone());failed.append(True)
                else:
                    value=torch.tensor(cf[names].iloc[0].to_numpy(float),dtype=torch.float32)*scale+shift
                    rows.append(value);failed.append(False)
            except Exception as exc:
                print(f"ElliCE {ds} source {i} failed: {type(exc).__name__}: {exc}",flush=True);rows.append(x.clone());failed.append(True)
            if (i+1)%5==0 or i+1==len(source):
                torch.save({"counterfactuals":torch.stack(rows),"failed":torch.tensor(failed),"completed":i+1},cache)
                print(f"ElliCE {ds}: {i+1}/{len(source)}",flush=True)
        z=torch.stack(rows);native_failed=torch.tensor(failed)
    if schema:
        legal,cat=validate_cf(z,schema)
    else:legal=torch.isfinite(z).all(1);cat=legal.clone()
    failure=native_failed|(~legal)|(score(current,z)<.5)
    fav=train[valid(current,train)];lof=LocalOutlierFactor(n_neighbors=min(20,len(fav)-1),novelty=True).fit(fav.numpy())
    return z,failure,evaluate(z,source,current,futures,fav,lof,failure),legal,cat


def run_geo_german(d,source,base,current,proxies,futures,schema):
    fav=d["train_x"][valid(current,d["train_x"])];w=np.ones(len(fav))/len(fav);eu=torch.cdist(base,fav)
    _,_,_,mm=pair_violations(current,base,fav);straight=(mm>=-1e-10).numpy();geo,_,conn,_=geometry(current,base,fav,straight,eu.numpy(),10)
    finite=geo[conn];num=geo.copy();num[~conn]=finite.max()*10;plan,_,_,_=exact_constrained_plan(num**2,conn,w)
    p=torch.tensor(plan,dtype=base.dtype);dest=(p/p.sum(1,keepdim=True).clamp_min(1e-30))@fav
    risks=[];barys=[]
    for lam in np.linspace(0,1,101):
        z=(1-lam)*base+lam*dest;barys.append(z);risks.append(max(float(torch.relu(.5-score(pm,z)).mean()) for pm in proxies))
    r0,r1=risks[0],risks[-1];gain=(r0-np.asarray(risks))/max(r0-r1,1e-12);ii=int(np.argmax(gain-np.linspace(0,1,101)))
    bary=barys[ii];projected=project_to_legal_domain(bary,schema);rawlegal,rawcat=validate_cf(bary,schema);legal,cat=validate_cf(projected,schema)
    distortion=projected-bary;changed=(distortion.abs().sum(1)>1e-8);failure=(~legal)|(score(current,projected)<.5)
    lof=LocalOutlierFactor(n_neighbors=min(20,len(fav)-1),novelty=True).fit(fav.numpy());metrics=evaluate(projected,source,current,futures,fav,lof,failure)
    metrics.update({"selected_lambda":ii/100,"numerical_barycenter_current_validity":float((score(current,bary)>=.5).float().mean()),
      "projected_current_validity_unconditional":float((score(current,projected)>=.5).float().mean()),
      "legality_before_projection":float(rawlegal.float().mean()),"legality_after_projection":float(legal.float().mean()),
      "mean_projection_l1":float(distortion.abs().sum(1).mean()),"mean_projection_l2":float(torch.linalg.vector_norm(distortion,dim=1).mean()),
      "categorical_projection_rate":float(changed.float().mean())})
    return bary,projected,failure,metrics,rawlegal,rawcat,legal,cat


def audit_rows(ds,method,ids,raw,final,failure,schema=None,projection=False):
    if schema:rv,rc=validate_cf(raw,schema);lv,lc=validate_cf(final,schema)
    else:rv=torch.isfinite(raw).all(1);rc=rv;lv=torch.isfinite(final).all(1);lc=lv
    delta=final-raw
    return [{"dataset":ds,"method":method,"instance_id":int(ids[i]),"raw_cf_valid":bool(rv[i]),
      "legal_cf_valid":bool(lv[i]),"categorical_legal":bool(lc[i]),"projection_applied":bool(projection and (delta[i].abs().sum()>1e-8)),
      "projection_l1":float(delta[i].abs().sum()),"projection_l2":float(torch.linalg.vector_norm(delta[i])),
      "current_valid_after_projection":bool(not failure[i]),"failure":bool(failure[i])} for i in range(len(final))]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);ap.add_argument("--future-root",required=True)
    ap.add_argument("--heloc-home",required=True);ap.add_argument("--diabetes-home",required=True);ap.add_argument("--german-home",required=True)
    ap.add_argument("--corrected-root",required=True);ap.add_argument("--margin-root",required=True);ap.add_argument("--logistic-root",required=True)
    ap.add_argument("--heloc-current-root",required=True);ap.add_argument("--diabetes-current-root",required=True);ap.add_argument("--german-root",required=True)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);future_root=Path(a.future_root)
    roots={"corrected":Path(a.corrected_root),"margin":Path(a.margin_root),"logistic":Path(a.logistic_root),
           "heloc_current":Path(a.heloc_current_root),"diabetes_current":Path(a.diabetes_current_root)}
    summaries=[];audit=[];packs={}
    for ds,home in (("heloc",a.heloc_home),("diabetes",a.diabetes_home),("german_credit",a.german_home)):
        d,source,ids,base,current,proxies,methods=load_common(ds,home,roots,Path(a.german_root));packs[ds]=(d,source,ids,base,current,proxies,methods)
        schema=schema_from_training(d["train_x"],d["feature_names"]) if ds=="german_credit" else None
        futures=future_models(ds,d["train_x"].shape[1],future_root);z,fail,m,legal,cat=run_ellice(ds,d,source,ids,current,futures,out,schema)
        m.update({"dataset":ds,"method":"ElliCE","generation":"yes","categorical_legal":float(cat.float().mean()),"full_evaluation":"yes"});summaries.append(m)
        audit += audit_rows(ds,"ElliCE",ids,z,z,fail,schema)
        if ds=="german_credit":
            for name,x in methods.items():
                lf,lc=validate_cf(x,schema);ff=(~lf)|(score(current,x)<.5);audit+=audit_rows(ds,name,ids,x,x,ff,schema)
            bary,proj,fail,g,rl,rc,ll,lc=run_geo_german(d,source,base,current,proxies,futures,schema)
            torch.save({"z_bary":bary,"z_projected":proj,"failed":fail},out/"geo_lift_german_projected_counterfactuals.pt")
            g.update({"dataset":ds,"method":"Geo-LIFT","generation":"yes","categorical_legal":float(lc.float().mean()),"full_evaluation":"yes"});summaries.append(g)
            audit+=audit_rows(ds,"Geo-LIFT",ids,bary,proj,fail,schema,True)
            # ROAR remains N/A: official CARLA only handles independent binary
            # rounding or pairs, not German multiway one-hot blocks.
            summaries.append({"dataset":ds,"method":"ROAR","generation":"yes","categorical_legal":np.nan,
              "full_evaluation":"N/A","failure_rate":1.0,"notes":"N/A -- continuous solution cannot be legally realized under the categorical benchmark representation without introducing an additional post-processing rule not specified by ROAR."})
    pd.DataFrame(summaries).to_csv(out/"requested_method_results.csv",index=False)
    pd.DataFrame(audit).to_csv(out/"categorical_realization_audit.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=="__main__":main()
