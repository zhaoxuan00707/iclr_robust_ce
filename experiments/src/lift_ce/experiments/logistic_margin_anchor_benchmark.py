"""HELOC/Diabetes benchmark with an AdamW-regularized logistic classifier."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from .german_credit_strict_audit import solve_wachter,trex_trajectory
from .stability_weighted_anchor_v1 import weighted_sinkhorn
from .heloc_benchmark_milestone import bootstrap_budget,valid
from .heloc_method_bootstrap import calibration_stress
from .heloc_debug_diagnostics import w2_sinkhorn,markdown_table
from .corrected_future_benchmark_v1 import make_spec,apply_spec,metrics,empirical_w2,range_group
from ..data.tabular import load_tabular_three_way
from ..utils import seed_everything
from ..models.tabular import TabularMLP

MODEL_KIND="logistic"
WEIGHT_DECAY=1e-4

class Logistic(nn.Module):
    def __init__(self,d):super().__init__();self.linear=nn.Linear(d,1)
    def forward(self,x):return self.linear(x).squeeze(-1)

def train_logistic(x,y,epochs,seed,weight_decay=1e-4):
    seed_everything(seed);m=Logistic(x.shape[1]) if MODEL_KIND=="logistic" else TabularMLP(x.shape[1],[64,32]);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=WEIGHT_DECAY);loss=nn.BCEWithLogitsLoss();loader=DataLoader(TensorDataset(x,y),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
    # Epoch counts inherited from the MLP correspond to very different update
    # counts across HELOC and tiny Diabetes. Require a fixed convergence budget
    # for the logistic model while retaining the original lower bound.
    effective_epochs=max(epochs,int(np.ceil(1000/max(1,len(loader))))) if MODEL_KIND=="logistic" else epochs
    for _ in range(effective_epochs):
      for a,b in loader:opt.zero_grad(set_to_none=True);v=loss(m(a),b);v.backward();opt.step()
    m.eval();return m

def select_lift(path):
    g=path.sort_values('native_parameter');r0,r1=g.iloc[0].calibration_risk,g.iloc[-1].calibration_risk;gain=(r0-g.calibration_risk)/max(r0-r1,1e-12)-g['native_parameter'];return gain.idxmax()

def evaluate(ds,home,out,seed=0):
    out.mkdir(parents=True,exist_ok=True);(out/'models').mkdir(exist_ok=True)
    if ds=='heloc':cfg={'source':'openml','openml_name':'heloc','openml_version':1,'missing_values':[-7,-8,-9]};envs=('multiplicity','generic_population_shift','mixed');sevs=(0.,1.,2.);epochs=30
    else:cfg={'source':'openml','openml_id':37,'positive_label':'tested_negative'};envs=('generic_population_shift','target_side_shift','mixed_shift');sevs=(.5,1.,2.5);epochs=40
    d=load_tabular_three_way(cfg,home,.6,.2,seed);train_x,train_y,cal_x,cal_y,test_x,test_y=(d[k] for k in ('train_x','train_y','cal_x','cal_y','test_x','test_y'));model=train_logistic(train_x,train_y,epochs,seed);torch.save(model.state_dict(),out/'models/current.pt')
    with torch.no_grad():score=torch.sigmoid(model(test_x));negative=(score<.5).nonzero().flatten();loc=negative[torch.randperm(len(negative),generator=torch.Generator().manual_seed(seed+2))[:min(128,len(negative))]]
    source=test_x[loc];source_ids=np.asarray(d['test_ids'])[loc.numpy()];thresholds=[.5]+[round(float(x),2) for x in np.arange(.55,.951,.05)];solved={}
    for p in thresholds:solved[p],_=solve_wachter(model,source,p,{}, {},5,500,seed+90_000)
    base=solved[.5];fav=train_x[valid(model,train_x)];scores=torch.sigmoid(model(fav)).detach();margin=torch.relu(scores-.5);weights={'LIFT-UniformAnchor':torch.ones(len(fav))/len(fav)};raw=torch.minimum(margin,torch.tensor(.25))+1e-6;weights['LIFT-MarginAnchor']=raw/raw.sum();ground=torch.cdist(base,fav).square();dest={};weightrows=[]
    for name,w in weights.items():
        plan=weighted_sinkhorn(ground,w,iterations=1000,tol=1e-8);assert float((plan.sum(0)-w).abs().max())<1e-7;dest[name]=(plan/plan.sum(1,keepdim=True).clamp_min(1e-30))@fav;weightrows.append({'dataset':ds,'anchor_type':name,'ESS':float(1/w.square().sum()),'ESS_over_N1':float(1/w.square().sum()/len(w)),'mean_target_score':float((w*scores).sum()),'mean_target_margin':float((w*margin).sum())})
    bcal,_=bootstrap_budget(cal_x,20,min(128,len(cal_x)),seed+4);proxies=[]
    for j in range(5):stressed,_=calibration_stress(cal_x,bcal,seed+10_000+j);proxies.append(train_logistic(torch.cat((train_x,stressed)),torch.cat((train_y,cal_y)),epochs,seed+1000+j))
    candidates=[]
    for p,z in solved.items():candidates.append(('Base' if p==.5 else 'HighConf',p,z,'fixed' if p==.5 else 'confidence'))
    for name,dd in dest.items():
      for lam in np.linspace(0,1,101):candidates.append((name,round(float(lam),2),(1-lam)*base+lam*dd,'lambda'))
    trajectories={}
    for lr in (.005,.01,.02):t,s=trex_trajectory(model,base,lr,{}, {},seed+50_000);trajectories[lr]=(t,s)
    for lr,(t,s) in trajectories.items():
      for steps in (50,100,200):
       ss=s[:steps]
       for tau in np.arange(.5,.951,.05):ok=ss>=tau;idx=ok.float().argmax(0);idx[~ok.any(0)]=steps;candidates.append(('T-Rex',round(float(tau),2),t[idx,torch.arange(len(base))],f'lr={lr};steps={steps}'))
    lof=LocalOutlierFactor(n_neighbors=min(20,len(fav)-1),novelty=True).fit(fav.numpy());rows=[];tensors={}
    for cid,(method,param,z,detail) in enumerate(candidates):
        vals=[];risks=[]
        with torch.no_grad():
          for proxy in proxies:p=torch.sigmoid(proxy(z));vals.append(float((p>=.5).float().mean()));risks.append(float(torch.relu(torch.tensor(.5)-p).mean()))
        l2=torch.linalg.vector_norm(z-source,dim=1);nn=torch.cdist(z,fav).min(1).values;rows.append({'candidate_id':cid,'method':method,'native_parameter':param,'detail':detail,'current_validity':float(valid(model,z).float().mean()),'mean_L2_cost':float(l2.mean()),'median_L2_cost':float(l2.median()),'p90_L2_cost':float(torch.quantile(l2,.9)),'NN_target_distance':float(nn.mean()),'LOF':float(lof.score_samples(z.numpy()).mean()),'population_target_distance':w2_sinkhorn(z,fav),'proxy_mean_validity':np.mean(vals),'proxy_worst_validity':min(vals),'calibration_risk':max(risks)});tensors[cid]=z
    front=pd.DataFrame(rows);front['calibrated_flag']=0;front.loc[front.method=='Base','calibrated_flag']=1
    for method in ('HighConf','T-Rex'):
        q=front[front.method==method];feas=q[q.proxy_mean_validity>=.95];r=(feas.sort_values('mean_L2_cost').iloc[0] if len(feas) else q.sort_values(['proxy_mean_validity','mean_L2_cost'],ascending=[False,True]).iloc[0]);front.loc[r.name,'calibrated_flag']=1
    for method in ('LIFT-UniformAnchor','LIFT-MarginAnchor'):front.loc[select_lift(front[front.method==method]),'calibrated_flag']=1
    cor=[abs(float(np.corrcoef(train_x[:,j],train_y)[0,1])) for j in range(train_x.shape[1])];features=np.argsort(cor)[-3:][::-1] if ds=='diabetes' else [];future={};sanity=[]
    for env in envs:
      for sev in sevs:
       for k in range(5):
        spec=make_spec(ds,env,sev,k,train_x,train_y,features,bcal,seed+20_000+1000*k);sub=env in ('multiplicity','mixed','mixed_shift') and sev>0;sx,sy,_=apply_spec(spec,train_x,train_y,seed+20_011+1000*k,sub);tx,ty,_=apply_spec(spec,test_x,test_y,seed+20_029+1000*k,False);fm=train_logistic(sx,sy,epochs,seed+30_000+10_000*envs.index(env)+100*int(sev*10 if ds=='diabetes' else sev)+k);torch.save(fm.state_dict(),out/'models'/f'{env}_s{sev:g}_k{k}.pt');sm=metrics(fm,tx,ty);rawshift=0 if env=='multiplicity' else empirical_w2(train_x,sx);rg=range_group(rawshift/bcal,sm);future[(env,sev,k)]=(fm,rg);sanity.append({'dataset':ds,'environment':env,'severity':sev,'future_model_id':k,'range_group':rg,**sm,'relative_shift':rawshift/bcal})
    evalrows=[]
    for _,r in front.iterrows():
      z=tensors[int(r.candidate_id)]
      for (env,sev,k),(fm,rg) in future.items():evalrows.append({'candidate_id':r.candidate_id,'method':r.method,'native_parameter':r.native_parameter,'environment':env,'severity':sev,'future_model_id':k,'range_group':rg,'validity':float(valid(fm,z).float().mean())})
    ev=pd.DataFrame(evalrows);front['mean_future_validity']=front.candidate_id.map(ev.groupby('candidate_id').validity.mean());meaning=ev[ev.range_group.isin(['in_range','near_range'])];front['near_in_range_worst']=front.candidate_id.map(meaning.groupby('candidate_id').validity.min())
    for env in envs:front[f'worst__{env}']=front.candidate_id.map(ev[ev.environment==env].groupby('candidate_id').validity.min())
    # L2 matched costs.
    methods=('HighConf','T-Rex','LIFT-UniformAnchor','LIFT-MarginAnchor');lo=max(front[front.method==m].mean_L2_cost.min() for m in methods);hi=min(front[front.method==m].mean_L2_cost.max() for m in methods);matched=[]
    for budget in np.linspace(lo,hi,20) if hi>=lo else []:
      for method in methods:
        q=front[(front.method==method)&(front.mean_L2_cost<=budget+1e-9)]
        if q.empty:continue
        r=q.sort_values(['near_in_range_worst','mean_L2_cost'],ascending=[False,True]).iloc[0];matched.append({'dataset':ds,'cost_budget':budget,'method':method,'native_parameter':r.native_parameter,'actual_L2_cost':r.mean_L2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':r.NN_target_distance,'LOF':r.LOF,'population_target_distance':r.population_target_distance})
    front.to_csv(out/f'{ds}_logistic_frontier.csv',index=False);ev.to_csv(out/f'{ds}_logistic_future_evaluation.csv',index=False);pd.DataFrame(sanity).to_csv(out/f'{ds}_logistic_future_sanity.csv',index=False);pd.DataFrame(matched).to_csv(out/f'{ds}_logistic_matched_cost.csv',index=False);pd.DataFrame(weightrows).to_csv(out/f'{ds}_logistic_margin_weights.csv',index=False);torch.save(tensors,out/f'{ds}_logistic_counterfactuals.pt');return front,pd.DataFrame(matched),pd.DataFrame(weightrows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--heloc-data-home',required=True);p.add_argument('--diabetes-data-home',required=True);p.add_argument('--output',required=True);p.add_argument('--classifier',choices=('logistic','mlp'),default='logistic');p.add_argument('--weight-decay',type=float,default=1e-4);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);res=[]
    global MODEL_KIND,WEIGHT_DECAY;MODEL_KIND=a.classifier;WEIGHT_DECAY=a.weight_decay
    for ds,home in (('heloc',a.heloc_data_home),('diabetes',a.diabetes_data_home)):res.append((ds,*evaluate(ds,home,out/ds)))
    selected=pd.concat([r[1][r[1].calibrated_flag==1].assign(dataset=r[0]) for r in res]);matched=pd.concat([r[2] for r in res]);weights=pd.concat([r[3] for r in res]);prefix='logistic' if MODEL_KIND=='logistic' else 'mlp_wd';selected.to_csv(out/f'{prefix}_selected_results.csv',index=False);matched.to_csv(out/f'{prefix}_matched_cost.csv',index=False);weights.to_csv(out/f'{prefix}_margin_weights.csv',index=False)
    title='Logistic-regression LIFT benchmark' if MODEL_KIND=='logistic' else f'MLP increased-weight-decay benchmark ({WEIGHT_DECAY:g})';desc=('one linear logit with a minimum 1,000-update convergence budget' if MODEL_KIND=='logistic' else 'MLP [64,32] with the original 30/40 epoch budgets')
    report=[f'# {title}','', f'Classifier: {desc}, BCEWithLogitsLoss, AdamW lr=1e-3, weight decay={WEIGHT_DECAY:g}; all other benchmark rules are retained.','', '## Selected results','',markdown_table(selected),'','## Margin weights','',markdown_table(weights),'','## Matched-cost results','',markdown_table(matched)];(out/f'{prefix}_margin_anchor_benchmark.md').write_text('\n'.join(report));print(selected[['dataset','method','native_parameter','current_validity','mean_L2_cost','mean_future_validity','near_in_range_worst']].to_string(index=False))
if __name__=='__main__':main()
