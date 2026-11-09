"""Gradient-sensitive Wasserstein adversaries for frozen HELOC LIFT."""
from __future__ import annotations
import argparse,copy,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import pearsonr,spearmanr
from torch.func import functional_call
from .heloc_coupled_b_lambda_v1 import TAU,adverse_directions
from .corrected_future_benchmark_v1 import load_model
from .path_lift_decision_compatible_ot_v1 import score
from ..data.tabular import load_tabular_three_way

RATIOS=(.5,1.,1.5,2.,3.);M=5;KINDS=('ADV-CF','ADV-GRAD','ADV-HYBRID')

def flat_params(m):return torch.cat([p.detach().flatten().cpu() for p in m.parameters()])
def cosine(a,b):
    den=float(a.norm()*b.norm());return float(a@b/den) if den>1e-12 else np.nan
def grad_vector(model,x,y,create_graph=False):
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);g=torch.autograd.grad(loss,tuple(model.parameters()),create_graph=create_graph);return torch.cat([v.reshape(-1) for v in g])
def param_gradient(model,params,buffers,x,y,create_graph=True):
    logits=functional_call(model,(params,buffers),(x,));loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);g=torch.autograd.grad(loss,tuple(params.values()),create_graph=create_graph);return g,torch.cat([v.reshape(-1) for v in g])

def optimize_delta(model,x,y,coarse,b,kind,seed,steps=30,outer_lr=.15,ft_lr=2e-4):
    torch.manual_seed(seed);delta=(.02*b*torch.randn_like(x));delta=delta/delta.square().sum(1).mean().sqrt().clamp_min(1e-12);delta=(.05*b*delta).requires_grad_(True);opt=torch.optim.Adam([delta],lr=outer_lr)
    params=dict(model.named_parameters());buffers=dict(model.named_buffers());_,g0=param_gradient(model,params,buffers,x,y,False);g0=g0.detach();scale_grad=None;scale_cf=None;history=[]
    for step in range(steps):
        gs,g=param_gradient(model,params,buffers,x+delta,y,True);jg=torch.linalg.vector_norm(g-g0)
        updated={name:p-ft_lr*gg for (name,p),gg in zip(params.items(),gs)};pcf=torch.sigmoid(functional_call(model,(updated,buffers),(coarse,)));jcf=torch.relu(TAU-pcf).mean()
        if scale_grad is None:scale_grad=jg.detach().clamp_min(1e-6);scale_cf=jcf.detach().clamp_min(1e-6)
        obj=jg/scale_grad if kind=='ADV-GRAD' else jcf/scale_cf+jg/scale_grad
        opt.zero_grad();(-obj).backward();opt.step()
        with torch.no_grad():
            rms=delta.square().sum(1).mean().sqrt();
            if rms>b:delta.mul_(b/rms)
            history.append((step,float(rms),float(jg.detach()),float(jcf.detach()),float(obj.detach())))
    return delta.detach(),pd.DataFrame(history,columns=['step','pre_projection_rms','J_grad','J_CF','objective'])

def fine_tune(current,x,y,seed,device,epochs=8):
    torch.manual_seed(seed);m=copy.deepcopy(current).to(device);m.train();opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
    for ep in range(epochs):
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+ep))
        for a in range(0,len(x),256):
            q=order[a:a+256];loss=torch.nn.functional.binary_cross_entropy_with_logits(m(x[q].to(device)),y[q].to(device));opt.zero_grad();loss.backward();opt.step()
    m.eval();return m.cpu()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-home',required=True);p.add_argument('--current-root',required=True);p.add_argument('--future-root',required=True);p.add_argument('--coupled-root',required=True);p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=30);a=p.parse_args();root=Path(a.coupled_root);od=root/'gradient_adversary';od.mkdir(exist_ok=True);(od/'models').mkdir(exist_ok=True);(od/'deltas').mkdir(exist_ok=True);device=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu');start=time.perf_counter()
    cfg={'source':'openml','openml_name':'heloc','openml_version':1,'missing_values':[-7,-8,-9]};d=load_tabular_three_way(cfg,a.data_home,.6,.2,0);tx,ty,ex=(d[k] for k in ('train_x','train_y','test_x'))
    current=load_model(Path(a.current_root)/'models/current.pt',tx.shape[1]);theta0=flat_params(current);art=torch.load(root/'artifacts.pt',map_location='cpu',weights_only=True);base,fav,dest=art['base'],art['favorable'],art['destinations'];bref=float(json.loads((root/'metadata.json').read_text())['b_ref']);coarse=torch.cat([((1-l)*base+l*dest['Geo-LIFT']) for l in (0,.25,.5,.75,1.)])
    # Exact reconstruction of current ADV-CF directions for selected b ratios.
    allrat=(.5,.75,1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.);prev=[];cfvec={}
    for bi,ratio in enumerate(allrat):vec,unit=adverse_directions(current,coarse,ratio*bref,10,9300+bi,prev);prev=unit[:1];cfvec[ratio]=vec[:M]
    proxies=[];gradrows=[];hist=[];current_dev=copy.deepcopy(current).to(device);current_dev.train();xd,yd,cd=tx.to(device),ty.to(device),coarse.to(device)
    with torch.enable_grad():g0=grad_vector(current_dev,xd,yd,False).detach().cpu()
    for bi,ratio in enumerate(RATIOS):
      b=ratio*bref
      for m in range(M):
       for kind in KINDS:
        mp=od/'models'/f'{kind}_b{ratio:g}_m{m}.pt';dp=od/'deltas'/f'{kind}_b{ratio:g}_m{m}.pt'
        if kind=='ADV-CF':
            delta=cfvec[ratio][m].repeat(len(tx),1);src=(root/'models'/f'b{allrat.index(ratio)}_m{m}.pt' if allrat.index(ratio)<5 else root/'models'/f'extended_b{ratio:g}_m{m}.pt');model=load_model(src,tx.shape[1]);h=pd.DataFrame()
        elif mp.exists() and dp.exists():delta=torch.load(dp,map_location='cpu',weights_only=True);model=load_model(mp,tx.shape[1]);h=pd.DataFrame()
        else:
            delta,h=optimize_delta(current_dev,xd,yd,cd,b,kind,18000+1000*bi+100*m+(0 if kind=='ADV-GRAD' else 1),a.steps);delta=delta.cpu();model=fine_tune(current,tx+delta,ty,19000+1000*bi+100*m+(0 if kind=='ADV-GRAD' else 1),device);torch.save(delta,dp);torch.save(model.state_dict(),mp)
        if len(h):h['adversary']=kind;h['b_over_b_ref']=ratio;h['proxy_model_id']=m;hist.append(h)
        with torch.enable_grad():gd=grad_vector(copy.deepcopy(current).train(),tx+delta,ty,False).detach();dg=gd-g0
        dd=float(delta.square().sum(1).mean().sqrt());u=flat_params(model)-theta0;proxies.append({'adversary':kind,'b':b,'b_over_b_ref':ratio,'proxy_model_id':m,'delta':delta,'model':model,'delta_data':dd,'rho_data':dd/b,'Delta_g':dg,'Delta_g_norm':float(dg.norm()),'gradient_cosine_gdelta_g0':cosine(gd,g0),'Delta_theta':u,'Delta_theta_norm':float(u.norm())});gradrows.append({'adversary':kind,'b':b,'b_over_b_ref':ratio,'proxy_model_id':m,'delta_data':dd,'rho_data':dd/b,'Delta_g_norm':float(dg.norm()),'gradient_cosine_gdelta_g0':cosine(gd,g0)})
    if hist:pd.concat(hist).to_csv(od/'optimization_history.csv',index=False)
    # Cross-seed Delta_g cosine within each adversary/b.
    for q in gradrows:
        peers=[r['Delta_g'] for r in proxies if r['adversary']==q['adversary'] and r['b_over_b_ref']==q['b_over_b_ref'] and r['proxy_model_id']!=q['proxy_model_id']];u=next(r['Delta_g'] for r in proxies if r['adversary']==q['adversary'] and r['b_over_b_ref']==q['b_over_b_ref'] and r['proxy_model_id']==q['proxy_model_id']);q['mean_Delta_g_cosine_other_seeds']=float(np.nanmean([cosine(u,v) for v in peers]))
    pd.DataFrame(gradrows).to_csv(od/'adversary_gradient_change.csv',index=False)
    pd.DataFrame([{k:q[k] for k in ('adversary','b','b_over_b_ref','proxy_model_id','delta_data','rho_data','Delta_g_norm','Delta_theta_norm')} for q in proxies]).to_csv(od/'adversary_parameter_drift.csv',index=False)
    # Fixed function probes and strength metrics.
    probes={'factual':ex[torch.linspace(0,len(ex)-1,min(256,len(ex))).round().long()],'P_CF':base,'Geo_path':coarse,'P1':fav[torch.linspace(0,len(fav)-1,min(256,len(fav))).round().long()]};p0={k:score(current,x) for k,x in probes.items()};func=[];cf=[]
    for q in proxies:
      for name,x in probes.items():u=score(q['model'],x)-p0[name];func.append({'adversary':q['adversary'],'b':q['b'],'b_over_b_ref':q['b_over_b_ref'],'proxy_model_id':q['proxy_model_id'],'probe_block':name,'D_func':float(torch.sqrt(u.square().mean())),'mean_score_change':float(u.mean()),'mean_adverse_score_drop':float(torch.relu(-u).mean())})
      pm=score(q['model'],base);pc=p0['P_CF'];cf.append({'adversary':q['adversary'],'b':q['b'],'b_over_b_ref':q['b_over_b_ref'],'proxy_model_id':q['proxy_model_id'],'baseline_validity':float((pc>=TAU).float().mean()),'proxy_validity':float((pm>=TAU).float().mean()),'validity_drop':float((pc>=TAU).float().mean()-(pm>=TAU).float().mean()),'margin_invalidity':float(torch.relu(TAU-pm).mean()),'mean_score_drop':float((pc-pm).mean()),'mean_adverse_score_drop':float(torch.relu(pc-pm).mean())})
    func=pd.DataFrame(func);cf=pd.DataFrame(cf);func.to_csv(od/'adversary_function_drift.csv',index=False);cf.to_csv(od/'adversary_cf_drop.csv',index=False)
    # Strongest real near/in-range failure, post-hoc only.
    audit=pd.read_csv(Path(a.future_root)/'shift_scale_audit.csv');real=[]
    for r in audit[audit.range_group.isin(['in_range','near_range'])].itertuples():
        path=Path(a.future_root)/'models'/f'{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt';
        if r.environment=='mixed_shift':path=Path(a.future_root)/'models'/f'balanced2_{r.environment}_s{r.severity:g}_k{r.future_model_id}.pt'
        fm=load_model(path,tx.shape[1]);pm=score(fm,base);drop=float((p0['P_CF']>=TAU).float().mean()-(pm>=TAU).float().mean());real.append((drop,float(torch.relu(TAU-pm).mean()),r,fm))
    dropstar,lstar,rs,fstar=max(real,key=lambda x:(x[0],x[1]));ur=score(fstar,base)-p0['P_CF'];dr=float(torch.sqrt(ur.square().mean()));post=[]
    for q in proxies:
        up=score(q['model'],base)-p0['P_CF'];row=cf[(cf.adversary==q['adversary'])&(cf.b_over_b_ref==q['b_over_b_ref'])&(cf.proxy_model_id==q['proxy_model_id'])].iloc[0];post.append({'adversary':q['adversary'],'b':q['b'],'b_over_b_ref':q['b_over_b_ref'],'proxy_model_id':q['proxy_model_id'],'real_environment':rs.environment,'real_severity':rs.severity,'real_model_id':rs.future_model_id,'parameter_drift_ratio_to_real':np.nan,'parameter_ratio_available':False,'P_CF_function_drift_ratio_to_real':float(torch.sqrt(up.square().mean()))/dr,'function_direction_cosine_to_real':cosine(up,ur),'CF_drop_ratio_to_real':row.validity_drop/dropstar if dropstar else np.nan,'real_function_drift':dr,'real_validity_drop':dropstar})
    post=pd.DataFrame(post);post.to_csv(od/'adversary_real_posthoc_alignment.csv',index=False)
    # Central strength table and training-only family selection criterion.
    strength=pd.DataFrame([{**{k:q[k] for k in ('adversary','b','b_over_b_ref','proxy_model_id','delta_data','rho_data','Delta_g_norm','Delta_theta_norm')},'D_func_PCF':float(func[(func.adversary==q['adversary'])&(func.b_over_b_ref==q['b_over_b_ref'])&(func.proxy_model_id==q['proxy_model_id'])&(func.probe_block=='P_CF')].D_func.iloc[0]),'validity_drop_PCF':float(cf[(cf.adversary==q['adversary'])&(cf.b_over_b_ref==q['b_over_b_ref'])&(cf.proxy_model_id==q['proxy_model_id'])].validity_drop.iloc[0])} for q in proxies]);strength=strength.merge(post[['adversary','b_over_b_ref','proxy_model_id','function_direction_cosine_to_real','P_CF_function_drift_ratio_to_real']],on=['adversary','b_over_b_ref','proxy_model_id']);strength.to_csv(od/'adversary_proxy_strength.csv',index=False)
    # Gradient-change predictive correlations.
    cor=[]
    for kind,g in strength.groupby('adversary'):
      for target in ('Delta_theta_norm','D_func_PCF','validity_drop_PCF'):
        cor.append({'adversary':kind,'target':target,'pearson':pearsonr(g.Delta_g_norm,g[target]).statistic if g.Delta_g_norm.std()>0 and g[target].std()>0 else np.nan,'spearman':spearmanr(g.Delta_g_norm,g[target]).statistic})
    pd.DataFrame(cor).to_csv(od/'adversary_gradient_correlations.csv',index=False)
    central=strength.groupby(['adversary','b_over_b_ref'],as_index=False).agg(mean_Delta_g=('Delta_g_norm','mean'),mean_Delta_theta=('Delta_theta_norm','mean'),mean_function_drift_PCF=('D_func_PCF','mean'),mean_CF_validity_drop=('validity_drop_PCF','mean'),mean_function_cosine_real=('function_direction_cosine_to_real','mean'),mean_function_strength_ratio_real=('P_CF_function_drift_ratio_to_real','mean'));central.to_csv(od/'adversary_mechanism_table.csv',index=False)
    # Trigger: family must reach >=2x ADV-CF mean P_CF drift at the same b.
    piv=central.pivot(index='b_over_b_ref',columns='adversary',values='mean_function_drift_PCF');ratios={k:float((piv[k]/piv['ADV-CF']).max()) for k in ('ADV-GRAD','ADV-HYBRID')};dpiv=central.pivot(index='b_over_b_ref',columns='adversary',values='mean_CF_validity_drop');drop_ratios={k:float((dpiv[k]/dpiv['ADV-CF'].replace(0,np.nan)).max()) for k in ('ADV-GRAD','ADV-HYBRID')};trigger=max(ratios.values())>=2. or max(drop_ratios.values())>=1.5
    family_score={k:float(central[central.adversary==k].mean_function_drift_PCF.mean()/central[central.adversary=='ADV-CF'].mean_function_drift_PCF.mean()+central[central.adversary==k].mean_CF_validity_drop.mean()/central[central.adversary=='ADV-CF'].mean_CF_validity_drop.mean()) for k in ('ADV-GRAD','ADV-HYBRID')};winner=max(family_score,key=family_score.get) if trigger else None
    metadata={'device':str(device),'outer_steps':a.steps,'outer_lr':.15,'fine_tune_lr':2e-4,'fine_tune_epochs':8,'gradient_parameter_subset':'all trainable parameters','hybrid_normalization':'first-step detached J_CF and J_grad scales; equal normalized weight','b_ref':bref,'b_ratios':RATIOS,'M':M,'real_future_used_for_optimization_or_selection':False,'strongest_real_reference':{'environment':rs.environment,'severity':float(rs.severity),'model_id':int(rs.future_model_id),'P_CF_function_drift':dr,'validity_drop':dropstar},'max_drift_multiplier_vs_ADV_CF':ratios,'max_CF_drop_multiplier_vs_ADV_CF':drop_ratios,'training_side_family_score':family_score,'coupled_rerun_triggered':trigger,'training_side_winner':winner,'runtime_seconds':time.perf_counter()-start};(od/'metadata.json').write_text(json.dumps(metadata,indent=2));torch.save({'deltas':{f"{q['adversary']}__{q['b_over_b_ref']}__{q['proxy_model_id']}":q['delta'] for q in proxies}},od/'optimized_perturbations.pt');print(json.dumps(metadata,indent=2));print(central.to_string(index=False))

if __name__=='__main__':main()
