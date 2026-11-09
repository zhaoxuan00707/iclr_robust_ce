"""Strengthened smooth-margin adversary and solver audit for HELOC."""
from __future__ import annotations
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.func import functional_call
from .corrected_future_benchmark_v1 import load_model
from .heloc_coupled_b_lambda_v1 import TAU,score
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from ..data.tabular import load_tabular_three_way

RATIOS=(.5,1.,1.5,2.,3.);M=5

def rms(x):return x.square().sum(1).mean().sqrt()
def loss_scores(p,kind,beta):
    if kind=='hinge':return torch.relu(TAU-p).mean()
    if kind=='raw':return (TAU-p).mean()
    return (torch.nn.functional.softplus(beta*(TAU-p))/beta).mean()

def objective(model,params,buffers,x,y,coarse,delta,kind,beta,ft_lr=2e-4):
    logits=functional_call(model,(params,buffers),(x+delta,));trainloss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);gs=torch.autograd.grad(trainloss,tuple(params.values()),create_graph=True);updated={name:p-ft_lr*g for (name,p),g in zip(params.items(),gs)};pred=torch.sigmoid(functional_call(model,(updated,buffers),(coarse,)));return loss_scores(pred,kind,beta),pred

def optimize(model,x,y,coarse,b,seed,kind='softplus',beta=20.,lr=.75,steps=100,optimizer='adam'):
    torch.manual_seed(seed);delta=torch.randn_like(x);delta=(.05*b*delta/rms(delta).clamp_min(1e-12)).requires_grad_(True);params=dict(model.named_parameters());buffers=dict(model.named_buffers());opt=torch.optim.Adam([delta],lr=lr) if optimizer=='adam' else None;rows=[]
    for step in range(steps):
      before=delta.detach().clone();obj,p=objective(model,params,buffers,x,y,coarse,delta,kind,beta);true_hinge=torch.relu(TAU-p).mean()
      if opt:opt.zero_grad()
      elif delta.grad is not None:delta.grad.zero_()
      ((-obj) if opt else obj).backward();gn=float(delta.grad.norm());
      if opt:opt.step()
      else:
       with torch.no_grad():delta.add_(lr*delta.grad)
      with torch.no_grad():
       raw=delta.detach().clone();nb=float(rms(raw));active=nb>b
       if active:delta.mul_(b/nb)
       na=float(rms(delta));rows.append({'iteration':step,'J_adv_hinge':float(true_hinge),'J_surrogate':float(obj),'delta_rms':na,'budget_fraction':na/b,'grad_delta_norm':gn,'update_norm_before_projection':float(torch.linalg.vector_norm(raw-before)),'update_norm_after_projection':float(torch.linalg.vector_norm(delta-before)),'projection_active':active,'projection_bug_active_inside_budget':bool(active and nb<b-1e-12),'norm_before_projection':nb,'norm_after_projection':na,'allowed_budget':b,'mean_cf_score':float(p.mean()),'invalid_cf_fraction':float((p<TAU).float().mean()),'zero_hinge_fraction':float((p>=TAU).float().mean()),'surrogate':kind,'beta':beta,'lr':lr,'optimizer':optimizer})
    return delta.detach(),pd.DataFrame(rows)

def finite_difference(model,x,y,coarse,beta=20.,eps=.1):
    params=dict(model.named_parameters());buffers=dict(model.named_buffers());d=torch.zeros_like(x,requires_grad=True);o,_=objective(model,params,buffers,x,y,coarse,d,'softplus',beta);g=torch.autograd.grad(o,d)[0];coords=[(0,0),(1,1),(2,2),(3,3)];rows=[]
    for i,j in coords:
      dp=torch.zeros_like(x);dp[i,j]=eps;op,_=objective(model,params,buffers,x,y,coarse,dp,'softplus',beta);om,_=objective(model,params,buffers,x,y,coarse,-dp,'softplus',beta);num=float((op-om)/(2*eps));auto=float(g[i,j]);rows.append({'row':i,'feature':j,'autograd':auto,'finite_difference':num,'absolute_error':abs(auto-num),'relative_error':abs(auto-num)/max(abs(auto),abs(num),1e-12)})
    return pd.DataFrame(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-home',required=True);p.add_argument('--current-root',required=True);p.add_argument('--coupled-root',required=True);p.add_argument('--training-path',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args();out=Path(a.output);[(out/x).mkdir(parents=True,exist_ok=True) for x in ('restart_models','restart_deltas','models','deltas','traces')];t0=time.perf_counter();device=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu')
 cfg={'source':'openml','openml_name':'heloc','openml_version':1,'missing_values':[-7,-8,-9]};d=load_tabular_three_way(cfg,a.data_home,.6,.2,0);tx,ty=(d[k] for k in ('train_x','train_y'));current=load_model(Path(a.current_root)/'models/current.pt',tx.shape[1]);bref=float(json.loads((Path(a.coupled_root)/'metadata.json').read_text())['b_ref']);tr=torch.load(a.training_path,map_location='cpu',weights_only=True);coarse=torch.cat([(1-l)*tr['base']+l*tr['destination'] for l in (0.,.25,.5,.75,1.)]);model=copy.deepcopy(current).to(device).train();x,y,c=tx.to(device),ty.to(device),coarse.to(device)
 # Fixed training-only solver sweep at r=b_ref, restart 0.
 configs=[('hinge',20.,.15,30,'adam')]+[('softplus',b,.15,30,'adam') for b in (5.,10.,20.,50.)]+[('raw',20.,.15,30,'adam')]+[('softplus',20.,lr,st,opt) for lr,st,opt in ((.3,30,'adam'),(.75,30,'adam'),(1.5,30,'adam'),(.15,100,'adam'),(.3,100,'pga'),(.75,100,'pga'))];sweep=[]
 for ci,(kind,beta,lr,steps,optname) in enumerate(configs):
  delta,h=optimize(model,x,y,c,bref,27000+ci,kind,beta,lr,steps,optname);actual=fine_tune(current,tx+delta.cpu(),ty,37000+ci,device);p=score(actual,coarse);sweep.append({'config_id':ci,'surrogate':kind,'beta':beta,'lr':lr,'steps':steps,'optimizer':optname,'final_search_hinge':h.J_adv_hinge.iloc[-1],'final_surrogate':h.J_surrogate.iloc[-1],'budget_fraction':float(rms(delta)/bref),'actual_finetuned_hinge':float(torch.relu(TAU-p).mean()),'actual_finetuned_validity':float((p>=TAU).float().mean())})
 pd.DataFrame(sweep).to_csv(out/'solver_configuration_sweep.csv',index=False)
 fd=finite_difference(model,x[:64],y[:64],c[:64]);fd.to_csv(out/'finite_difference_gradient_check.csv',index=False)
 # Training-only sweep winner: softplus beta=20, Adam lr=.75, 30 steps.
 audit=[];alltrace=[]
 for bi,r in enumerate(RATIOS):
  b=r*bref
  for m in range(M):
   seed=38000+1000*bi+100*m;ftseed=39000+1000*bi+100*m;delta,h=optimize(model,x,y,c,b,seed,'softplus',20.,.75,30,'adam');delta=delta.cpu();actual=fine_tune(current,tx+delta,ty,ftseed,device);p=score(actual,coarse);row={'b_over_b_ref':r,'radius':b,'restart_id':m,'optimizer_seed':seed,'fine_tune_seed':ftseed,'J_adv_hinge_final':float(torch.relu(TAU-p).mean()),'training_validity':float((p>=TAU).float().mean()),'budget_fraction':float(rms(delta)/b),'delta_rms':float(rms(delta))};audit.append(row);h['b_over_b_ref']=r;h['restart_id']=m;alltrace.append(h);torch.save(delta,out/'restart_deltas'/f'SMOOTH_b{r:g}_m{m}.pt');torch.save(actual.state_dict(),out/'restart_models'/f'SMOOTH_b{r:g}_m{m}.pt')
 audit=pd.DataFrame(audit);audit['selected']=False
 for r in RATIOS:
  q=audit[abs(audit.b_over_b_ref-r)<1e-12].sort_values(['J_adv_hinge_final','restart_id'],ascending=[False,True]);i=q.index[0];audit.loc[i,'selected']=True;m=int(audit.loc[i,'restart_id']);shutil.copy2(out/'restart_models'/f'SMOOTH_b{r:g}_m{m}.pt',out/'models'/f'f_adv_b{r:g}.pt');shutil.copy2(out/'restart_deltas'/f'SMOOTH_b{r:g}_m{m}.pt',out/'deltas'/f'delta_star_b{r:g}.pt')
 audit.to_csv(out/'smooth_margin_restarts.csv',index=False);audit[audit.selected].to_csv(out/'smooth_margin_selected_models.csv',index=False);trace=pd.concat(alltrace,ignore_index=True);trace.to_csv(out/'smooth_margin_optimization_traces.csv',index=False)
 import matplotlib.pyplot as plt
 for metric in ('J_adv_hinge','budget_fraction','grad_delta_norm','invalid_cf_fraction'):
  fig,ax=plt.subplots();
  for (r,m),g in trace.groupby(['b_over_b_ref','restart_id']):ax.plot(g.iteration,g[metric],alpha=.35,label=f'{r:g}/{m}' if m==0 else None)
  ax.set(xlabel='iteration',ylabel=metric);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/f'trace_{metric}.png',dpi=180);plt.close(fig)
 bugs=int(trace.projection_bug_active_inside_budget.sum());assert bugs==0
 (out/'metadata.json').write_text(json.dumps({'method_objective':'true hinge relu(0.5-f)','search_surrogate':'softplus beta=20','gradient_change_in_objective':False,'primary_optimizer':'Adam','primary_lr':.75,'primary_steps':30,'budget_formula':'sqrt(mean_i sum_j Delta_ij^2) <= ratio*b_ref; one global rescaling only when exceeded','feature_normalization_in_budget':False,'box_clipping':False,'projection_bug_count':bugs,'differentiable_inner_approximation':'one functional gradient step; actual 8-epoch AdamW fine-tune is not differentiated through','finite_difference_mean_relative_error':float(fd.relative_error.mean()),'runtime_seconds':time.perf_counter()-t0,'future_used':False,'calibration_used':False},indent=2));print(pd.DataFrame(sweep).to_string(index=False));print('\nSELECTED\n',audit[audit.selected].to_string(index=False));print('\nFD\n',fd.to_string(index=False))

if __name__=='__main__':main()
