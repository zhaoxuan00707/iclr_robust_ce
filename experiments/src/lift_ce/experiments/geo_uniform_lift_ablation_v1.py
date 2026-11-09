"""Uniform-anchor kNN favorable-region geodesic OT ablation."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr,spearmanr
from sklearn.neighbors import NearestNeighbors
from .path_lift_decision_compatible_ot_v1 import setting_artifacts,pair_violations,score
from .constrained_lift_ablation_v1 import exact_constrained_plan,evaluate_destination
from .stability_weighted_anchor_v1 import weighted_sinkhorn
from .heloc_benchmark_milestone import valid

KS=(5,10,20)

def graph(model,nodes,k,batch=32768):
 nn=NearestNeighbors(n_neighbors=min(k+1,len(nodes))).fit(nodes.numpy());dist,ind=nn.kneighbors(nodes.numpy());src=np.repeat(np.arange(len(nodes)),ind.shape[1]-1);dst=ind[:,1:].reshape(-1);ed=dist[:,1:].reshape(-1);ok=np.zeros(len(src),bool);grid=torch.linspace(0,1,5)
 for a in range(0,len(src),batch):
  ii=torch.tensor(src[a:a+batch]);jj=torch.tensor(dst[a:a+batch]);x=(1-grid[:,None,None])*nodes[ii][None]+grid[:,None,None]*nodes[jj][None];ok[a:a+batch]=(score(model,x.flatten(0,1)).view(5,-1)>=.5).all(0).numpy()
 r=np.r_[src[ok],dst[ok]];c=np.r_[dst[ok],src[ok]];v=np.r_[ed[ok],ed[ok]];g=coo_matrix((v,(r,c)),shape=(len(nodes),len(nodes))).tocsr();gu=coo_matrix((np.ones(len(v)),(r,c)),shape=g.shape).tocsr();return g,gu,float(ok.mean()),len(src),int(ok.sum())

def geometry(model,base,fav,straight,eu,k):
 t=time.perf_counter();bv=valid(model,base).numpy();bi=np.flatnonzero(bv);nodes=torch.cat((base[bv],fav));g,gu,accept,ncand,nkeep=graph(model,nodes,k);sp=dijkstra(g,directed=False,indices=np.arange(len(bi)));hp=dijkstra(gu,directed=False,indices=np.arange(len(bi)),unweighted=True);geo=np.full(eu.shape,np.inf);hops=np.full(eu.shape,np.nan);geo[bi]=sp[:,len(bi):];hops[bi]=hp[:,len(bi):];geo=np.maximum(geo,eu);geo[straight]=eu[straight];hops[straight]=1;connected=np.isfinite(geo);return geo,hops,connected,{'k':k,'graph_nodes':len(nodes),'invalid_PCF_nodes':int((~bv).sum()),'invalid_P1_nodes':int((~valid(model,fav).numpy()).sum()),'candidate_edges':ncand,'kept_edges':nkeep,'edge_acceptance':accept,'runtime_seconds':time.perf_counter()-t}

def trex_rows(ds,setting,front,roots):
 if setting=='logistic':return front[front.method=='T-Rex'].rename(columns={'native_parameter':'lambda'}).copy()
 ap=Path('/mnt/big_disk/xu.zhao/robust_ce_results/heloc_prior_balanced_audit_seed0/common_candidate_frontier.csv') if ds=='heloc' else roots['diabetes_current']/'diabetes_candidate_frontier.csv';a=pd.read_csv(ap);a=a[a.method=='trex'].set_index('candidate_id');rows=[]
 for r in front[front.method=='T-Rex'].itertuples():
  q=a.loc[int(r.candidate_id)];rows.append({'method':'T-Rex','lambda':r.native_parameter,'mean_L2_cost':r.mean_l2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':q.mean_nn_target_distance,'LOF':q.mean_lof,'population_target_distance':q.population_target_distance})
 return pd.DataFrame(rows)

def run(ds,setting,home,roots):
 d,source,ids,base,current,proxies,futures,front,tensors,oldtrex=setting_artifacts(ds,setting,home,roots);fav=d['train_x'][valid(current,d['train_x'])];w=torch.ones(len(fav))/len(fav);eu=torch.cdist(base,fav);_,_,_,mmin=pair_violations(current,base,fav);straight=(mmin>=-1e-10).numpy();pu=weighted_sinkhorn(eu.square(),w,iterations=1000,tol=1e-8);du=(pu/pu.sum(1,keepdim=True).clamp_min(1e-30))@fav;label=f'{ds}__{setting}';uniform=evaluate_destination(ds,setting,source,base,fav,current,proxies,futures,du,'Uniform-LIFT',np.nan,float(straight.mean()));allres=[uniform];aud=[];couplings=[];pair_primary=None;primary_plan=None;primary_dest=None;primary_ratio=None
 for k in KS:
  geo,hops,conn,ga=geometry(current,base,fav,straight,eu.numpy(),k);finite=geo[conn];penalty=(finite.max()*10 if len(finite) else eu.numpy().max()*100);numeric=geo.copy();numeric[~conn]=penalty;plan,eta,_,_=exact_constrained_plan(numeric**2,conn,w.numpy());pt=torch.tensor(plan,dtype=base.dtype);dest=(pt/pt.sum(1,keepdim=True).clamp_min(1e-30))@fav;name='Geo-LIFT' if k==10 else f'Geo-LIFT-k{k}';q=evaluate_destination(ds,setting,source,base,fav,current,proxies,futures,dest,name,eta,float(straight.mean()));q['k']=k;allres.append(q);ratio=np.full_like(geo,np.nan);ratio[conn]=geo[conn]/np.maximum(eu.numpy()[conn],1e-12);ga.update({'setting':label,'connected_pair_fraction':float(conn.mean()),'disconnected_pair_rate':float((~conn).mean()),'eta_geo_star':eta,'mean_geo_over_L2':float(np.nanmean(ratio)),'median_geo_over_L2':float(np.nanmedian(ratio)),'p90_geo_over_L2':float(np.nanquantile(ratio,.9)),'dominant_target_change_rate':float((pu.argmax(1).numpy()!=plan.argmax(1)).mean()),'coupling_L1_difference':float(np.abs(pu.numpy()-plan).sum()),'max_source_marginal_error':float(np.abs(plan.sum(1)-1/len(base)).max()),'max_target_marginal_error':float(np.abs(plan.sum(0)-w.numpy()).max())});aud.append(ga);couplings.append(pd.DataFrame({'setting':label,'k':k,'source_id':np.asarray(ids),'uniform_dominant_target':pu.argmax(1).numpy(),'geo_dominant_target':plan.argmax(1),'dominant_changed':pu.argmax(1).numpy()!=plan.argmax(1)}))
  if k==10:
   ii,jj=np.indices(geo.shape);pair_primary=pd.DataFrame({'setting':label,'source_id':np.asarray(ids)[ii.ravel()],'target_id':jj.ravel(),'L2_distance':eu.numpy().ravel(),'geodesic_distance':geo.ravel(),'geo_over_L2':ratio.ravel(),'connected':conn.ravel(),'hop_count':hops.ravel()});primary_plan=plan;primary_dest=dest;primary_ratio=ratio
 # Source distortion under Uniform coupling and source-level future failure at calibrated Uniform lambda.
 up=uniform[uniform.calibrated_flag==1].iloc[0];z=(1-float(up['lambda']))*base+float(up['lambda'])*du;dist=[]
 for i in range(len(base)):
  cond=pu[i]/pu[i].sum();rr=torch.tensor(np.nan_to_num(primary_ratio[i],nan=0.),dtype=cond.dtype);mass=torch.tensor(np.isfinite(primary_ratio[i]),dtype=cond.dtype);den=float((cond*mass).sum());di=(float((cond*rr).sum()/den) if den>1e-30 else np.nan);vals=[]
  for e,s,k,rg,fm in futures:vals.append(float(valid(fm,z[i:i+1]).item()))
  dist.append({'setting':label,'source_id':int(ids[i]),'uniform_coupling_weighted_distortion':di,'future_validity':np.mean(vals),'future_flip_rate':1-np.mean(vals)})
 sd=pd.DataFrame(dist);ok=np.isfinite(sd.uniform_coupling_weighted_distortion);aud[-2 if len(aud)>1 else -1] if False else None
 pear=pearsonr(sd.loc[ok,'uniform_coupling_weighted_distortion'],sd.loc[ok,'future_validity']).statistic if ok.sum()>2 else np.nan;spear=spearmanr(sd.loc[ok,'uniform_coupling_weighted_distortion'],sd.loc[ok,'future_validity']).statistic if ok.sum()>2 else np.nan;sd['pearson_distortion_vs_validity']=pear;sd['spearman_distortion_vs_validity']=spear
 return pd.concat(allres,ignore_index=True,sort=False),pd.DataFrame(aud),pd.concat(couplings,ignore_index=True),pair_primary,sd,front

def matched(results,fronts,roots):
 out=[]
 for setting,g in results.groupby('setting'):
  ds,kind=setting.split('__');t=trex_rows(ds,kind,fronts[setting],roots);pool=pd.concat((g[g.method.isin(['Uniform-LIFT','Geo-LIFT'])],t),ignore_index=True,sort=False);methods=['T-Rex','Uniform-LIFT','Geo-LIFT'];lo=max(pool[pool.method==m].mean_L2_cost.min() for m in methods);hi=min(pool[pool.method==m].mean_L2_cost.max() for m in methods)
  for b in np.linspace(lo,hi,20):
   for method in methods:
    q=pool[(pool.method==method)&(pool.mean_L2_cost<=b+1e-9)].sort_values(['near_in_range_worst','mean_L2_cost'],ascending=[False,True]);r=q.iloc[0];out.append({'setting':setting,'cost_budget':b,'method':method,'actual_L2_cost':r.mean_L2_cost,'current_validity':r.current_validity,'mean_future_validity':r.mean_future_validity,'near_in_range_worst':r.near_in_range_worst,'NN_target_distance':r.NN_target_distance,'LOF':r.LOF,'population_target_distance':r.population_target_distance})
 return pd.DataFrame(out)

def main():
 p=argparse.ArgumentParser()
 for x in ('heloc-data-home','diabetes-data-home','corrected-root','margin-root','logistic-root','heloc-current-root','diabetes-current-root','output'):p.add_argument('--'+x,required=True)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);roots={'corrected':Path(a.corrected_root),'margin':Path(a.margin_root),'logistic':Path(a.logistic_root),'heloc_current':Path(a.heloc_current_root),'diabetes_current':Path(a.diabetes_current_root)};spec=[('heloc','baseline_mlp',a.heloc_data_home),('heloc','logistic',a.heloc_data_home),('diabetes','baseline_mlp',a.diabetes_data_home)];res=[run(*s,roots) for s in spec];names=['geo_uniform_results','geo_uniform_graph_audit','geo_uniform_coupling','geo_uniform_pairwise_distances','geo_uniform_source_distortion'];frames=[]
 for i,n in enumerate(names):f=pd.concat([r[i] for r in res],ignore_index=True);f.to_csv(out/f'{n}.csv',index=False);frames.append(f)
 results,aud,coup,pair,sd=frames;fronts={f'{d}__{s}':r[5] for (d,s,_),r in zip(spec,res)};mc=matched(results,fronts,roots);mc.to_csv(out/'geo_uniform_matched_cost.csv',index=False);sel=results[results.calibrated_flag==1];(out/'geo_uniform_lift_ablation_v1.md').write_text('# Geo-Uniform-LIFT ablation v1\n\n'+sel.to_csv(index=False)+'\n'+aud.to_csv(index=False));print(sel[['setting','method','k','lambda','current_validity','mean_future_validity','near_in_range_worst','mean_L2_cost']].to_string(index=False));print(aud.to_string(index=False))
if __name__=='__main__':main()
