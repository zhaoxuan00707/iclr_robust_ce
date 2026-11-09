"""Calibration-only ElliCE native optimizer diagnostic; does not edit live core."""
import argparse,json,sys,time,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/src'),str(ROOT/'experiments/vendor/ellice'),str(ROOT/'experiments/vendor/RobustX')]
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain
from lift_ce.baselines.official_adapters import _tabular_wrapper_class
from ellice import Data
from ellice.generators.continuous import ContinuousGenerator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 parent=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1/german_logistic')
 p=torch.load(parent/'prepared.pt',weights_only=False);d=p['data'];names=d['feature_names'];domain=MixedDomain(d['train_x'],p['groups'])
 model=TabularMLP(len(names),p['hidden']);model.load_state_dict(torch.load(parent/'current.pt',weights_only=True));model.eval()
 lo,sc=domain.low,domain.scale;train=(d['train_x']-lo)/sc;source=(p['sources']['cal']-lo)/sc
 frame=pd.DataFrame(train.numpy(),columns=names);frame['target']=d['train_y'].numpy();data=Data(dataframe=frame,target_column='target')
 wrapper=_tabular_wrapper_class(sc,lo)(model);groups=[[names[j] for j in ids] for ids in p['groups'].values()]
 configs={'default':dict(learning_rate=.1,patience=50,max_iterations=1000),'longer':dict(learning_rate=.1,patience=300,max_iterations=3000),'smaller_step':dict(learning_rate=.03,patience=300,max_iterations=3000)}
 (a.output/'protocol.json').write_text(json.dumps(dict(parent=str(parent),split='cal',configs=configs,epsilons=[.005,.01,.05],regularization=.0001,seed=0),indent=2))
 rows=[];outputs={}
 for eps in [.005,.01,.05]:
  generator=ContinuousGenerator(model=wrapper,data=data,eps=eps,reg_coef=1e-4,device='cpu')
  for label,params in configs.items():
   batch=[]
   for i,x in enumerate(source):
    torch.manual_seed(90000+i);np.random.seed(90000+i);started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always')
     result=generator.generate(pd.Series(x.numpy(),index=names),one_hot_groups=groups,target_class=1,progress_bar=False,requires='valid',**params)
    z=torch.tensor(result[names].iloc[0].to_numpy(float),dtype=torch.float32)[None];elapsed=time.perf_counter()-started;decoded=z*sc+lo
    with torch.no_grad():
     h=generator._get_penult_features(z);aug=torch.cat([h,torch.ones(len(h),1)],1);nom=(aug@generator.omega_c).item();rob=nom-torch.linalg.vector_norm(generator.Q_inv_sqrt@aug.T).item();direct=model(decoded).item()
    assert abs(nom-direct)<1e-5
    legal=bool(torch.allclose(domain.project(decoded,True),decoded,atol=1e-5))
    rows.append(dict(config=label,epsilon=eps,index=i,nominal_logit=nom,robust_logit=rob,current_valid=nom>0,ellipsoid_robust=rob>0,legal=legal,cost=float((decoded-p['sources']['cal'][i]).norm()),seconds=elapsed,warnings=' | '.join(str(w.message) for w in caught)))
    batch.append(decoded[0]);
   outputs[f'{label}_{eps}']=torch.stack(batch);pd.DataFrame(rows).to_csv(a.output/'raw.csv',index=False);torch.save(outputs,a.output/'outputs.pt');print(eps,label,'DONE',flush=True)
 frame=pd.DataFrame(rows);summary=frame.groupby(['epsilon','config']).agg(n=('index','size'),current_valid=('current_valid','mean'),ellipsoid_robust=('ellipsoid_robust','mean'),mean_robust_logit=('robust_logit','mean'),cost=('cost','mean'),seconds=('seconds','sum'),warning_count=('warnings',lambda x:sum(bool(v) for v in x))).reset_index()
 out=ROOT/'results/101_ellice_convergence';out.mkdir(exist_ok=True);frame.to_csv(out/'raw.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
 (out/'report.md').write_text('# ElliCE calibration-only convergence screen\n\nRaw: `'+str(a.output)+'`. Frozen German/logistic seed0 model and all calibration factuals. Epsilon and ellipsoid regularization unchanged across optimizer comparisons; no test scoring. Success means positive independently recomputed worst-case ellipsoid logit, not merely a nonempty returned frame.\n\n'+summary.to_markdown(index=False)+'\n')
 (a.output/'complete.json').write_text(json.dumps({'complete':True,'rows':len(frame)}));print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
