"""Independent saved-output checks and seed-level aggregation for task103."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain
from lift_ce.counterfactuals.geodesic_excess import exact_w2
from torch.func import functional_call

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 rows=[];future=[];checks=[];accuracy=[];changes=[]
 for worker in sorted(a.input.glob('seed*_*_*')):
  if not worker.is_dir() or not (worker/'complete.json').exists():continue
  meta=json.loads((worker/'metadata.json').read_text());prep=load(worker/'prepared.pt');d=prep['data'];seed=meta['seed'];ds=worker.name.split('_')[1];clf=worker.name.split('_')[2]
  model=TabularMLP(d['train_x'].shape[1],prep['hidden']);model.load_state_dict(load(worker/'current.pt'));model.eval()
  domain=MixedDomain(d['train_x'],prep['groups']);source=prep['sources']['test'];bank=load(worker/'future_test.pt')
  with torch.no_grad(): acc=float(((model(d['test_x']).sigmoid()>=.5)==d['test_y'].bool()).float().mean())
  accuracy.append(dict(seed=seed,dataset=ds,classifier=clf,test_accuracy=acc,n_test=len(d['test_y'])))
  frame=pd.read_csv(worker/'results.csv')
  for row in frame.to_dict('records'):
   method=row['method'];real=row['realization'];obj=load(worker/f'test_{method}.pt');z=domain.project(obj['cf'],True) if real=='hard' else obj['cf'];saved=load(worker/f'scores_{method}_{real}.pt')
   with torch.no_grad():
    cv=model(z).sigmoid()>=.5
    pred=torch.stack([functional_call(model,(params,buffers),(z,)).sigmoid()>=.5 for params,buffers in bank['states']])
   cost=torch.linalg.vector_norm(z-source,dim=1)
   assert torch.equal(pred,saved['predictions']) and torch.equal(cv,saved['current'])
   assert abs(float(cost.mean())-row['Cost'])<1e-5
   error=float((domain.project(z)-z).abs().max())
   assert error<1e-4,(worker,method,error)
   if method.startswith('grace') and real!='hard':
    base=load(worker/f"test_{method.removeprefix('grace_')}.pt")['cf']
    detour=exact_w2(base,z).distance+exact_w2(z,prep['p1']).distance
    assert abs(detour-row['detour'])<1e-4
    before=torch.linalg.vector_norm(base-source,dim=1);delta=cost-before
    for i,value in enumerate(delta):changes.append(dict(seed=seed,dataset=ds,classifier=clf,method=method,factual_id=prep['source_ids']['test'][i],base_proximity=float(before[i]),grace_proximity=float(cost[i]),change=float(value)))
   checks.append(dict(worker=str(worker),method=method,realization=real,passed=True,simplex_error=error))
   row.update(seed=seed,seconds=obj.get('seconds'),raw_path=str(worker));rows.append(row)
   for j,scenario in enumerate(bank['records']):future.append(dict(seed=seed,dataset=ds,classifier=clf,method=method,realization=real,future_index=j,FV=float(pred[j].float().mean()),**{("future_"+k if k in ("seed","realization") else k):v for k,v in scenario.items()}))
 df=pd.DataFrame(rows)
 if df.empty:raise RuntimeError('No completed workers')
 df.to_csv(a.output/'per_seed_metrics.csv',index=False);pd.DataFrame(future).to_csv(a.output/'per_future_model.csv',index=False);pd.DataFrame(checks).to_csv(a.output/'independent_checks.csv',index=False);pd.DataFrame(accuracy).to_csv(a.output/'classifier_accuracy.csv',index=False)
 ch=pd.DataFrame(changes);ch.to_csv(a.output/'per_factual_proximity_changes.csv',index=False)
 if not ch.empty:
  ch.groupby(['dataset','classifier','method','seed']).change.agg(median='median',p95=lambda x:x.quantile(.95),maximum='max').to_csv(a.output/'proximity_change_summary.csv')
 agg=df.groupby(['dataset','classifier','method','realization']).agg(seeds=('seed','nunique'),FV_mean=('FV','mean'),FV_sd=('FV','std'),Cost_mean=('Cost','mean'),Cost_sd=('Cost','std'),CV_mean=('current_validity','mean'),failure_aware_FV_mean=('failure_aware_FV','mean'),failures=('native_failures','sum'),seconds_mean=('seconds','mean')).reset_index()
 agg.to_csv(a.output/'aggregated.csv',index=False)
 text=['# Two-seed full-baseline replication','',f'Raw source: `{a.input}`. Completed workers: {len(accuracy)}/16. Saved-output checks: {len(checks)}.','', 'Mean and sample SD use training/split seeds as units (two seeds); future-model scores are nested and include repeated scenarios. FV/CV fractions in CSV. Mixed relaxed and hard results are distinct. Failed generations remain in denominators. Seed0 is preserved and not pooled because these adapters differ.','', '| Dataset | Classifier | Method | Realization | Seeds | FV (%) | Proximity | Native failures |','|---|---|---|---|---:|---:|---:|---:|']
 for r in agg.itertuples():text.append(f'| {r.dataset} | {r.classifier} | {r.method} | {r.realization} | {r.seeds} | {100*r.FV_mean:.2f} ± {100*r.FV_sd:.2f} | {r.Cost_mean:.4f} ± {r.Cost_sd:.4f} | {r.failures} |')
 (a.output/'report.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
