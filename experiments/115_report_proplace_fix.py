"""Verify repaired outputs against frozen assets; export corrected two-seed row."""
import sys,json,argparse
from pathlib import Path
import torch,pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments/src'),str(ROOT/'experiments/vendor/RobustX'),str(ROOT/'experiments/vendor/ellice')]
from lift_ce.experiments.appendix_tabular import load,score,MixedDomain
from lift_ce.models.tabular import TabularMLP
from lift_ce.experiments.run_official_proplace import FrozenFrameDataset
from lift_ce.baselines.tabular_replication_adapters import SignedDeltaEvaluator
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.lib.tasks.ClassificationTask import ClassificationTask

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed2-version',type=int,default=2);ap.add_argument('--output',type=Path,default=ROOT/'results/104_proplace_license_fix');args=ap.parse_args()
 torch.set_num_threads(1);rows=[];checks=[];future=[]
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 for seed in (1,2):
  raw=Path(f'/home/xu.zhao/robust_ce_results/iclr_runs/proplace_license_fix/20261109_seed{seed}_v{args.seed2_version if seed==2 else 1}')
  provenance=json.loads((raw/'provenance.json').read_text());parent=Path(provenance['parent']);prep=load(parent/'prepared.pt');d=prep['data'];obj=load(raw/'test_proplace.pt');old=load(Path(provenance.get('previous') or str(parent))/'test_proplace.pt');domain=MixedDomain(d['train_x'],prep['groups'])
  ids=provenance['retried_indices'];keep=[i for i in range(len(obj['cf'])) if i not in ids];assert torch.equal(obj['cf'][keep],old['cf'][keep])
  model=TabularMLP(d['train_x'].shape[1],prep['hidden']);model.load_state_dict(load(parent/'current.pt'));model.eval()
  wrapped=SimpleNNModel(d['train_x'].shape[1],prep['hidden'],1,seed=0)
  layers=[v for v in model.features if isinstance(v,torch.nn.Linear)]+[model.classifier]
  with torch.no_grad():
   for a,b in zip(layers,[v for v in wrapped.model if isinstance(v,torch.nn.Linear)]):b.weight.copy_(a.weight);b.bias.copy_(a.bias)
  data=FrozenFrameDataset(seed=0);data.data=pd.DataFrame(d['train_x'].numpy(),columns=d['feature_names']);data.data['target']=d['train_y'].numpy();evaluator=SignedDeltaEvaluator(ClassificationTask(wrapped,data))
  native=[]
  for i,z in enumerate(obj['cf']):
   robust=evaluator.evaluate(pd.Series(z.numpy(),index=d['feature_names']),delta=provenance['selected_param'],bias_delta=provenance['selected_param'],M=1000);native.append(robust)
   checks.append(dict(seed=seed,factual=i,retried=i in ids,native_robust=bool(robust),failure=bool(obj['native_failure'][i])))
  bank=load(parent/'future_test.pt')
  for real in ('relaxed','hard'):
   z=domain.project(obj['cf'],True) if real=='hard' else obj['cf'];s=score(model,z,bank['states'],prep['sources']['test']);saved=load(raw/f'scores_{real}.pt');assert torch.equal(s['predictions'],saved['predictions']);assert abs(s['Cost']-saved['Cost'])<1e-6
   rows.append(dict(seed=seed,realization=real,FV=100*s['FV'],proximity=s['Cost'],current_validity=100*s['current_validity'],failures=sum(obj['native_failure']),native_robust_fraction=float(np.mean(native)) if real=='relaxed' else None))
   for j,p in enumerate(s['predictions']):future.append(dict(seed=seed,realization=real,future_index=j,FV=float(p.float().mean())))
 a=pd.DataFrame(rows);a.to_csv(out/'per_seed.csv',index=False);pd.DataFrame(checks).to_csv(out/'native_checks.csv',index=False);pd.DataFrame(future).to_csv(out/'per_future_model.csv',index=False)
 agg=a.groupby('realization').agg(FV_mean=('FV','mean'),FV_std=('FV','std'),proximity_mean=('proximity','mean'),proximity_std=('proximity','std'));agg.to_csv(out/'summary.csv')
 (out/'report.md').write_text('# PROPLACE German Credit MLP follow-up\n\nFrozen radius .005; only previously failed/skipped cases retried. All successful original CFs unchanged. Original runs preserved. Statistics use two training seeds, sample SD. Relaxed results do not assert hard-category feasibility. Native robustness independently checked per factual in native_checks.csv.\n\n'+a.to_markdown(index=False)+'\n\n'+agg.to_markdown()+'\n')
 print(a.to_string(index=False));print(agg.to_string())
if __name__=='__main__':main()
