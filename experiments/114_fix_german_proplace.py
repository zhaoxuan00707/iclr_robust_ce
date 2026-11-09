"""Retry only license-skipped German MLP CFs; preserve frozen selection and successes."""
import argparse,copy,json,os,sys,time
from pathlib import Path
import torch,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/src'),str(ROOT/'experiments/vendor/RobustX'),str(ROOT/'experiments/vendor/ellice')]
from lift_ce.experiments.appendix_tabular import load,save,generate,score,MixedDomain
from lift_ce.models.tabular import TabularMLP
PARENT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/tabular_replication/20261109_full_v1')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--previous',type=Path);ap.add_argument('--timeout',type=float,default=30.);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
 src=PARENT/f'seed{a.seed}_german_mlp';prep=load(src/'prepared.pt');cfg=json.loads((src/'metadata.json').read_text())['config'];cfg['proplace_license_fallback']=True;cfg['baseline_instance_timeout_seconds']=a.timeout
 d=prep['data'];domain=MixedDomain(d['train_x'],prep['groups']);model=TabularMLP(d['train_x'].shape[1],prep['hidden']);model.load_state_dict(load(src/'current.pt'));model.eval()
 param=json.loads((src/'selection.json').read_text())['proplace']['param'];old=load((a.previous or src)/'test_proplace.pt');ids=[i for i,e in enumerate(old['errors']) if e];source=prep['sources']['test'];start=time.time()
 repaired=generate('proplace',param,model,d,source[ids],domain,cfg,a.output/'retries.pt',categorical_mode='relaxed')
 merged=copy.deepcopy(old)
 for j,i in enumerate(ids):
  merged['cf'][i]=repaired['cf'][j];merged['errors'][i]=repaired['errors'][j];merged['native_failure'][i]=repaired['native_failure'][j]
 merged['status']='executed' if repaired['status']=='executed' else repaired['status'];save(merged,a.output/'test_proplace.pt')
 keep=[i for i in range(len(source)) if i not in ids]
 assert torch.equal(merged['cf'][keep],old['cf'][keep])
 assert float((domain.project(merged['cf'])-merged['cf']).abs().max())<1e-4
 bank=load(src/'future_test.pt');rows=[]
 for real in ('relaxed','hard'):
  z=domain.project(merged['cf'],True) if real=='hard' else merged['cf'];metrics=score(model,z,bank['states'],source);save(metrics,a.output/f'scores_{real}.pt')
  usable=~torch.tensor(merged['native_failure']);row=dict(seed=a.seed,dataset='german',classifier='mlp',method='proplace',realization=real,n=len(source),param=param,status=merged['status'],native_failures=sum(merged['native_failure']),FV=metrics['FV'],Cost=metrics['Cost'],current_validity=metrics['current_validity'],failure_aware_FV=float((metrics['predictions']&usable[None]).float().mean()))
  rows.append(row)
 pd.DataFrame(rows).to_csv(a.output/'results.csv',index=False)
 provenance=dict(instance_timeout_seconds=a.timeout,parent=str(src),previous=str(a.previous) if a.previous else None,seed=a.seed,selected_param=param,retried_indices=ids,successful_originals_unchanged=True,seconds=time.time()-start,remaining_errors=[(i,e) for i,e in enumerate(merged['errors']) if e],master_records=repaired['master_solver_records'])
 (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(rows),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'complete':True,'remaining_failures':sum(merged['native_failure'])}))
if __name__=='__main__':main()
