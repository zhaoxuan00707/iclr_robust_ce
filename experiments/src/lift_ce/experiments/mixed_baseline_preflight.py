"""Calibration-only smoke check of opt-in mixed-domain baseline adapters."""
import argparse,json,time
from pathlib import Path
import pandas as pd
import torch
from ..models.tabular import TabularMLP
from ..counterfactuals.mixed_update_aware import MixedDomain
from .appendix_tabular import generate,load

PARENT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--categorical-mode',choices=['relaxed','hard'],default='relaxed')
    p.add_argument('--n',type=int,default=1);a=p.parse_args();torch.set_num_threads(1);rows=[]
    for dataset in ['german','fico']:
      for architecture in ['mlp','logistic']:
        parent=PARENT/f'{dataset}_{architecture}';prepared=load(parent/'prepared.pt');d=prepared['data']
        cfg=json.loads((parent/'metadata.json').read_text())['config'];domain=MixedDomain(d['train_x'],prepared['groups'])
        model=TabularMLP(d['train_x'].shape[1],prepared['hidden']);model.load_state_dict(load(parent/'current.pt'));model.eval()
        source=prepared['sources']['cal'][:a.n]
        for method in ['roar','proplace']:
            param=cfg['baseline_grid'][method][0]
            name=f'{dataset}_{architecture}_{method}_{a.categorical_mode}'
            print('START',name,flush=True)
            result=generate(method,param,model,d,source,domain,cfg,a.output/(name+'.pt'),categorical_mode=a.categorical_mode)
            z=result['cf'];hard=domain.project(z,True)
            with torch.no_grad():cv=float((model(z).sigmoid()>=.5).float().mean());hard_cv=float((model(hard).sigmoid()>=.5).float().mean())
            simplex_error=float((domain.project(z)-z).abs().max());hard_error=float((hard-z).abs().max())
            assert simplex_error<1e-5,(name,simplex_error)
            if a.categorical_mode=='hard':assert hard_error<1e-5
            row=dict(dataset=dataset,classifier=architecture,method=method,mode=a.categorical_mode,n=len(z),param=param,status=result['status'],native_failures=sum(result['native_failure']),current_validity=cv,hard_current_validity=hard_cv,proximity=float((z-source).norm(dim=1).mean()),simplex_error=simplex_error,hard_error=hard_error,seconds=result['seconds'],errors=json.dumps(result['errors']))
            rows.append(row);pd.DataFrame(rows).to_csv(a.output/'preflight.csv',index=False);print(row,flush=True)
    (a.output/'complete.json').write_text(json.dumps({'complete':True,'time':time.time(),'calibration_only':True,'rows':len(rows),'parent':str(PARENT)},indent=2))
if __name__=='__main__':main()
