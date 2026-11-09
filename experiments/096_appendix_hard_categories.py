"""Predeclared hard categorical training-shift stress diagnostic after CF freeze."""
import argparse,copy,json,sys
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(1);rows=[];checks=[]
    for name in ['german_mlp','german_logistic','fico_mlp','fico_logistic']:
        worker=a.run/name
        if not (worker/'complete.json').exists():raise RuntimeError('Main run is incomplete')
        pop=load(worker/'prepared.pt');d=pop['data'];x=d['train_x'];y=d['train_y'];domain=MixedDomain(x,pop['groups']);cfg=json.loads((worker/'metadata.json').read_text())['config']
        model=TabularMLP(x.shape[1],pop['hidden']);model.load_state_dict(load(worker/'current.pt'));model.eval();states=[];protocol=[]
        for level,probability in enumerate([.05,.10,.20]):
          for k in range(5):
            seed=300000+100*level+k;gen=torch.Generator().manual_seed(seed);shift=x.clone()
            for ids in domain.groups.values():
                mask=torch.rand(len(x),generator=gen)<probability
                replacement=torch.randint(len(x),(len(x),),generator=gen)
                shift[mask.nonzero().flatten()[:,None],ids]=x[replacement[mask]][:,ids]
            assert torch.equal(shift[:,domain.numeric],x[:,domain.numeric])
            assert torch.allclose(domain.project(shift,True),shift,atol=1e-6)
            m=copy.deepcopy(model)
            for _ in range(cfg['update']['steps']):
                loss=torch.nn.functional.binary_cross_entropy_with_logits(m(shift),y);grads=torch.autograd.grad(loss,tuple(m.parameters()))
                with torch.no_grad():
                    for parameter,grad in zip(m.parameters(),grads):parameter.copy_(parameter-cfg['update']['lr']*grad)
            states.append(m.state_dict());protocol.append(dict(seed=seed,probability=probability,realization=k,changed_fraction=float((shift!=x).any(1).float().mean())))
            for objpath in sorted(worker.glob('test_*.pt')):
                obj=load(objpath);method=objpath.stem.removeprefix('test_')
                for form in (['relaxed','hard'] if method.startswith('grace') else ['native']):
                    z=domain.project(obj['cf'],True) if form=='hard' else obj['cf']
                    with torch.no_grad():pred=m(z).sigmoid()>=.5
                    rows.append(dict(worker=name,method=method,realization=form,shift_probability=probability,shift_seed=seed,FV=float(pred.float().mean()),status=obj['status']))
        torch.save(dict(states=states,protocol=protocol),a.output/f'{name}.pt');checks.append(dict(worker=name,hard_category_legality=True,numerical_features_unchanged=True,models=15))
    frame=pd.DataFrame(rows);out=ROOT/'results/exp3_table1_single_run';frame.to_csv(out/'hard_category_shift_raw.csv',index=False)
    frame.groupby(['worker','method','realization','shift_probability','status'],dropna=False).FV.mean().reset_index().to_csv(out/'hard_category_shift_summary.csv',index=False)
    (out/'hard_category_shift_checks.json').write_text(json.dumps(dict(raw=str(a.output),checks=checks),indent=2))
if __name__=='__main__':main()
