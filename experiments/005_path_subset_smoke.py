"""Construction-only subset smoke; subset-derived budget, not a primary result."""
from pathlib import Path
from types import SimpleNamespace
import json
import time
import torch
import run
run.bootstrap()
from lift_ce.experiments.path_level_geodesic_wro_v1 import inputs
from lift_ce.counterfactuals.path_level_solver import solve_path, straight_derived_initial
from lift_ce.counterfactuals.geodesic_excess import exact_w2

PARENT = Path('/home/xu.zhao/robust_ce_results/iclr_runs/path_level_geodesic_wro_v1/20260916_path_v1')

def main():
    args=SimpleNamespace(output=PARENT,dataset='heloc',device='cpu')
    out,config,train,current,bank=inputs(args,small=True)
    source=torch.load(out/'train_population.pt',weights_only=True)['path']['base'][:8]
    target=train['target'][:16]
    D=exact_w2(source,target).distance
    controls=straight_derived_initial(source,target)
    started=time.perf_counter()
    result=solve_path(bank,current,source,target,controls,1.1*D,D,train['x'].amin(0),train['x'].amax(0),config['lambdas'],{**config['optimizer'],'steps':2})
    if len(result['trace'])!=3:raise ValueError('subset optimizer did not execute')
    report={'purpose':'subset execution check only, not full-instance feasibility or performance',
            'source_count':8,'target_count':16,'model_count':8,'D_subset':D,'B_subset':1.1*D,
            'budget_scope':'subset anchors only; primary absolute B_geo is unchanged',
            'status':result['status'],'seconds':time.perf_counter()-started,'trace':result['trace'],
            'excluded_from_primary_candidate_pool':True}
    file=Path(__file__).resolve().parents[1]/'results/005_path_subset_smoke.json'
    file.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trace'},indent=2))

if __name__=='__main__':main()
