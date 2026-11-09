"""Read-only endpoint feasibility audit. No future model outcomes are accessed."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import run
run.bootstrap()
import torch
from lift_ce.counterfactuals.geodesic_excess import exact_w2
from lift_ce.counterfactuals.path_preflight import check_path_preconditions
from lift_ce.experiments.fixed_risk_geometry_v1 import context

REPO = Path(__file__).resolve().parents[1]
PARENT = Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1')


def main():
    out, meta, train, current = context(SimpleNamespace(output=PARENT, dataset='heloc'))
    config = json.loads((out/'resolved_config.json').read_text())
    lower, upper = train['x'].amin(0), train['x'].amax(0)
    target = train['target']
    rows = []
    hashes = {}
    for split in ('train', 'cal', 'test'):
        path = out/f'{split}_population.pt'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[path.name] != meta['files'][path.name]:
            raise ValueError(f'Frozen population changed: {path}')
        population = torch.load(path, weights_only=True)
        source = population['path']['base']
        torch.testing.assert_close(population['path']['target'], target, rtol=0, atol=0)
        plan = exact_w2(source, target)
        with torch.no_grad():
            s = torch.sigmoid(current(source)).reshape(-1)
            t = torch.sigmoid(current(target)).reshape(-1)
        invalid_s, invalid_t = float((s < .5).double().mean()), float((t < .5).double().mean())
        box_s = float(((source < lower-1e-5) | (source > upper+1e-5)).any(1).double().mean())
        box_t = float(((target < lower-1e-5) | (target > upper+1e-5)).any(1).double().mean())
        if split == 'train' and abs(plan.distance-config['D']) > 1e-6:
            raise ValueError('Training anchor distance differs from frozen config')
        for beta in config['beta_geo']:
            budget = beta*config['D']
            result = check_path_preconditions(plan.distance,budget,invalid_s,invalid_t,box_s,box_t,
                                              geometry_tolerance=1e-6*max(1,config['D']))
            rows.append({'split':split,'beta_geo':beta,'D_train':config['D'],
                         **result,'source_invalid_mass':invalid_s,'target_invalid_mass':invalid_t,
                         'source_outside_training_box_mass':box_s,'target_outside_training_box_mass':box_t,
                         'minimum_source_score':float(s.min()),'minimum_target_score':float(t.min()),
                         'ot_marginal_residual':plan.marginal_residual})
    report={'parent':str(PARENT),'population_hashes':hashes,'rows':rows,
            'domain_scope':'Box retained on optimizable controls; path range diagnostic only; f0>=.5 required at every deployment-grid point.',
            'heldout_outcomes_accessed':False,'raw_outputs_modified':False,
            'new_method_implemented':False}
    dest=REPO/'results/004_path_preflight.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    with dest.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
