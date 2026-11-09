"""Controlled aggregation ablation, reusing immutable Task100 inputs and banks."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..models.tabular import TabularMLP
from ..counterfactuals.mixed_update_aware import MixedDomain, sensitivity_aggregation_diagnostics
from .appendix_tabular import grace, load, score, sync, save, dump as dump_object

REPO=Path(__file__).resolve().parents[4]
CONFIG=REPO/'experiments/configs/population_individual_adversary.json'

def dump(path, obj):
    dump_object(obj, path)


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def file_hash(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=['diabetes','heloc','german'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--r-ratio',type=float,required=True)
    p.add_argument('--beta',type=float,required=True)
    p.add_argument('--adversary-mode',choices=['population','individual'],default='population')
    p.add_argument('--sensitivity-chunk-size',type=int,default=1)
    p.add_argument('--profile-steps',type=int,help='Debug-only shorter solve, separately labeled')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads(CONFIG.read_text());parent=Path(cfg['parent_run'])/(a.dataset+'_mlp')
    meta=json.loads((parent/'metadata.json').read_text());settings=meta['config']
    if a.profile_steps is not None:settings['solver']['steps']=a.profile_steps
    data=load(parent/'prepared.pt');d=data['data'];original_base=load(parent/'test_wachter.pt')['cf']
    if a.n>len(original_base):raise ValueError(f'Only {len(original_base)} parent factuals; cannot sample {a.n} without replacement')
    ids=torch.randperm(len(original_base),generator=torch.Generator().manual_seed(a.seed))[:a.n]
    base=original_base[ids].clone();source=data['sources']['test'][ids].clone();p1=data['p1']
    torch.save(dict(indices=ids,base=base,source=source,source_ids=[data['source_ids']['test'][i] for i in ids.tolist()]),a.output/'inputs.pt')
    files=['prepared.pt','current.pt','test_wachter.pt','future_test.pt','selection.json']
    manifest=dict(parent=str(parent),parent_sha256={f:file_hash(parent/f) for f in files},
        indices=ids.tolist(),base_sha256=tensor_hash(base),source_sha256=tensor_hash(source),
        reference_sha256=tensor_hash(p1),training_sha256=tensor_hash(d['train_x']),
        training_labels_sha256=tensor_hash(d['train_y']),seed_unit=cfg['seed_unit'])
    dump(a.output/'input_manifest.json',manifest);dump(a.output/'config.json',settings)
    device=torch.device(a.device)
    model=TabularMLP(d['train_x'].shape[1],data['hidden']);model.load_state_dict(load(parent/'current.pt'));model.eval();model=model.to(device)
    domain=MixedDomain(d['train_x'],data['groups']);x=d['train_x'].to(device);y=d['train_y'].to(device)
    diagkwargs=dict(**settings['update'],temperature=settings['smooth_temperature'],cat_temperature=settings['categorical_temperature'],smoothing=settings['categorical_smoothing'],chunk_size=a.sensitivity_chunk_size)
    initial=sensitivity_aggregation_diagnostics(model,x,y,base.to(device),domain,**diagkwargs)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    sync();start=time.perf_counter()
    result=grace(model,x,y,base,p1,domain,a.r_ratio*data['bref'],a.beta,settings,a.adversary_mode,a.sensitivity_chunk_size)
    sync();elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
    save(result,a.output/'result.pt')
    final=sensitivity_aggregation_diagnostics(model,x,y,result['q'].to(device),domain,**diagkwargs)
    # Evaluation only after outputs have been saved. Reuse exact frozen test bank.
    bankobj=load(parent/'future_test.pt');bank=[({k:v.to(device) for k,v in params.items()},{k:v.to(device) for k,v in buffers.items()}) for params,buffers in bankobj['states']]
    assert len(bank)==45
    evaluation=score(model,result['cf'],bank,source);base_eval=score(model,base,bank,source)
    save(dict(refined=evaluation,base=base_eval),a.output/'evaluation.pt')
    row=dict(dataset=a.dataset,seed=a.seed,n=a.n,variant=a.adversary_mode,r_ratio=a.r_ratio,
        r=a.r_ratio*data['bref'],beta=a.beta,FV=evaluation['FV'],proximity=evaluation['Cost'],
        current_validity=evaluation['current_validity'],geometry_violation=result['W_base_Q']+result['W_Q_P1']-result['budget'],
        geometry_feasible=result['feasible'],nominal_loss=result['nominal'],sensitivity=result['sensitivity'],
        robustness_penalty=a.r_ratio*data['bref']*result['sensitivity'],objective=result['surrogate'],
        optimization_seconds=elapsed,core_generation_seconds=result['seconds'],peak_gpu_bytes=peak,
        iterations=result['iterations'],base_FV=base_eval['FV'],base_proximity=base_eval['Cost'],
        base_current_validity=base_eval['current_validity'],profile_only=a.profile_steps is not None,
        device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        **final)
    assert final['population_sensitivity']<=final['individual_sensitivity']+1e-6*max(1.,final['individual_sensitivity'])
    assert final['mean_vector_max_error']<1e-5
    dump(a.output/'initial_diagnostics.json',initial);dump(a.output/'metrics.json',row)
    pd.DataFrame(result['history']).to_csv(a.output/'trace.csv',index=False)
    rows=[]
    for i,record in enumerate(bankobj['records']):
        for method,metrics in [('Base',base_eval),(a.adversary_mode,evaluation)]:
            rows.append(dict(**record,model_index=i,variant=method,FV=float(metrics['predictions'][i].float().mean())))
    pd.DataFrame(rows).to_csv(a.output/'per_future_model.csv',index=False)
    if domain.groups:
        hard=domain.project(result['cf'],True);hardmetrics=score(model,hard,bank,source)
        from ..counterfactuals.geodesic_excess import exact_w2
        violation=exact_w2(base,hard).distance+exact_w2(hard,p1).distance-result['budget']
        dump(a.output/'hard_diagnostic.json',dict(FV=hardmetrics['FV'],current_validity=hardmetrics['current_validity'],proximity=hardmetrics['Cost'],geometry_violation=violation,geometry_feasible=violation<=settings['solver']['feas_tol']*max(1.,result['D'])))
    dump(a.output/'complete.json',dict(complete=True,time=time.time()))
    print(json.dumps(row),flush=True)

if __name__=='__main__':main()
