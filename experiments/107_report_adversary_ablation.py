"""Independent saved-output/pairing checks and per-population ablation report."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments/src')]
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.geodesic_excess import exact_w2

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    jobs=json.loads((a.run/'jobs.json').read_text());assert all(j['exitcode']==0 for j in jobs)
    torch.set_num_threads(1);metrics=[];future=[];checks=[];pairs={};hard=[]
    for job in jobs:
        folder=a.run/job['name'];assert (folder/'complete.json').exists()
        m=json.loads((folder/'metrics.json').read_text());manifest=json.loads((folder/'input_manifest.json').read_text());parent=Path(manifest['parent'])
        for f,h in manifest['parent_sha256'].items():assert sha(parent/f)==h,(folder,f)
        prepared=load(parent/'prepared.pt');inputs=load(folder/'inputs.pt');result=load(folder/'result.pt');d=prepared['data']
        model=TabularMLP(d['train_x'].shape[1],prepared['hidden']);model.load_state_dict(load(parent/'current.pt'));model.eval()
        states=load(parent/'future_test.pt');z=result['cf'];n=len(z)
        assert n==m['n'] and torch.equal(result['q'][result['assignment']],z)
        assert len(result['assignment'].unique())==n
        pop=load(parent/'test_wachter.pt')['cf'];assert torch.equal(pop[inputs['indices']],inputs['base'])
        assert torch.equal(prepared['sources']['test'][inputs['indices']],inputs['source'])
        with torch.no_grad():cv=float((model(z).sigmoid()>=.5).float().mean())
        cost=float((z-inputs['source']).norm(dim=1).mean());predictions=[];basepred=[]
        for params,buffers in states['states']:
            model.load_state_dict({**params,**buffers})
            with torch.no_grad():predictions.append(model(z).sigmoid()>=.5);basepred.append(model(inputs['base']).sigmoid()>=.5)
        pred=torch.stack(predictions);fv=float(pred.float().mean())
        for k,v in [('FV',fv),('proximity',cost),('current_validity',cv)]:assert abs(m[k]-v)<2e-6,(folder,k)
        vio=exact_w2(inputs['base'],result['q']).distance+exact_w2(result['q'],prepared['p1']).distance-result['budget']
        cfg=json.loads((folder/'config.json').read_text());tolerance=cfg['solver']['feas_tol']*max(1.,result['D'])
        assert abs(vio-m['geometry_violation'])<1e-6 and (vio<=tolerance)==m['geometry_feasible']
        assert abs(m['objective']-m['nominal_loss']-m['robustness_penalty'])<1e-6
        pen=m['population_sensitivity'] if m['variant']=='population' else m['individual_sensitivity']
        assert abs(pen-m['sensitivity'])<1e-5*max(1.,pen)
        assert m['population_sensitivity']<=m['individual_sensitivity']+1e-6
        if not m['zero_denominator']:assert abs(m['kappa']-m['population_sensitivity']/m['individual_sensitivity'])<1e-8
        else:assert m['kappa'] is None
        per=pd.read_csv(folder/'per_future_model.csv');actual=per[per.variant==m['variant']].sort_values('model_index')
        assert len(actual)==45 and np.allclose(actual.FV,pred.float().mean(1).numpy(),atol=2e-6)
        for key in ['dataset','seed','n','r_ratio','beta']:per[key]=m[key]
        per['optimized_variant']=m['variant'];future.append(per)
        m['physical_gpu']=job['gpu'];metrics.append(m)
        key=(m['dataset'],m['seed'],m['n'],m['r_ratio'],m['beta']);pairs.setdefault(key,{})[m['variant']]=(manifest,result,folder)
        if (folder/'hard_diagnostic.json').exists():hard.append({**{k:m[k] for k in ['dataset','seed','n','r_ratio','beta','variant']},**json.loads((folder/'hard_diagnostic.json').read_text())})
        checks.append(dict(job=job['name'],source_hashes=True,base_subset=True,assignment=True,metrics=True,per_future=True,geometry_recomputed=True,geometry_feasible=m['geometry_feasible']))
    for key,pair in pairs.items():
        assert set(pair)=={'population','individual'}
        p0,r0,f0=pair['population'];p1,r1,f1=pair['individual'];assert p0==p1
        assert r0['budget']==r1['budget'] and r0['D']==r1['D']
        d0=json.loads((f0/'initial_diagnostics.json').read_text());d1=json.loads((f1/'initial_diagnostics.json').read_text());assert d0==d1
        if key[3]==0:assert torch.equal(r0['q'],r1['q']),('zero radius mismatch',key)
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(metrics);frame.to_csv(out/'per_seed_metrics.csv',index=False)
    pd.concat(future,ignore_index=True).to_csv(out/'per_future_model.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'independent_checks.csv',index=False)
    if hard:pd.DataFrame(hard).to_csv(out/'hard_category_diagnostics.csv',index=False)
    groups=['dataset','n','r_ratio','beta','variant'];values=['FV','proximity','current_validity','geometry_violation','nominal_loss','robustness_penalty','optimization_seconds','peak_gpu_bytes','kappa']
    agg=frame.groupby(groups)[values].agg(['mean','std','count']);agg.columns=['_'.join(c) for c in agg.columns];agg.reset_index().to_csv(out/'aggregated.csv',index=False)
    display=[]
    for key,g in frame.groupby(groups,sort=False):
        row=dict(zip(groups,key));row['seeds']=len(g)
        for col,mult,digits in [('FV',100,2),('proximity',1,4),('current_validity',100,2),('optimization_seconds',1,2),('kappa',1,4)]:
            vals=g[col]*mult;row[col]=f'{vals.mean():.{digits}f}' + (f' +/- {vals.std(ddof=1):.{digits}f}' if len(vals)>1 else '')
        row['feasible']=f"{int(g.geometry_feasible.sum())}/{len(g)}"
        display.append(row)
    paired=[]
    for key,g in frame.groupby(['dataset','n','seed','r_ratio','beta']):
        rows=g.set_index('variant');paired.append(dict(zip(['dataset','n','seed','r_ratio','beta'],key),FV_ind_minus_pop=float(rows.loc['individual','FV']-rows.loc['population','FV']),proximity_ind_minus_pop=float(rows.loc['individual','proximity']-rows.loc['population','proximity'])))
    pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
    text='# Controlled population-versus-individual adversary ablation\n\n'
    text+=f'Raw run: `{a.run}`. {len(metrics)} saved outputs and {len(pairs)} paired comparisons independently checked. No parent output changed.\n\n'
    text+='Mean +/- sample standard deviation across seeds3/4/5 (where three runs exist). Seeds resample factual/Base populations from the fixed parent seed0 classifier cohort; they are not independent classifier-training seeds. Populations may overlap substantially, especially German (32 drawn from37). All modes/seeds reuse the exact same45 parent future checkpoints for their dataset; repeated shifts are not independent statistical units.\n\n'
    text+=pd.DataFrame(display).to_markdown(index=False)+'\n\n'
    text+='FV/current validity are percentages; proximity is original standardized-feature L2. Times are synchronized optimization wall seconds including final objective diagnostics, excluding input loading, evaluation and kappa diagnostics. Peak allocated GPU memory is in per_seed_metrics.csv. Pair modes run sequentially on the same GPU; other dataset/seed jobs can overlap on other GPUs/shared host.\n\n'
    text+='Only sensitivity aggregation changed. Both methods share Base, reference, data-side perturbation map, complete10-step SGD(.01) update, smooth temperature.05, optimizer, stopping, exact geometry and assignment. Individual sensitivities use sqrt(N) times the flattened training-perturbation gradient norm; kappa numerator and denominator use identical vectors at the final Q. Zero denominator is flagged and kappa is null.\n\n'
    text+='HELOC original calibration selected r=0, so its paired outputs must be identical. The r/b_ref3 HELOC setting was declared before evaluation as an additional positive-radius diagnostic with unchanged beta1.1. German uses the original r/b_ref3,beta1.1. These settings were not retuned for32 supports.\n\n'
    text+='German optimization is the original A.5 relaxed categorical geometry; hard realization is a separate diagnostic. Infeasible final outputs remain labeled and must not be treated as successful constrained solutions. No validity repair or convergence certificate. No new morphology/actionability claims.\n\n'
    text+='All required raw metrics are in per_seed_metrics.csv, per-scenario scores in per_future_model.csv, and signed individual-minus-population differences in paired_differences.csv. No64-support German experiment: the frozen parent cohort has only37 factuals; duplication would change the population.\n'
    (out/'report.md').write_text(text)
    print('VERIFIED',len(metrics),'outputs;',len(pairs),'pairs',flush=True)
if __name__=='__main__':main()
