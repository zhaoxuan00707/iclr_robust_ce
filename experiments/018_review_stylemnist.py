"""Independent completion review; never rewrites experiment/model outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.models.vision import SmallCNN,StrongConvAutoencoder
from lift_ce.counterfactuals.geodesic_excess import exact_w2


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();out=a.output
    torch.set_num_threads(1)
    config=json.loads((out/'resolved_config.json').read_text());manifest=json.loads((out/'experiment_manifest.json').read_text());frozen=json.loads((out/'frozen.json').read_text())
    assert digest(out/'resolved_config.json')==manifest['config_sha256']
    assert digest(config['checkpoint'])==manifest['checkpoint_sha256']
    for key,filename in [('frozen_images_sha256','frozen_images.pt'),('selected_latents_sha256','selected_latents.pt'),('attack_bank_manifest_sha256','attack_bank_manifest.csv')]:assert frozen[key]==digest(out/filename)
    payload=torch.load(config['checkpoint'],weights_only=True,map_location='cpu')
    ae=StrongConvAutoencoder(64).to(a.device).eval();ae.load_state_dict(payload['ae'])
    pop=torch.load(out/'population.pt',weights_only=True,map_location='cpu')
    methods=torch.load(out/'frozen_images.pt',weights_only=True,map_location='cpu')
    selected=torch.load(out/'selected_latents.pt',weights_only=True,map_location='cpu')
    bank=pd.read_csv(out/'attack_bank_manifest.csv');assert len(bank)==24 and bank.feasible.all()
    models=[];worst_upper=0.
    for r in bank.to_dict('records'):
        assert digest(r['checkpoint'])==r['checkpoint_sha256'] and digest(r['perturbation_path'])==r['perturbation_sha256']
        delta=torch.load(r['perturbation_path'],weights_only=True,map_location='cpu')
        upper=float(delta.double().flatten(1).square().sum(1).mean().sqrt())
        worst_upper=max(worst_upper,upper)
        assert upper<=config['data_radius_pixel_l2']+1e-6
        assert (pop['x']+delta).min()>=-1e-7 and (pop['x']+delta).max()<=1+1e-7
        assert r['W_data']<=upper+1e-5
        m=SmallCNN().to(a.device).eval();m.load_state_dict(torch.load(r['checkpoint'],weights_only=True,map_location=a.device));models.append(m)
    table=pd.read_csv(out/'method_cost_risk.csv').set_index('method');max_risk_error=0.
    decoder_errors=[]; reconstructed={}
    with torch.no_grad():
        for name,images in methods.items():
            risk=max(float(torch.relu(.5-torch.sigmoid(m(images.to(a.device)))).double().mean()) for m in models)
            max_risk_error=max(max_risk_error,abs(risk-table.loc[name,'Phi_hat_r']))
            assert abs(risk-table.loc[name,'Phi_hat_r'])<1e-7
            assert len(images)==64 and torch.isfinite(images).all()
            cost=float((images-pop['factual']).flatten(1).norm(dim=1).mean())
            assert abs(cost-table.loc[name,'pixel_L2_cost'])<1e-6
        D=exact_w2(pop['base_z'],pop['target_z']).distance
        for name,q in selected.items():
            beta=float(name.split('beta')[1]);length=exact_w2(pop['base_z'],q).distance+exact_w2(q,pop['target_z']).distance
            assert length<=beta*D+1e-6*max(1,D)
            decoded=ae.decode(q.to(a.device)).cpu(); repeated=ae.decode(q.to(a.device)).cpu()
            error=(decoded-methods[name]).abs()
            # ConvTranspose CUDA kernels were not run in deterministic mode.
            # Measure observed repeatability and require tiny absolute/mean errors.
            assert float(error.max())<1e-4 and float(error.mean())<1e-6
            decoder_errors.append({'method':name,'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),'repeat_max_abs':float((decoded-repeated).abs().max())})
            reconstructed[name]=decoded
    changed_decisions=0
    with torch.no_grad():
        for ckpt in sorted((out/'checkpoints').glob('future_*.pt')):
            m=SmallCNN().to(a.device).eval();m.load_state_dict(torch.load(ckpt,weights_only=True,map_location=a.device))
            for name,image in reconstructed.items():
                changed_decisions+=int(((m(image.to(a.device))>=0)!=(m(methods[name].to(a.device))>=0)).sum())
    assert changed_decisions==0
    # Recompute every saved future-validity value from original frozen images.
    max_future_error=0.
    with torch.no_grad():
        for file in sorted((out/'jobs').glob('future_*/metrics.csv')):
            rows=pd.read_csv(file)
            m=SmallCNN().to(a.device).eval();m.load_state_dict(torch.load(rows.iloc[0].checkpoint,weights_only=True,map_location=a.device))
            for row in rows.itertuples():
                value=float((m(methods[row.method].to(a.device))>=0).double().mean())
                max_future_error=max(max_future_error,abs(value-row.future_validity))
    assert max_future_error<1e-12
    fv=pd.read_csv(out/'future_validity.csv');assert len(fv)==45*len(methods)
    assert fv.groupby(['method','family','severity','realization']).size().eq(1).all()
    assert ((fv.future_validity>=0)&(fv.future_validity<=1)).all()
    assert len(np.unique(np.load(out/'factual_ids.npy')))==64
    assert len(np.unique(np.load(out/'target_ids.npy')))==64
    jobs=list((out/'jobs').glob('*/complete.json'));assert len(jobs)==68
    for r in jobs:
        d=json.loads(r.read_text());assert d['config_sha256']==manifest['config_sha256']
    before=min(json.loads(p.read_text())['start_utc'] for p in (out/'jobs').glob('future_*/complete.json'))
    assert frozen['time_utc']<before
    for name in ['future_validity_severity.pdf','qualitative_grid.pdf']:assert (out/'figures'/name).read_bytes().startswith(b'%PDF')
    review={'status':'passed','complete_jobs':len(jobs),'bank_models':len(bank),'future_models':45,'methods':len(methods),
            'max_bank_risk_recompute_error':max_risk_error,'max_future_validity_recompute_error':max_future_error,'max_paired_data_distance':worst_upper,
            'future_evaluation_started_after_freeze':True,'all_geometry_checks_pass':True,
            'run':str(out),'current_validity_repair':False,'decoder_numerical_errors':decoder_errors,'future_decisions_changed_on_redecode':changed_decisions}
    result=ROOT/'results/009_stylemnist_grace_v1/completion_review.json';result.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review,indent=2))

if __name__=='__main__':main()
