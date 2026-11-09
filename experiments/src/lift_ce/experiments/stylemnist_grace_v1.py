"""MNIST 4->9 style-shift GRACE, frozen image assets and full-update bank."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.datasets import MNIST
from torchvision.transforms import functional as TF
from ..models.vision import SmallCNN, StrongConvAutoencoder
from ..config import BaseCEConfig
from ..counterfactuals.vision import generate_near_boundary_ce, generate_stability_baseline
from ..counterfactuals.geodesic_excess import exact_w2
from ..counterfactuals.minimal_validity_repair import solve_raw, geometry
from ..experiments.heloc_conditional_adversary_strengthened_v1 import differentiable_adamw_update, search_loss

REPO = Path(__file__).resolve().parents[4]
CONFIG = REPO/'experiments/configs/stylemnist_grace_v1.json'


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def assets(config, device):
    payload = load(Path(config['checkpoint']))
    model = SmallCNN().to(device).eval()
    ae = StrongConvAutoencoder(64).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload['model']); ae.load_state_dict(payload['ae'])
    return model, ae, payload


def full_update(base, x, y, config, device):
    """Same eight-epoch AdamW Update for every accepted/future distribution."""
    u = config['update']; torch.manual_seed(u['seed']); torch.cuda.manual_seed_all(u['seed'])
    model = copy.deepcopy(base).to(device).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=u['lr'], weight_decay=u['weight_decay'])
    for epoch in range(u['epochs']):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(u['seed']+epoch))
        for start in range(0, len(x), u['batch_size']):
            idx = order[start:start+u['batch_size']].to(x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[idx].to(device)), y[idx].to(device))
            loss.backward(); optimizer.step()
    return model.eval().requires_grad_(False)


def rms(delta):
    return delta.flatten(1).square().sum(1).mean().sqrt()


def project_delta(x, delta, radius):
    delta = (x+delta).clamp(0,1)-x
    return delta * torch.clamp(torch.as_tensor(radius,device=x.device)/rms(delta).clamp_min(1e-12),max=1)


def data_w2(x, shifted, labels, device):
    """Exact class-preserving empirical OT on flattened pixel coordinates."""
    total = 0.
    for label in (0,1):
        a=x[labels==label].flatten(1).double().to(device)
        b=shifted[labels==label].flatten(1).double().to(device)
        cost=torch.cdist(a,b).square().cpu().numpy()
        i,j=linear_sum_assignment(cost); total += float(cost[i,j].sum())
    return (total/len(x))**.5


class ImageBank:
    """Differentiable decoded-image classifier bank; models are fixed."""
    def __init__(self, models, ae):
        self.models=models; self.ae=ae; self.count=len(models)
    def logits(self, points, start=0, end=None):
        images=self.ae.decode(points)
        return torch.stack([m(images) for m in self.models[start:end]])


def risk_on_images(models, images):
    with torch.no_grad():
        values=torch.stack([torch.relu(.5-torch.sigmoid(m(images))).double().mean() for m in models])
    return float(values.max())


def bank_models(out, config, device, initial_only=False):
    records=[]
    for f in sorted((out/'jobs').glob('attack_*/candidates.json')):
        if initial_only and not f.parent.name.startswith('attack_initial_'):continue
        records.extend(json.loads(f.read_text()))
    if not records: raise ValueError('empty attack bank')
    models=[]
    for r in records:
        assert r['feasible'] and sha(Path(r['checkpoint']))==r['checkpoint_sha256']
        m=SmallCNN().to(device).eval().requires_grad_(False); m.load_state_dict(load(Path(r['checkpoint'])))
        models.append(m)
    return records,models


def prepare(args, config):
    out=args.output
    if (out/'prepared.json').exists():return
    model,ae,payload=assets(config,args.device)
    raw=MNIST(config['data_dir'],train=True,download=False)
    test=MNIST(config['data_dir'],train=False,download=False)
    ids=torch.nonzero((raw.targets==4)|(raw.targets==9)).flatten()
    ids=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(config['seed']))]
    train_ids=ids[:config['train_size']]; cal_ids=ids[config['train_size']:config['train_size']+config['cal_size']]
    x=raw.data[train_ids,None].float()/255; y=(raw.targets[train_ids]==9).float()
    cx=raw.data[cal_ids,None].float()/255; cy=(raw.targets[cal_ids]==9).float()
    # Recover original MNIST test IDs exactly; do not invent source metadata.
    lookup={hashlib.sha256(t.numpy().tobytes()).hexdigest():i for i,t in enumerate(test.data)}
    factual=payload['sources']; factual_ids=[]
    for image in factual:
        idx=lookup[hashlib.sha256((image[0]*255).round().byte().numpy().tobytes()).hexdigest()]
        torch.testing.assert_close(test.data[idx,None].float()/255,image,atol=0,rtol=0)
        assert test.targets[idx]==4
        factual_ids.append(idx)
    target_choices=torch.nonzero(y==1).flatten()
    target_choices=target_choices[torch.randperm(len(target_choices),generator=torch.Generator().manual_seed(21))[:config['target_count']]]
    target=x[target_choices]
    np.save(out/'factual_ids.npy',np.array(factual_ids));np.save(out/'target_ids.npy',train_ids[target_choices].numpy())
    np.save(out/'train_ids.npy',train_ids.numpy());np.save(out/'calibration_ids.npy',cal_ids.numpy())
    cfg=BaseCEConfig(steps=96,autoencoder_type='strong',latent_dim=64)
    t=time.perf_counter()
    base=generate_near_boundary_ce(model,ae,factual,target,cfg,torch.device(args.device)).counterfactuals
    base_seconds=time.perf_counter()-t
    with torch.no_grad():
        z=ae.encode(base.to(args.device)).cpu();target_z=ae.encode(target.to(args.device)).cpu()
        rec=ae(cx.to(args.device)); factual_z=ae.encode(factual.to(args.device)).cpu()
        scores=torch.sigmoid(model(cx.to(args.device)));rs=torch.sigmoid(model(rec))
        decoded=ae.decode(z.to(args.device))
        diagnostics={'calibration_n':len(cx),'calibration_accuracy':float(((scores>=.5)==cy.to(args.device).bool()).float().mean()),
          'calibration_reconstruction_mse':float((rec-cx.to(args.device)).square().mean()),
          'calibration_reconstruction_decision_agreement':float(((scores>=.5)==(rs>=.5)).float().mean()),
          'base_validity':float((torch.sigmoid(model(base.to(args.device)))>=.5).float().mean()),
          'base_encode_decode_validity':float((torch.sigmoid(model(decoded))>=.5).float().mean()),
          'base_encode_decode_pixel_l2':float((decoded-base.to(args.device)).flatten(1).norm(dim=1).mean()),
          'target_decoded_validity':float((torch.sigmoid(model(ae.decode(target_z.to(args.device))))>=.5).float().mean())}
    dump(out/'representation_checks.json',diagnostics)
    # This gate checks asset compatibility before GRACE, never repairs controls.
    if diagnostics['calibration_accuracy']<.90 or diagnostics['calibration_reconstruction_decision_agreement']<.95:
        raise ValueError(f'asset compatibility gate failed: {diagnostics}')
    np.save(out/'base_cf.npy',base.numpy());np.save(out/'base_cf_latents.npy',z.numpy())
    torch.save({'x':x,'y':y,'factual':factual,'factual_z':factual_z,'target':target,'target_z':target_z,'base':base,'base_z':z},out/'population.pt')
    baseline={'Base':base}; times={'Base':base_seconds}
    for name in config['baselines']:
        t=time.perf_counter()
        if name=='highconf': result=generate_near_boundary_ce(model,ae,factual,target,cfg,torch.device(args.device),target_probability=.9)
        else: result=generate_stability_baseline(model,ae,factual,target,cfg,method=name,device=torch.device(args.device))
        baseline[name]=result.counterfactuals;times[name]=time.perf_counter()-t
    torch.save(baseline,out/'baselines.pt');dump(out/'baseline_runtime.json',times)
    D=exact_w2(z,target_z).distance
    dump(out/'prepared.json',{'D':D,'population_sha256':sha(out/'population.pt'),'base_sha256':sha(out/'base_cf.npy'),
                             'factual_sha256':sha(out/'factual_ids.npy'),'target_sha256':sha(out/'target_ids.npy')})
    print('PREPARED',diagnostics,'D=',D,flush=True)


def attack(args, config):
    out=args.output; job=out/'jobs'/args.job
    if (job/'complete.json').exists():return
    job.mkdir(parents=True,exist_ok=True)
    pop=load(out/'population.pt');base,ae,_=assets(config,args.device)
    source='Base'; beta=None
    if args.job.startswith('attack_initial_'): images=pop['base']; restart=int(args.job.rsplit('_',1)[1])
    else:
        _,_,b,r=args.job.split('_');beta=float(b)/100;restart=int(r);source=f'beta{int(b):03d}'
        q=load(out/'jobs'/f'opt_initial_{b}'/'selected.pt')['points'].to(args.device)
        with torch.no_grad():images=ae.decode(q).cpu()
    settings=config['attack']; seed=20260916+restart+(0 if beta is None else int(beta*100)*100)
    torch.manual_seed(seed)
    x=pop['x'].to(args.device);y=pop['y'].to(args.device);images=images.to(args.device)
    radius=config['data_radius_pixel_l2'];delta=torch.randn_like(x)
    delta=project_delta(x,.05*radius*delta/rms(delta),radius).requires_grad_(True)
    optimizer=torch.optim.Adam([delta],lr=settings['lr']);trace=[]
    # Truncated surrogate retains historical differentiable AdamW logic.
    # Each iteration uses a prespecified minibatch; full bank updates use all data.
    base.eval().requires_grad_(True)
    for step in range(settings['steps']):
        idx=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+step))[:settings['batch_size']].to(args.device)
        pred=differentiable_adamw_update(base,x[idx],y[idx],delta[idx],images,settings['unroll_K'])
        objective=search_loss(pred,'softplus',10.)
        grad=torch.autograd.grad(objective,delta)[0]
        if not torch.isfinite(grad).all():raise ValueError('nonfinite attack gradient')
        optimizer.zero_grad();delta.grad=-grad;optimizer.step()
        with torch.no_grad():delta.copy_(project_delta(x,delta,radius))
        trace.append({'step':step,'surrogate_loss':float(objective.detach()),'paired_distance':float(rms(delta.detach())), 'gradient_norm':float(grad.norm())})
        if step%10==0: print(args.job,'search',step,trace[-1],flush=True)
    pd.DataFrame(trace).to_csv(job/'search_trace.csv',index=False)
    records=[]
    for fraction in settings['radial_fractions']:
        label=f'radial{int(fraction*100):03d}'; delta_path=job/f'{label}_delta.pt'
        d=project_delta(x,delta.detach()*fraction,radius).cpu(); shifted=pop['x']+d
        torch.save(d,delta_path)
        upper=float(rms(d));distance=data_w2(pop['x'],shifted,pop['y'],args.device)
        feasible=distance<=radius+1e-6 and upper<=radius+1e-6
        if not feasible:raise ValueError('infeasible data perturbation')
        m=full_update(base,shifted,pop['y'],config,args.device)
        path=out/'checkpoints'/f'{args.job}_{label}.pt';torch.save(m.cpu().state_dict(),path)
        records.append({'candidate':f'{args.job}_{label}','source':source,'source_beta':beta,'restart':restart,
          'seed':seed,'fraction':fraction,'perturbation_path':str(delta_path),'perturbation_sha256':sha(delta_path),
          'shifted_training_id':sha(out/'train_ids.npy')+':'+sha(delta_path),'W_data':distance,
          'paired_W_data_upper_bound':upper,'radius':radius,'feasible':feasible,'checkpoint':str(path),
          'checkpoint_sha256':sha(path),'full_update_seed':config['update']['seed'],'full_update_epochs':config['update']['epochs']})
        print(args.job,label,'full update complete',distance,flush=True)
    dump(job/'candidates.json',records)


def optimize(args,config):
    out=args.output;job=out/'jobs'/args.job
    if (job/'complete.json').exists():return
    job.mkdir(parents=True,exist_ok=True)
    _,phase,b=args.job.split('_');beta=int(b)/100
    records,models=bank_models(out,config,args.device,initial_only=phase=='initial')
    _,ae,_=assets(config,args.device);bank=ImageBank(models,ae)
    pop=load(out/'population.pt');z=pop['base_z'].to(args.device);target=pop['target_z'].to(args.device)
    plan=exact_w2(z,target);perm=torch.empty(len(z),dtype=torch.long);perm[plan.source_index]=plan.target_index
    target=target[perm.to(target.device)];D=plan.distance
    starts={'base':z,'midpoint':(z+target)/2,'target':target}
    candidates=[];trace=[]
    for name,initial in starts.items():
        def progress(row):
            if row['step']%20==0:print(args.job,name,'step',row['step'],'risk',row['true_bank_hinge'],'slack',row['slack'],flush=True)
        retained,history=solve_raw(bank,z,target,initial,D,beta*D,float('-inf'),float('inf'),config['optimizer'],on_step=progress)
        candidates.extend([{**r,'initialization':name,'origin_job':args.job} for r in retained])
        trace.extend([{**r,'initialization':name} for r in history])
    best=min(candidates,key=lambda r:r['true_bank_hinge'])
    torch.save(candidates,job/'candidates.pt');torch.save(best,job/'selected.pt')
    pd.DataFrame(trace).to_csv(job/'trace.csv',index=False)
    dump(job/'bank_used.json',{'candidate_ids':[r['candidate'] for r in records]})


def freeze(args,config):
    out=args.output
    if (out/'frozen.json').exists():return
    records,models=bank_models(out,config,args.device);_,ae,_=assets(config,args.device)
    pd.DataFrame(records).to_csv(out/'attack_bank_manifest.csv',index=False)
    pop=load(out/'population.pt');z=pop['base_z'];target=pop['target_z'];D=exact_w2(z,target).distance
    candidates=[]
    for f in sorted((out/'jobs').glob('opt_*/candidates.pt')):candidates.extend(load(f))
    rows=[]
    for i,r in enumerate(candidates):
        q=r['points']; p0=exact_w2(z,q);p1=exact_w2(q,target)
        with torch.no_grad():image=ae.decode(q.to(args.device))
        risk=risk_on_images(models,image)
        cost=float((image.cpu()-pop['factual']).flatten(1).norm(dim=1).mean())
        rows.append({'candidate_index':i,'origin_job':r['origin_job'],'initialization':r['initialization'],
                     'length':p0.distance+p1.distance,'risk':risk,'pixel_cost':cost})
    pd.DataFrame(rows).to_csv(out/'candidate_cross_evaluation.csv',index=False)
    selected={};geometry_rows=[]
    for beta in config['beta_geo']:
        eligible=[r for r in rows if r['length']<=beta*D+1e-6*max(1,D)]
        best=min(eligible,key=lambda r:(r['risk'],r['pixel_cost'],r['candidate_index']))
        q=candidates[best['candidate_index']]['points'];name=f'GRACE_beta{beta:.2f}'
        selected[name]=q
        geometry_rows.append({'method':name,'beta_geo':beta,'D':D,'B_geo':beta*D,
          'total_length':best['length'],'slack':beta*D-best['length'],'feasible':True,
          'Phi_hat_r':best['risk'],'source_candidate':best['candidate_index']})
    torch.save(selected,out/'selected_latents.pt')
    methods=load(out/'baselines.pt')
    with torch.no_grad():
        for name,q in selected.items():methods[name]=ae.decode(q.to(args.device)).cpu()
    torch.save(methods,out/'frozen_images.pt')
    pd.DataFrame(geometry_rows).to_csv(out/'geometry_feasibility.csv',index=False)
    risks=[]
    with torch.no_grad():
        for name,image in methods.items():
            latent=ae.encode(image.to(args.device)).cpu()
            risks.append({'method':name,'Phi_hat_r':risk_on_images(models,image.to(args.device)),
              'pixel_L2_cost':float((image-pop['factual']).flatten(1).norm(dim=1).mean()),
              'realized_latent_L2_cost':float((latent-pop['factual_z']).norm(dim=1).mean())})
    pd.DataFrame(risks).to_csv(out/'method_cost_risk.csv',index=False)
    dump(out/'frozen.json',{'frozen_images_sha256':sha(out/'frozen_images.pt'),
      'selected_latents_sha256':sha(out/'selected_latents.pt'),'attack_bank_manifest_sha256':sha(out/'attack_bank_manifest.csv'),
      'future_outcomes_seen':False,'candidate_count':len(candidates),'bank_size':len(records),'time_utc':datetime.now(timezone.utc).isoformat()})


def shifted_images(x,family,strength,seed):
    """Historical pilot operators, with a separately declared severity multiplier."""
    g=torch.Generator().manual_seed(seed)
    if family=='rotation':
        signs=torch.where(torch.rand(len(x),generator=g)<.5,-1.,1.)
        return torch.stack([TF.rotate(image,float(signs[i]*strength)) for i,image in enumerate(x)])
    if family=='translation':
        return torch.stack([TF.affine(image,0.,[int(round(strength)),-int(round(strength))],1.,0.) for image in x])
    if family=='noise':return (x+strength*torch.randn(x.shape,generator=g)).clamp(0,1)
    raise ValueError(family)


def future(args,config):
    out=args.output;job=out/'jobs'/args.job
    if (job/'complete.json').exists():return
    job.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((out/'frozen.json').read_text());assert frozen['frozen_images_sha256']==sha(out/'frozen_images.pt')
    _,family,si,ri=args.job.split('_');si=int(si);ri=int(ri)
    severity=config['future']['severity_multipliers'][si]
    strength=config['future']['nominal_strengths'][config['future']['families'].index(family)]*severity
    seed=800000+ri
    pop=load(out/'population.pt');base,_,_=assets(config,args.device)
    shifted=shifted_images(pop['x'],family,strength,seed)
    model=full_update(base,shifted,pop['y'],config,args.device)
    ckpt=out/'checkpoints'/f'{args.job}.pt';torch.save(model.state_dict(),ckpt)
    rows=[]
    with torch.no_grad():
        for name,images in load(out/'frozen_images.pt').items():
            scores=torch.sigmoid(model(images.to(args.device)))
            rows.append({'method':name,'family':family,'severity':severity,'strength':strength,'realization':ri,
              'shift_seed':seed,'update_seed':config['update']['seed'],'future_validity':float((scores>=.5).double().mean()),
              'mean_hinge':float(torch.relu(.5-scores).double().mean()),'checkpoint':str(ckpt),
              'evaluation':'fixed unshifted returned CF images; shifted training distribution only'})
    pd.DataFrame(rows).to_csv(job/'metrics.csv',index=False)
    dump(job/'shift.json',{'family':family,'severity':severity,'strength':strength,'shift_seed':seed,
      'train_ids_sha256':sha(out/'train_ids.npy'),'checkpoint_sha256':sha(ckpt),
      'paired_pixel_displacement':float(rms(shifted-pop['x']))})


def report(args,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output
    files=sorted((out/'jobs').glob('future_*/metrics.csv'))
    expected=len(config['future']['families'])*len(config['future']['severity_multipliers'])*config['future']['realizations']
    if len(files)!=expected:raise ValueError(f'future jobs {len(files)} != {expected}')
    fv=pd.concat([pd.read_csv(f) for f in files],ignore_index=True);fv.to_csv(out/'future_validity.csv',index=False)
    severity=fv.groupby(['method','family','severity'],as_index=False).agg(Future_Validity=('future_validity','mean'),std=('future_validity','std'))
    severity.to_csv(out/'shift_severity.csv',index=False)
    runtime=[]
    for p in sorted((out/'jobs').glob('*/complete.json')):
        r=json.loads(p.read_text());runtime.append({'job':p.parent.name,**r})
    rt=pd.DataFrame(runtime);rt.to_csv(out/'runtime.csv',index=False)
    cost=pd.read_csv(out/'method_cost_risk.csv');nominal=fv[fv.severity==1].groupby('method',as_index=False).agg(Future_Validity=('future_validity','mean'),worst_future_validity=('future_validity','min'))
    main=cost.merge(nominal,on='method');bt=json.loads((out/'baseline_runtime.json').read_text())
    opt_seconds=rt[rt.job.str.startswith('opt_')].seconds.sum();attack_seconds=rt[rt.job.str.startswith('attack_')].seconds.sum()
    main['runtime_seconds']=main.method.map(lambda m: bt.get(m,bt['Base']+opt_seconds/4))
    main['shared_attack_bank_seconds']=main.method.map(lambda m:attack_seconds if m.startswith('GRACE') else 0.)
    main.to_csv(out/'main_results.csv',index=False)
    geom=pd.read_csv(out/'geometry_feasibility.csv');ablation=geom.drop(columns=['Phi_hat_r']).merge(main,on='method').sort_values('beta_geo');ablation.to_csv(out/'geometry_ablation.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(12,3.4),sharey=True)
    for ax,family in zip(axs,config['future']['families']):
        for name in main.method:
            d=severity[(severity.family==family)&(severity.method==name)]
            ax.plot(d.severity,d.Future_Validity,marker='o',markersize=3,label=name.replace('GRACE_beta','GRACE β='))
        ax.set(title=family,xlabel='External severity multiplier',ylim=(-.02,1.02));ax.grid(alpha=.2)
    axs[0].set_ylabel('Future Validity');axs[-1].legend(fontsize=6,loc='best');fig.tight_layout();fig.savefig(out/'figures/future_validity_severity.pdf');plt.close(fig)
    pop=load(out/'population.pt');images=load(out/'frozen_images.pt');names=['Base']+[f'GRACE_beta{b:.2f}' for b in config['beta_geo']]
    fig,axs=plt.subplots(8,6,figsize=(7,9))
    backing=[]
    for i in range(8):
        for j,(name,x) in enumerate([('Factual',pop['factual'])]+[(n,images[n]) for n in names]):
            axs[i,j].imshow(x[i,0],cmap='gray',vmin=0,vmax=1);axs[i,j].axis('off')
            if i==0:axs[i,j].set_title(name.replace('GRACE_beta','β='),fontsize=9)
            for py in range(28):
                for px in range(28):backing.append((i,name,py,px,float(x[i,0,py,px])))
    fig.tight_layout(pad=.25);fig.savefig(out/'figures/qualitative_grid.pdf');fig.savefig(out/'qualitative_examples/grid.png',dpi=180);plt.close(fig)
    pd.DataFrame(backing,columns=['fixed_factual_row','method','pixel_y','pixel_x','intensity']).to_csv(out/'qualitative_examples/pixels.csv',index=False)
    monotonic=bool((np.diff(ablation.Phi_hat_r)<=1e-9).all());frozen=json.loads((out/'frozen.json').read_text())
    text=['# MNIST 4→9 with style shifts: GRACE v1','',
      'Completed new run; original checkpoints and historical outputs preserved. User confirmed that StyleMNIST denotes existing MNIST with style shifts.',
      '', '## Assets and task',f"Frozen SmallCNN and StrongConvAutoencoder(64): `{config['checkpoint']}`. 4 and 9 denote digit labels, not classifier IDs. 64 fixed saved factual digit-4 test images; 64 seed-fixed digit-9 training references; 5000 training images. Exact MNIST IDs and hashes are saved.",
      '', '## Main result', 'Nominal severity 1; mean over three shift families and three realizations. Cost is factual-to-returned-image pixel L2. Runtime excludes shared bank construction; the shared bank time is reported separately.', main.to_markdown(index=False),
      '', '## Geometry budget ablation',ablation.to_markdown(index=False),
      f'Observed nonincreasing shared-bank risk across beta: {monotonic}. Nested candidate pooling enforces the ordering for the retained candidate set; this is not a global-optimization guarantee.',
      '', '## Geometry and realization','Optimization and exact unregularized W2 geometry use 64-dimensional encoder coordinates. Classifier evaluation uses Dec(h). Base uses original generator images, so Dec(Enc(Base)) can differ; representation_checks.json quantifies this. Pixel L2 is primary recourse cost; realized encode/decode latent L2 is separate. Raw solver receives no current classifier and no current-validity repair is run.',
      '', '## Shift protocol','Historical rotation, translation and noise operators are reused. NEW extensions: external severity multipliers and fixed full updates from f0 replace historical scratch fitting. Future models train on shifted data; returned CF images remain fixed. Rotation/noise realizations vary transformation seeds; translation and zero-shift realizations duplicate the same deterministic distribution and are not independent replicates. This is one frozen classifier and factual set, not a multi-seed benchmark.',
      '', '## Compute',f"One server, four assigned GPUs 1,2,3,5. Sum of worker elapsed GPU occupancy: {rt.seconds.sum()/3600:.4f} GPU-hours (wall time per GPU job, not hardware utilization integral). Shared attack construction: {attack_seconds:.2f} seconds. All optimization jobs: {opt_seconds:.2f} seconds.",
      '', '## Failures and limitations',f"Final bank contains {frozen['bank_size']} persisted full-update models; {frozen['candidate_count']} retained Q candidates were cross-evaluated. Geometry feasibility is recorded independently. See logs for any interrupted/failed attempts; complete.json exists only for successful jobs.",
      'Finite bank and local solver; truncated minibatch hypergradients with frozen BN during search differ from complete-update gradients. Accepted models use all training examples, all parameters, eight epochs and fixed AdamW protocol. Pixel perturbations are constrained by paired transport and verified by exact label-preserving OT. No final future outcomes configure GRACE. Existing T-Rex/SNS baselines are explicitly paper-inspired implementations, not canonical reproductions. No autoencoder or base classifier retraining. Broader class/seed coverage and semantic human evaluation remain limitations.',
      '',f'Raw run: `{out}`. Configuration, immutable experiment manifest, bank and job provenance are saved alongside CSVs and PDF figures.']
    (out/'report.md').write_text('\n\n'.join(text)+'\n')
    result=REPO/'results/009_stylemnist_grace_v1';result.mkdir(exist_ok=True)
    for name in ['report.md','main_results.csv','geometry_ablation.csv','geometry_feasibility.csv','shift_severity.csv','representation_checks.json','resolved_config.json']:
        shutil.copyfile(out/name,result/name)
    dest=REPO/'figures/stylemnist_grace_v1';dest.mkdir(exist_ok=True)
    for p in (out/'figures').glob('*.pdf'):shutil.copyfile(p,dest/p.name)
    dump(out/'completed.json',{'future_jobs':expected,'bank_size':frozen['bank_size'],'gpu_hours':float(rt.seconds.sum()/3600),'monotonic_retained_bank_risk':monotonic})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','attack','optimize','freeze','future','report'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--job',default='prepare')
    args=parser.parse_args();out=args.output;torch.set_num_threads(1)
    config=json.loads((out/'resolved_config.json').read_text()) if (out/'resolved_config.json').exists() else json.loads(CONFIG.read_text())
    for name in ['jobs','logs','checkpoints','figures','qualitative_examples']:(out/name).mkdir(exist_ok=True)
    if not (out/'resolved_config.json').exists():
        dump(out/'resolved_config.json',config)
        shutil.copyfile(REPO/'results/008_stylemnist_asset_inventory.md',out/'stylemnist_asset_inventory.md')
        dump(out/'classifier_role_manifest.json',{'meaning':'digit 4 -> digit 9, confirmed by user','f0_checkpoint':config['checkpoint'],'f0_key':'model','ae_key':'ae','checkpoint_sha256':sha(Path(config['checkpoint'])),'seed_metadata':'historical CLI default0; original command unknown'})
        hardware=subprocess.run(['/usr/bin/nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv'],capture_output=True,text=True).stdout
        dump(out/'hardware_manifest.json',{'hostname':socket.gethostname(),'assigned_physical_gpu_ids':config['gpu_ids'],'hardware':hardware,'scheduler':'SLURM probes timed out; local per-GPU subprocess launcher'})
        dump(out/'experiment_manifest.json',{'config_sha256':sha(out/'resolved_config.json'),'source_sha256':sha(Path(__file__)),
          'checkpoint_sha256':sha(Path(config['checkpoint'])),'created_utc':datetime.now(timezone.utc).isoformat(),
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'protocol_frozen_before_expensive_run':True})
    manifest=json.loads((out/'experiment_manifest.json').read_text())
    assert manifest['config_sha256']==sha(out/'resolved_config.json')
    assert manifest['checkpoint_sha256']==sha(Path(config['checkpoint']))
    job=out/'jobs'/args.job;job.mkdir(exist_ok=True)
    if (job/'complete.json').exists():print('SKIP completed',args.job);return
    started=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
    metadata={'hostname':socket.gethostname(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'device_name':torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU','seed':config['seed'],
      'task':config['task'],'phase':args.phase,'start_utc':utc,'git_commit':manifest['git_commit'],'config_sha256':manifest['config_sha256']}
    dump(job/'started.json',metadata)
    globals()[args.phase](args,config)
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    dump(job/'complete.json',{**metadata,'seconds':time.perf_counter()-started,'end_utc':datetime.now(timezone.utc).isoformat()})

if __name__=='__main__':main()
