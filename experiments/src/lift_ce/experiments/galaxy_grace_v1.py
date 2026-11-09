"""Original multiclass LeapFactual Galaxy replication of geometry-only GRACE."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,shutil,socket,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.func import functional_call
from ..galaxy.assets import (load_classifier,load_autoencoder,load_flow,CLASSIFIER,AUTOENCODER,FLOW,TRAIN,ALL_TRAIN,TEST,CLASS_NAMES)
from ..counterfactuals.geodesic_excess import exact_w2
from ..counterfactuals.minimal_validity_repair import solve_raw
from ..counterfactuals.single_budget_solver import bank_value_gradient

ROOT=Path(__file__).resolve().parents[4]
CONFIG=Path(os.environ.get('GALAXY_CONFIG',str(ROOT/'experiments/configs/galaxy_grace_v1.json')))

def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def load(p):return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def seed(n):
    torch.manual_seed(n);torch.cuda.manual_seed_all(n);np.random.seed(n);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def process_images(ae,images,device,operation='encode',batch=4):
    with torch.no_grad():return torch.cat([getattr(ae,operation)(x.to(device)).cpu() for x in images.split(batch)])

def probabilities(model,images,target,device,batch=4):
    with torch.no_grad():return torch.cat([model(x.to(device)).softmax(1)[:,target].cpu() for x in images.split(batch)])

def target_logit(logits,target):
    keep=[i for i in range(logits.shape[1]) if i!=target]
    return logits[:,target]-torch.logsumexp(logits[:,keep],dim=1)

def paired_distance(delta,n=None):
    n=len(delta) if n is None else n
    return (delta.double().square().sum()/n).sqrt()

def project_delta(x,delta,r,n):
    delta=(x+delta).clamp(0,1)-x
    return delta*min(1.,float(r/paired_distance(delta,n).clamp_min(1e-12)))

def exact_data_w2(x,changed,labels,device):
    total=0.
    for label in range(10):
        a=x[labels==label].flatten(1).double().to(device);b=changed[labels==label].flatten(1).double().to(device)
        c=torch.cdist(a,b).square().cpu().numpy();i,j=linear_sum_assignment(c);total+=float(c[i,j].sum())
    return (total/len(x))**.5


def full_update(x,y,c,device):
    """All weights, all original training images, identical fixed Update."""
    model=load_classifier(device).train();u=c['update'];seed(u['seed'])
    opt=torch.optim.AdamW(model.parameters(),lr=u['lr'],weight_decay=u['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=u['scheduler_gamma'])
    losses=[]
    for epoch in range(u['epochs']):
        order=torch.randperm(len(x),generator=torch.Generator().manual_seed(u['seed']+epoch));total=0.
        for idx in order.split(u['batch_size']):
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[idx].to(device)),y[idx].to(device));loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();total+=float(loss.detach())*len(idx)
        scheduler.step();losses.append(total/len(x));print('update epoch',epoch,'loss',losses[-1],flush=True)
    return model.eval().requires_grad_(False),losses


def surrogate_scores(model,x,y,delta,q,targets,lr=2e-4):
    """Original GRACE one-step differentiable AdamW, multiclass CE adapter.

    Same smooth zero-variance floor as the historical truncated surrogate.
    Full-update acceptance does not rely on these approximate model parameters.
    """
    params=dict(model.named_parameters());buffers=dict(model.named_buffers())
    logits=functional_call(model,(params,buffers),(x+delta,));loss=F.cross_entropy(logits,y)
    gradients=torch.autograd.grad(loss,tuple(params.values()),create_graph=True)
    updated={n:p*(1-lr*.01)-lr*g/(torch.sqrt(g.square()+1e-12)+1e-8) for (n,p),g in zip(params.items(),gradients)}
    logits=functional_call(model,(updated,buffers),(q,))
    return logits.softmax(1).gather(1,targets[:,None]).squeeze(1)


class StreamingBank:
    chunk_size=1
    def __init__(self,records,target,ae,device):
        self.states=[load(Path(r['checkpoint'])) for r in records];self.model=load_classifier(device).eval().requires_grad_(False)
        self.count=len(records);self.target=target;self.ae=ae;self.device=device
    def logits(self,points,start=0,end=None):
        if end is None:end=self.count
        if end-start!=1:raise ValueError('StreamingBank requires chunk1')
        self.model.load_state_dict(self.states[start],strict=True)
        images=self.ae.decode(points)
        return target_logit(self.model(images),self.target)[None]
    def risk_images(self,images):
        values=[]
        for state in self.states:
            self.model.load_state_dict(state)
            p=probabilities(self.model,images,self.target,self.device)
            values.append(float(torch.relu(.5-p).double().mean()))
        return max(values)


def bank_records(out,r):
    rows=[]
    for p in sorted((out/'jobs').glob('attack_*/candidates.json')):rows.extend(json.loads(p.read_text()))
    return [row for row in rows if row['feasible'] and row['W_data']<=r+1e-6]


def prepare(args,c):
    out=args.output
    if (out/'prepared.json').exists():return
    transform=transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(150),transforms.Resize(128,antialias=True)])
    train=ImageFolder(str(TRAIN),transform=transform);alltrain=ImageFolder(str(ALL_TRAIN),transform=transform);test=ImageFolder(str(TEST),transform=transform)
    known={Path(p).name for p,y in train.samples};cal=[i for i,(p,y) in enumerate(alltrain.samples) if Path(p).name not in known]
    assert not known.intersection({Path(p).name for p,y in test.samples})
    x=torch.stack([train[i][0] for i in range(len(train))]);y=torch.tensor(train.targets)
    torch.save({'x':x,'y':y},out/'training.pt')
    rows=[]
    for split,ds,indices in [('train',train,range(len(train))),('calibration_pool',alltrain,cal),('test',test,range(len(test)))]:
        for i in indices:rows.append({'split':split,'index':i,'path':ds.samples[i][0],'label':ds.samples[i][1]})
    pd.DataFrame(rows).to_csv(out/'dataset_files.csv',index=False)
    manifests=[]
    n=c['factuals_per_target_per_split']
    for target in c['task_targets']:
        td=out/'tasks'/f'target{target:02d}';td.mkdir(parents=True,exist_ok=True)
        g=torch.Generator().manual_seed(c['seed']+target)
        choices=torch.tensor([i for i in range(len(train)) if train.targets[i]==target]);ids=choices[torch.randperm(len(choices),generator=g)[:c['references_per_target']]]
        target_images=x[ids];torch.save(target_images,td/'target_images.pt')
        np.save(td/'target_ids.npy',np.array([train.samples[i][0] for i in ids]))
        for split,ds,indices in [('calibration',alltrain,cal),('test',test,range(len(test)))]:
            source=c.get('source_by_target',{}).get(str(target))
            candidates=torch.tensor([i for i in indices if (ds.targets[i]==source if source is not None else ds.targets[i]!=target)]);chosen=candidates[torch.randperm(len(candidates),generator=g)[:n]].tolist()
            assert len(chosen)==n
            paths=[ds.samples[i][0] for i in chosen];images=torch.stack([ds[i][0] for i in chosen]);labels=torch.tensor([ds.targets[i] for i in chosen])
            torch.save({'images':images,'labels':labels},td/f'{split}_factual.pt');np.save(td/f'{split}_factual_ids.npy',np.array(paths))
            manifests.extend([{'task':target,'source_class':int(labels[j]),'split':split,'row':j,'path':p} for j,p in enumerate(paths)])
    pd.DataFrame(manifests).to_csv(out/'factual_manifest.csv',index=False)
    np.save(out/'factual_ids.npy',np.array([a['path'] for a in manifests if a['split']=='test']))
    np.save(out/'target_ids.npy',np.concatenate([np.load(out/'tasks'/f'target{t:02d}'/'target_ids.npy') for t in c['task_targets']]))
    dump(out/'dataset_manifest.json',{'class_names':CLASS_NAMES,'train_size':len(train),'calibration_pool_size':len(cal),'test_size':len(test),'train_root':str(TRAIN),'calibration_root':str(ALL_TRAIN),'test_root':str(TEST),'factuals_per_target_per_split':n,'seed':c['seed'],'files_sha256':sha(out/'dataset_files.csv'),'original_classifier_validation_test_reuse':True})
    dump(out/'prepared.json',{'training_sha256':sha(out/'training.pt'),'factual_manifest_sha256':sha(out/'factual_manifest.csv'),'time':stamp()})


def base(args,c):
    from ..galaxy.flow import progressive_leap
    out=args.output;target=int(args.job.split('_')[1]);td=out/'tasks'/f'target{target:02d}'
    model=load_classifier(args.device).eval().requires_grad_(False);ae=load_autoencoder(args.device);flow=load_flow(args.device)
    refs=load(td/'target_images.pt');ref_z=process_images(ae,refs,args.device);np.save(td/'target_latents.npy',ref_z.numpy())
    quality=[]
    for split in ['calibration','test']:
        facts=load(td/f'{split}_factual.pt');x=facts['images'];z=process_images(ae,x,args.device)
        target_y=F.one_hot(torch.full((len(x),),target,device=args.device),10).float()
        def classify(h):
            # Preserve original argmax-one-hot conditioning; stream microbatches.
            with torch.no_grad():return torch.cat([F.one_hot(model(ae.decode(q)).argmax(1),10).float() for q in h.split(c['microbatch'])])
        seed(c['seed']+target)
        begin=time.perf_counter()
        with torch.no_grad():
            trajectory=progressive_leap(flow,z.to(args.device),classify,target_y,**c['base_generator_args'],mute=True)
        raw=trajectory[-1].cpu();del trajectory
        images=process_images(ae,raw,args.device,'decode');base_seconds=time.perf_counter()-begin
        hz=process_images(ae,images,args.device)
        begin=time.perf_counter()
        with torch.no_grad():robust=progressive_leap(flow,raw.to(args.device),classify,target_y,**c['leapfactual_baseline'],mute=True)[-1].cpu()
        robust_images=process_images(ae,robust,args.device,'decode');baseline_seconds=time.perf_counter()-begin
        recon=process_images(ae,z,args.device,'decode');roundtrip=process_images(ae,hz,args.device,'decode')
        ps=probabilities(model,images,target,args.device);roundscores=probabilities(model,roundtrip,target,args.device)
        quality.append({'task':target,'split':split,'n':len(x),'reconstruction_mse':float((x-recon).square().mean()),'base_validity':float((ps>=.5).double().mean()),'base_roundtrip_validity':float((roundscores>=.5).double().mean()),'base_roundtrip_pixel_L2':float((images-roundtrip).flatten(1).norm(dim=1).mean()),'base_seconds':base_seconds,'leapfactual_extra_seconds':baseline_seconds})
        obj={'factual':x,'factual_z':z,'base_images':images,'base_z':hz,'base_raw_generator_latents':raw,'reference_z':ref_z,'leap_images':robust_images}
        torch.save(obj,td/f'{split}_population.pt')
        if split=='test':np.save(td/'base_cf_images.npy',images.numpy());np.save(td/'base_cf_latents.npy',hz.numpy())
        print('base',target,split,quality[-1],flush=True)
    pd.DataFrame(quality).to_csv(td/'base_quality.csv',index=False)


def attack(args,c):
    out=args.output;_,ri,restart=args.job.split('_');ri=int(ri);restart=int(restart);r=c['r_grid'][ri]
    job=out/'jobs'/args.job;data=load(out/'training.pt');n=len(data['x']);settings=c['attack'];s=c['seed']+1000+ri*100+restart;seed(s)
    indices=torch.randperm(n,generator=torch.Generator().manual_seed(s))[:settings['mutable_train_examples']]
    x=data['x'][indices].to(args.device);y=data['y'][indices].to(args.device)
    model=load_classifier(args.device).eval();ae=load_autoencoder(args.device)
    q=[];targets=[]
    for t in c['task_targets']:
        pop=load(out/'tasks'/f'target{t:02d}'/'calibration_population.pt')
        plan=exact_w2(pop['base_z'],pop['reference_z']);perm=torch.empty(len(pop['base_z']),dtype=torch.long);perm[plan.source_index]=plan.target_index
        for lam in [0.,.5,1.]:
            images=pop['base_images'] if lam==0 else process_images(ae,(1-lam)*pop['base_z']+lam*pop['reference_z'][perm],args.device,'decode')
            q.append(images);targets.extend([t]*len(images))
    q=torch.cat(q);targets=torch.tensor(targets)
    d=torch.randn_like(x);d=project_delta(x,d*.05*r/paired_distance(d,n),r,n).requires_grad_(True)
    opt=torch.optim.Adam([d],lr=settings['learning_rate']);trace=[]
    for step in range(settings['steps']):
        g=torch.Generator().manual_seed(s+step)
        idx=torch.randperm(len(x),generator=g)[:settings['batch_size']].to(args.device);qi=torch.randperm(len(q),generator=g)[:settings['query_batch_size']]
        pred=surrogate_scores(model,x[idx],y[idx],d[idx],q[qi].to(args.device),targets[qi].to(args.device),lr=c['update']['lr'])
        loss=F.softplus(10*(.5-pred)).mean()/10;gradient=torch.autograd.grad(loss,d)[0]
        if not torch.isfinite(gradient).all():raise ValueError('nonfinite hypergradient')
        opt.zero_grad();d.grad=-gradient;opt.step()
        with torch.no_grad():d.copy_(project_delta(x,d,r,n))
        trace.append({'step':step,'surrogate':float(loss.detach()),'paired_W_data_bound':float(paired_distance(d.detach(),n)),'gradient_norm':float(gradient.norm())})
        if step%5==0:print(args.job,trace[-1],flush=True)
    pd.DataFrame(trace).to_csv(job/'search_trace.csv',index=False)
    del model,ae,opt;torch.cuda.empty_cache();rows=[]
    for fraction in settings['radial_fractions']:
        delta=project_delta(x,d.detach()*fraction,r,n).cpu();shifted=data['x'].clone();shifted[indices]+=delta
        delta_path=job/f'delta_{fraction:.2f}.pt';torch.save({'indices':indices,'delta':delta},delta_path)
        exact=exact_data_w2(data['x'],shifted,data['y'],args.device);bound=float(paired_distance(delta,n));assert exact<=r+1e-6 and bound<=r+1e-6
        model,losses=full_update(shifted,data['y'],c,args.device)
        ckpt=out/'checkpoints'/f'{args.job}_{fraction:.2f}.pt';torch.save(model.cpu().state_dict(),ckpt);del model;torch.cuda.empty_cache()
        rows.append({'candidate':ckpt.stem,'search_radius':r,'restart':restart,'seed':s,'fraction':fraction,'W_data':exact,'paired_bound':bound,'feasible':True,'perturbation_path':str(delta_path),'perturbation_sha256':sha(delta_path),'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'update_epochs':c['update']['epochs'],'update_seed':c['update']['seed'],'training_sha256':json.loads((out/'prepared.json').read_text())['training_sha256'],'shifted_training_id':sha(delta_path),'source_beta':'fixed Base and latent geodesic mixtures, all targets','losses':losses})
    dump(job/'candidates.json',rows)


def shift(x,family,strength,s):
    if family=='brightness':return (x*strength).clamp(0,1)
    if family=='noise':return (x+strength*torch.randn(x.shape,generator=torch.Generator().manual_seed(s))).clamp(0,1)
    raise ValueError(family)


def calibration_model(args,c):
    out=args.output;i=int(args.job.split('_')[-1]);scenario=c['calibration_scenarios'][i];data=load(out/'training.pt')
    reused=reuse_shift_checkpoint(args,c)
    if reused is not None:
        dump(out/'jobs'/args.job/'model.json',{'checkpoint':str(reused),'checkpoint_sha256':sha(reused),'scenario':scenario,'reused':True});return
    x=shift(data['x'],scenario['family'],scenario['strength'],scenario['seed']);model,losses=full_update(x,data['y'],c,args.device)
    path=out/'checkpoints'/f'{args.job}.pt';torch.save(model.cpu().state_dict(),path)
    dump(out/'jobs'/args.job/'model.json',{'checkpoint':str(path),'checkpoint_sha256':sha(path),'scenario':scenario,'losses':losses})


def reuse_shift_checkpoint(args,c):
    source=c.get('reuse_shift_models_from')
    if source is None:return None
    source=Path(source);old=json.loads((source/'resolved_config.json').read_text())
    for key in ['update','future','calibration_scenarios']:assert old[key]==c[key]
    assert json.loads((source/'prepared.json').read_text())['training_sha256']==json.loads((args.output/'prepared.json').read_text())['training_sha256']
    assert json.loads((source/'model_manifest.json').read_text())['hashes']==json.loads((args.output/'model_manifest.json').read_text())['hashes']
    original=source/'checkpoints'/f'{args.job}.pt'
    metadata=source/'jobs'/args.job/('model.json' if args.phase=='calibration_model' else 'future_manifest.json')
    digest=sha(original);assert digest==json.loads(metadata.read_text())['checkpoint_sha256']
    destination=args.output/'checkpoints'/original.name
    if not destination.exists():destination.symlink_to(original)
    dump(args.output/'jobs'/args.job/'checkpoint_reuse.json',{'source':str(original),'sha256':digest,'reason':'identical frozen shifted-data update; only evaluation cohorts differ','original_complete':str(source/'jobs'/args.job/'complete.json')})
    return destination


def optimize(args,c):
    out=args.output;_,split,t,ri=args.job.split('_');t=int(t);ri=int(ri);r=c['r_grid'][ri]
    td=out/'tasks'/f'target{t:02d}';pop=load(td/f'{split}_population.pt');ae=load_autoencoder(args.device)
    records=bank_records(out,r);assert records
    bank=StreamingBank(records,t,ae,args.device);z=pop['base_z'].to(args.device);target=pop['reference_z'].to(args.device)
    plan=exact_w2(z,target);D=plan.distance;perm=torch.empty(len(z),dtype=torch.long);perm[plan.source_index]=plan.target_index;target=target[perm.to(args.device)]
    candidates=[];trace=[]
    for lam in c['geodesic_candidates']:
        candidates.append({'points':((1-lam)*z+lam*target).cpu(),'origin':f'geodesic_{lam}'})
    for beta in c['beta_grid']:
        for name,initial in [('midpoint',(z+target)/2),('target',target)]:
            def progress(row):
                if row['step']%20==0:print(args.job,beta,name,row['step'],row['true_bank_hinge'],row['slack'],flush=True)
            kept,history=solve_raw(bank,z,target,initial,D,beta*D,float('-inf'),float('inf'),c['optimizer'],on_step=progress)
            candidates.extend([{'points':a['points'],'origin':f'beta{beta}_{name}_{a["step"]}'} for a in kept])
            trace.extend([{**a,'beta_geo':beta,'initialization':name} for a in history])
    cross=[]
    for i,item in enumerate(candidates):
        q=item['points'];length=exact_w2(pop['base_z'],q).distance+exact_w2(q,pop['reference_z']).distance
        images=process_images(ae,q,args.device,'decode');risk=bank.risk_images(images);cost=float((images-pop['factual']).flatten(1).norm(dim=1).mean())
        cross.append({'index':i,'origin':item['origin'],'length':length,'risk':risk,'pixel_cost':cost})
    selected={};rows=[]
    for beta in c['beta_grid']:
        feasible=[row for row in cross if row['length']<=beta*D+max(1,D)*1e-6]
        best=min(feasible,key=lambda row:(row['risk'],row['pixel_cost'],row['index']));q=candidates[best['index']]['points']
        images=process_images(ae,q,args.device,'decode');selected[str(beta)]={'points':q,'images':images}
        rows.append({'task':t,'split':split,'r':r,'beta_geo':beta,'D':D,'B_geo':beta*D,'total_length':best['length'],'slack':beta*D-best['length'],'geometry_feasible':True,'Phi_hat_r':bank.risk_images(images),'pixel_L2_cost':float((images-pop['factual']).flatten(1).norm(dim=1).mean()),'candidate_index':best['index'],'bank_size':len(records)})
    job=out/'jobs'/args.job;torch.save(selected,job/'selected.pt');torch.save(candidates,job/'candidates.pt')
    pd.DataFrame(cross).to_csv(job/'cross_evaluation.csv',index=False);pd.DataFrame(rows).to_csv(job/'geometry.csv',index=False);pd.DataFrame(trace).to_csv(job/'trace.csv',index=False)
    dump(job/'bank_used.json',{'candidate_ids':[x['candidate'] for x in records]})


def select(args,c):
    out=args.output;model=load_classifier(args.device).eval().requires_grad_(False);rows=[]
    # Each selection sees only calibration factuals, calibration models and bank geometry.
    for t in c['task_targets']:
        pop=load(out/'tasks'/f'target{t:02d}'/'calibration_population.pt')
        for ri,r in enumerate(c['r_grid']):
            selection=load(out/'jobs'/f'opt_calibration_{t}_{ri}'/'selected.pt')
            for beta in c['beta_grid']:
                images=selection[str(beta)]['images'];hinges=[];valid=[]
                for i,_ in enumerate(c['calibration_scenarios']):
                    model.load_state_dict(load(out/'checkpoints'/f'calmodel_{i}.pt'));ps=probabilities(model,images,t,args.device)
                    hinges.append(float(torch.relu(.5-ps).double().mean()));valid.append(float((ps>=.5).double().mean()))
                rows.append({'task':t,'r':r,'r_index':ri,'beta_geo':beta,'calibration_max_mean_hinge':max(hinges),'calibration_mean_validity':np.mean(valid),'calibration_worst_validity':min(valid),'pixel_L2_cost':float((images-pop['factual']).flatten(1).norm(dim=1).mean()),'scenario_hinges':json.dumps(hinges)})
    table=pd.DataFrame(rows);table.to_csv(out/'calibration_results.csv',index=False)
    selected={}
    for t in c['task_targets']:
        best=min([x for x in rows if x['task']==t],key=lambda x:(round(x['calibration_max_mean_hinge'],6),x['pixel_L2_cost'],x['r'],x['beta_geo']))
        selected[str(t)]={'r_star':best['r'],'r_index':best['r_index'],'beta_geo_star':best['beta_geo'],'calibration_max_mean_hinge':best['calibration_max_mean_hinge']}
    dump(out/'selected_configuration.json',{'tasks':selected,'selection_rule':c['calibration_selection'],'future_outcomes_used':False,'selected_utc':stamp(),'calibration_results_sha256':sha(out/'calibration_results.csv')})


def freeze(args,c):
    out=args.output;selected=json.loads((out/'selected_configuration.json').read_text());ae=load_autoencoder(args.device)
    images={};quality=[];geom=[];risks=[]
    for t in c['task_targets']:
        spec=selected['tasks'][str(t)];ri=spec['r_index'];td=out/'tasks'/f'target{t:02d}';pop=load(td/'test_population.pt')
        q=load(out/'jobs'/f'opt_test_{t}_{ri}'/'selected.pt');methods={'Base':pop['base_images'],'LeapFactual':pop['leap_images']}
        methods.update({f'GRACE_beta{beta:.2f}':q[str(beta)]['images'] for beta in c['beta_grid']});methods['GRACE_selected']=q[str(spec['beta_geo_star'])]['images'];images[str(t)]=methods
        records=bank_records(out,spec['r_star']);bank=StreamingBank(records,t,ae,args.device)
        for name,x in methods.items():
            z=process_images(ae,x,args.device);mse=float((x-pop['factual']).square().mean());recon=process_images(ae,z,args.device,'decode')
            quality.append({'task':t,'method':name,'pixel_L2_cost':float((x-pop['factual']).flatten(1).norm(dim=1).mean()),'realized_latent_L2_displacement':float((z-pop['factual_z']).norm(dim=1).mean()),'factual_PSNR':-10*np.log10(max(mse,1e-15)),'output_reconstruction_mse':float((recon-x).square().mean())})
            risks.append({'task':t,'method':name,'r_star':spec['r_star'],'beta_geo_star':spec['beta_geo_star'],'Phi_hat_r':bank.risk_images(x)})
        del bank
        geom.append(pd.read_csv(out/'jobs'/f'opt_test_{t}_{ri}'/'geometry.csv'))
    torch.save(images,out/'frozen_images.pt');pd.DataFrame(quality).to_csv(out/'image_quality.csv',index=False);pd.DataFrame(risks).to_csv(out/'finite_bank_risk.csv',index=False)
    pd.concat(geom).to_csv(out/'geometry_feasibility.csv',index=False)
    allrecords=[]
    for file in sorted((out/'jobs').glob('attack_*/candidates.json')):allrecords.extend(json.loads(file.read_text()))
    pd.DataFrame(allrecords).to_csv(out/'attack_bank_manifest.csv',index=False)
    dump(out/'frozen.json',{'frozen_utc':stamp(),'images_sha256':sha(out/'frozen_images.pt'),'selected_configuration_sha256':sha(out/'selected_configuration.json'),'bank_manifest_sha256':sha(out/'attack_bank_manifest.csv'),'future_outcomes_seen':False})
    # Top-level convenience files retain task dimension; per-task IDs remain authoritative.
    np.save(out/'base_cf_images.npy',np.stack([load(out/'tasks'/f'target{t:02d}'/'test_population.pt')['base_images'].numpy() for t in c['task_targets']]))
    np.save(out/'base_cf_latents.npy',np.stack([load(out/'tasks'/f'target{t:02d}'/'test_population.pt')['base_z'].numpy() for t in c['task_targets']]))
    np.save(out/'target_latents.npy',np.stack([np.load(out/'tasks'/f'target{t:02d}'/'target_latents.npy') for t in c['task_targets']]))


def future(args,c):
    out=args.output;_,family,si,rep=args.job.split('_');si=int(si);rep=int(rep);severity=c['future']['severity'][si]
    strength=1-.2*severity if family=='brightness' else .04*severity;s=c['future']['seed_base']+rep
    frozen=json.loads((out/'frozen.json').read_text());assert frozen['images_sha256']==sha(out/'frozen_images.pt')
    ckpt=reuse_shift_checkpoint(args,c)
    if ckpt is None:
        data=load(out/'training.pt');x=shift(data['x'],family,strength,s);model,losses=full_update(x,data['y'],c,args.device)
        ckpt=out/'checkpoints'/f'{args.job}.pt';torch.save(model.cpu().state_dict(),ckpt);model.to(args.device)
    else:
        model=load_classifier(args.device).eval().requires_grad_(False);model.load_state_dict(load(ckpt));losses=[]
    rows=[]
    for target,methods in load(out/'frozen_images.pt').items():
        t=int(target)
        for name,images in methods.items():
            with torch.no_grad():logits=torch.cat([model(q.to(args.device)).cpu() for q in images.split(c['microbatch'])])
            ps=logits.softmax(1)[:,t]
            rows.append({'task':t,'method':name,'family':family,'severity':severity,'strength':strength,'realization':rep,'shift_seed':s,'future_validity':float((ps>=.5).double().mean()),'target_argmax_validity':float((logits.argmax(1)==t).double().mean()),'mean_hinge':float(torch.relu(.5-ps).double().mean()),'checkpoint':str(ckpt)})
    job=out/'jobs'/args.job;pd.DataFrame(rows).to_csv(job/'metrics.csv',index=False)
    dump(job/'future_manifest.json',{'scenario':{'family':family,'severity':severity,'strength':strength,'seed':s},'checkpoint_sha256':sha(ckpt),'losses':losses,'fixed_cf_images_sha256':frozen['images_sha256']})


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','base','attack','calibration_model','optimize','select','freeze','future']);p.add_argument('--output',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();out=a.output
    torch.set_num_threads(1);seed(20260916)
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.16)
    c=json.loads((out/'resolved_config.json').read_text()) if (out/'resolved_config.json').exists() else json.loads(CONFIG.read_text())
    c['base_generator_args']={k:v for k,v in c['base_generator'].items() if k!='solver'}
    for d in ['jobs','logs','tasks','checkpoints','figures','qualitative_examples']:(out/d).mkdir(exist_ok=True)
    if not (out/'experiment_manifest.json').exists():
        dump(out/'resolved_config.json',c)
        shutil.copyfile(ROOT/'results/010_galaxy_asset_inventory.md',out/'galaxy_asset_inventory.md');shutil.copyfile(ROOT/'results/010_galaxy_asset_checks.json',out/'asset_checks.json')
        models={'classifier':str(CLASSIFIER),'encoder_decoder':str(AUTOENCODER),'flow':str(FLOW),'class_names':CLASS_NAMES,'latent_dim':64,'hashes':{str(p):sha(p) for p in [CLASSIFIER,AUTOENCODER,FLOW]}}
        dump(out/'model_manifest.json',models)
        hw=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv'],text=True)
        dump(out/'hardware_manifest.json',{'hostname':socket.gethostname(),'gpu_ids':c['gpu_ids'],'hardware':hw,'scheduler':'sinfo unavailable (external8s timeout); per-GPU subprocess queues','memory_cap_per_worker_GiB':12.8})
        dump(out/'experiment_manifest.json',{'created_utc':stamp(),'config_sha256':sha(out/'resolved_config.json'),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'task_file_sha256':sha(ROOT/'tasks/001_galaxy_task.md'),'source_sha256':sha(Path(__file__))})
    manifest=json.loads((out/'experiment_manifest.json').read_text());assert manifest['config_sha256']==sha(out/'resolved_config.json')
    job=out/'jobs'/a.job;job.mkdir(exist_ok=True)
    if (job/'complete.json').exists():print('SKIP',a.job);return
    start=time.perf_counter();meta={'job':a.job,'phase':a.phase,'hostname':socket.gethostname(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'cuda_device_name':torch.cuda.get_device_name(),'seed':c['seed'],'task':a.job,'r_and_beta':'job-specific values encoded in job name and geometry/search CSVs; full grid in resolved_config.json','git_commit':manifest['git_commit'],'config_sha256':manifest['config_sha256'],'start_utc':stamp()};dump(job/'started.json',meta)
    globals()[a.phase](a,c);torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'seconds':time.perf_counter()-start,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9})

if __name__=='__main__':main()
