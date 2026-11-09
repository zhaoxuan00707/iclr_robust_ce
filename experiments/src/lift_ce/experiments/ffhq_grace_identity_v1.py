"""Matched FFHQ GRACE experiment with a fixed factual-content bottleneck."""
import os, json, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from PIL import Image
from ..ffhq.bottleneck import gradient_basis, FrozenContent, ConditionalBank
from .ffhq_grace_v1 import FaceBank, Generator, SmileClassifier, records, features, logits, decode
from .galaxy_grace_v1 import load, dump, sha, stamp
from ..counterfactuals.geodesic_excess import exact_w2
from ..counterfactuals.minimal_validity_repair import solve_raw
ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / 'experiments/configs/ffhq_grace_identity_v1.json'


def calibration_pool(parent):
    items = [load(p) for p in sorted((parent/'jobs').glob('sample_calibration_*/samples.pt'))]
    return {key: torch.cat([v[key] for v in items]) for key in ['w','x','y']}


def initialize(a,c):
    out=a.output;parent=Path(c['parent_run'])
    for name in ['jobs','logs','checkpoints','figures','qualitative_examples']:(out/name).mkdir(exist_ok=True)
    dump(out/'resolved_config.json',c)
    shutil.copyfile(ROOT/'results/015_ffhq_identity_v1/identity_bottleneck_inventory.md',out/'identity_bottleneck_inventory.md')
    names=['training.pt','references.pt','calibration_population.pt','test_population.pt','calibration_factuals.pt','test_factuals.pt','test_factual_ids.npy']
    inputs={str(parent/name):sha(parent/name) for name in names}
    for p in sorted((parent/'checkpoints').glob('*.pt')): inputs[str(p)]=sha(p)
    for name in ['frozen.pt','frozen.json','resolved_config.json','selected_configuration.json','attack_bank_manifest.csv']:
        inputs[str(parent/name)]=sha(parent/name)
    dump(out/'input_manifest.json',inputs)
    shutil.copyfile(parent/'test_factual_ids.npy',out/'factual_ids.npy')
    models=Path(c['identity']['root'])/'models/buffalo_l'
    for name in ['det_10g.onnx','w600k_r50.onnx']:
        shutil.copyfile(models/name,out/'checkpoints'/name)
    dump(out/'identity_model_manifest.json',{'source':c['identity']['source'],'models':{name:sha(out/'checkpoints'/name) for name in ['det_10g.onnx','w600k_r50.onnx']},'library':'insightface 0.7.3, onnxruntime 1.22.1 CPU','license':'upstream pretrained models: non-commercial research','new_evaluation_component':True})
    dump(out/'protocol_frozen.json',{'utc':stamp(),'gradient_ids':'all 256 training IDs only','energy_criterion':.95,'qualitative_indices':list(range(8)),'parent_future_models_previously_evaluated':True,'new_test_outcomes_used_for_choices':False,'config_sha256':sha(out/'resolved_config.json')})


def gradient(a,c):
    shard=int(a.job.split('_')[-1]);d=load(Path(c['parent_run'])/'training.pt');g=Generator(a.device);clf=SmileClassifier(a.device);grads=[]
    indices=list(range(shard,len(d['w']),8))
    for i in indices:
        w=d['w'][i:i+1].to(a.device).requires_grad_(True)
        score=clf(g.decode(w)).softmax(1)[0,0]
        grad=torch.autograd.grad(score,w)[0]
        assert torch.isfinite(grad).all();grads.append(grad.cpu())
        print('gradient',i,float(grad.norm()),flush=True)
    torch.save({'indices':indices,'gradients':torch.cat(grads)},a.output/'jobs'/a.job/'gradients.pt')


def basis(a,c):
    parent=Path(c['parent_run']);train=load(parent/'training.pt');cal=calibration_pool(parent);grad=torch.empty_like(train['w'])
    for p in (a.output/'jobs').glob('gradient_*/gradients.pt'):
        d=load(p);grad[d['indices']]=d['gradients']
    u,v,s,k=gradient_basis(grad,c['bottleneck']['energy'])
    path=a.output/'checkpoints/task_basis.pt';torch.save({'task':u,'content':v,'singular_values':s,'gradients':grad},path)
    assert torch.allclose(u.T@u,torch.eye(k,dtype=torch.double),atol=1e-10)
    scores={}
    for name,projector in [('u',u@u.T),('v',v@v.T)]:
        x=(train['w'].double()@projector).numpy();xt=(cal['w'].double()@projector).numpy()
        probe=LogisticRegression(C=1.,max_iter=2000,random_state=20260917).fit(x,train['y'].numpy())
        pred=probe.predict(xt);scores['smile_probe_'+name]=accuracy_score(cal['y'],pred)
        scores['smile_probe_'+name+'_balanced']=balanced_accuracy_score(cal['y'],pred)
        torch.save({'coef':torch.from_numpy(probe.coef_),'intercept':torch.from_numpy(probe.intercept_)},a.output/'checkpoints'/f'probe_{name}.pt')
    scores.update(task_dimension=k,content_dimension=512-k,latent_dimensions=512,gradient_energy=float(s[:k].square().sum()/s.square().sum()),calibration_majority_accuracy=float(torch.bincount(cal['y']).max()/len(cal['y'])))
    dump(a.output/'probe_diagnostics.json',scores)
    dump(a.output/'bottleneck_manifest.json',{'type':'approximate training-gradient SVD','basis_path':str(path),'sha256':sha(path),'dimension':k,'energy':scores['gradient_energy'],'frozen_utc':stamp(),'training_sha256':sha(parent/'training.pt'),'training_ids':train['ids'],'test_used':False,'centered':False,'u':'h @ U_task, k coordinates','v':'h_factual - (h_factual @ U_task) @ U_task.T, immutable 512-vector','decode':'StyleGAN3(v_factual + u @ U_task.T)','class_independence_established':False})
    print(scores,flush=True)


def reconstruction(a,c):
    cal=calibration_pool(Path(c['parent_run']));basis=load(a.output/'checkpoints/task_basis.pt')['task'];g=Generator(a.device);rows=[]
    for i in range(int(a.job.split('_')[-1]),len(cal['w']),8):
        h=cal['w'][i:i+1].to(a.device);r=FrozenContent(basis,h);rebuilt=r.realize(r.encode(h))
        with torch.no_grad(): x=g.decode(h);y=g.decode(rebuilt)
        rows.append({'calibration_index':i,'latent_max_error':float((h-rebuilt).abs().max()),'reconstruction_error':float((x-y).square().mean()),'native_max_error':float((x-y).abs().max())})
    pd.DataFrame(rows).to_csv(a.output/'jobs'/a.job/'reconstruction.csv',index=False)


def optimize(a,c):
    _,mode,split,ri,bi=a.job.split('_');ri=int(ri);bi=int(bi);parent=Path(c['parent_run']);r=c['r_grid'][ri];beta=c['beta_grid'][bi]
    pop=load(parent/f'{split}_population.pt');target=load(parent/'references.pt')['w'].to(a.device);z=pop['base_w'].to(a.device)
    full=FaceBank(records(parent,r),a.device);realization=None
    if mode=='bottleneck':
        realization=FrozenContent(load(a.output/'checkpoints/task_basis.pt')['task'],pop['factual_w'].to(a.device))
        z=realization.encode(z).float();target=realization.encode(target).float();bank=ConditionalBank(full,realization)
    else:bank=full
    plan=exact_w2(z,target);D=plan.distance;perm=torch.empty(len(z),dtype=torch.long);perm[plan.source_index]=plan.target_index;target=target[perm.to(a.device)]
    candidates=[{'points':((1-lam)*z+lam*target).cpu(),'origin':f'geodesic{lam}'} for lam in c['geodesic_candidates']];trace=[]
    for name,initial in [('midpoint',(z+target)/2),('target',target)]:
        kept,history=solve_raw(bank,z,target,initial,D,beta*D,float('-inf'),float('inf'),c['optimizer'],on_step=lambda row:print(a.job,name,row['step'],row['true_bank_hinge'],flush=True) if row['step']%10==0 else None)
        candidates.extend([{'points':v['points'],'origin':f'{name}_{v["step"]}'} for v in kept]);trace.extend([{**v,'start':name} for v in history])
    rows=[]
    for i,v in enumerate(candidates):
        q=v['points'];length=exact_w2(z,q).distance+exact_w2(q,target).distance;x=bank.canonical(q.to(a.device));v['x']=x;v['features']=bank.cached_features(q.to(a.device)).cpu()
        v['w']=realization.realize(q.to(a.device)).cpu() if realization else q
        rows.append({'candidate':i,'length':length,'risk':bank.risk(q.to(a.device)),'pixel_cost':float((x-pop['factual_x']).flatten(1).norm(dim=1).mean()),'origin':v['origin']})
    job=a.output/'jobs'/a.job;torch.save(candidates,job/'candidates.pt');pd.DataFrame(rows).to_csv(job/'candidates.csv',index=False);pd.DataFrame(trace).to_csv(job/'trace.csv',index=False)
    dump(job/'meta.json',{'r':r,'beta_geo':beta,'D':D,'mode':mode,'bank_ids':[v['candidate'] for v in records(parent,r)]})


def pool(a,c):
    _,mode,split,ri=a.job.split('_');ri=int(ri);all_candidates=[];rows=[]
    for bi in range(4):
        job=a.output/'jobs'/f'opt_{mode}_{split}_{ri}_{bi}';items=load(job/'candidates.pt');table=pd.read_csv(job/'candidates.csv');D=json.loads((job/'meta.json').read_text())['D']
        for row in table.to_dict('records'):row['candidate']+=len(all_candidates);rows.append(row)
        all_candidates+=items
    selected={};geom=[]
    for beta in c['beta_grid']:
        feasible=[v for v in rows if v['length']<=beta*D+1e-6*max(1,D)];best=min(feasible,key=lambda v:(v['risk'],v['pixel_cost'],v['candidate']));selected[str(beta)]=all_candidates[best['candidate']]
        geom.append({'r':c['r_grid'][ri],'beta_geo':beta,'D':D,'B_geo':beta*D,'total_length':best['length'],'slack':beta*D-best['length'],'Phi_hat_r':best['risk'],'pixel_L2_cost':best['pixel_cost'],'mode':mode})
    torch.save(selected,a.output/'jobs'/a.job/'selected.pt');pd.DataFrame(geom).to_csv(a.output/'jobs'/a.job/'geometry.csv',index=False)


def select(a,c):
    clf=SmileClassifier(a.device);parent=Path(c['parent_run']);rows=[]
    for ri,r in enumerate(c['r_grid']):
        selected=load(a.output/'jobs'/f'pool_bottleneck_calibration_{ri}'/'selected.pt');geom=pd.read_csv(a.output/'jobs'/f'pool_bottleneck_calibration_{ri}'/'geometry.csv')
        for beta in c['beta_grid']:
            item=selected[str(beta)];scores=[];valid=[]
            for i in range(4):
                projection=load(parent/'checkpoints'/f'calmodel_{i}.pt')['projection'].to(a.device)
                with torch.no_grad():p=clf.logits_features(item['features'].to(a.device),projection).softmax(1)[:,0]
                scores.append(float(torch.relu(.5-p).double().mean()));valid.append(float((p>=.5).double().mean()))
            row=geom[geom.beta_geo==beta].iloc[0].to_dict();rows.append({**row,'r_index':ri,'calibration_max_mean_hinge':max(scores),'calibration_mean_validity':np.mean(valid)})
    pd.DataFrame(rows).to_csv(a.output/'calibration_results.csv',index=False);best=min(rows,key=lambda v:(round(v['calibration_max_mean_hinge'],6),v['pixel_L2_cost'],v['r'],v['beta_geo']))
    basis=json.loads((a.output/'bottleneck_manifest.json').read_text())
    dump(a.output/'selected_configuration.json',{'r_star':best['r'],'r_index':best['r_index'],'beta_geo_star':best['beta_geo'],'bottleneck_type':basis['type'],'bottleneck_dimension':basis['dimension'],'basis_path':basis['basis_path'],'basis_sha256':basis['sha256'],'selected_utc':stamp(),'future_outcomes_used':False,'rule':c['selection']})


def freeze(a,c):
    out=a.output;parent=Path(c['parent_run']);spec=json.loads((out/'selected_configuration.json').read_text());ri=spec['r_index'];pop=load(parent/'test_population.pt');selected=load(out/'jobs'/f'pool_bottleneck_test_{ri}'/'selected.pt');oldspec=json.loads((parent/'selected_configuration.json').read_text())
    if spec['r_star']==oldspec['r_star']:
        old=load(parent/'frozen.pt');full=old[f"GRACE_beta{spec['beta_geo_star']:.2f}"].copy();full['w']=full['points'];full['reused_from']=str(parent/'frozen.pt')
    else:full=load(out/'jobs'/f'pool_full_test_{ri}'/'selected.pt')[str(spec['beta_geo_star'])]
    bank=FaceBank(records(parent,spec['r_star']),a.device)
    methods={'Base':{'w':pop['base_w'],'x':pop['base_x']},'Full_GRACE':full}
    methods.update({f'Bottleneck_beta{b:.2f}':selected[str(b)] for b in c['beta_grid']})
    methods['Bottleneck_GRACE']=selected[str(spec['beta_geo_star'])]
    basis=load(out/'checkpoints/task_basis.pt');real=FrozenContent(basis['task'],pop['factual_w'].to(a.device));u=real.encode(pop['base_w'].to(a.device)).float();w=real.realize(u)
    methods['Conditional_Base']={'w':w.cpu(),'points':u.cpu(),'x':bank.canonical(w)}
    risks=[]
    for name,item in methods.items():
        item['features']=bank.cached_features(item['w'].to(a.device)).cpu()
        drift=float(((item['w'].double()-pop['factual_w'].double())@basis['content']).abs().max())
        if name.startswith('Bottleneck') or name=='Conditional_Base':assert drift<2e-6
        risks.append({'method':name,'Phi_hat_r':bank.risk(item['w'].to(a.device)),'content_coordinate_max_drift':drift,'pixel_L2_224':float((item['x']-pop['factual_x']).flatten(1).norm(dim=1).mean())})
    torch.save(methods,out/'frozen.pt');pd.DataFrame(risks).to_csv(out/'finite_bank_risk.csv',index=False)
    shutil.copyfile(out/'jobs'/f'pool_bottleneck_test_{ri}'/'geometry.csv',out/'geometry_ablation.csv')
    pd.DataFrame(records(parent,spec['r_star'])).to_csv(out/'attack_bank_manifest.csv',index=False)
    dump(out/'frozen.json',{'sha256':sha(out/'frozen.pt'),'basis_sha256':sha(out/'checkpoints/task_basis.pt'),'selected_sha256':sha(out/'selected_configuration.json'),'frozen_utc':stamp(),'parent_future_models_previously_evaluated':True})


def future(a,c):
    parent=Path(c['parent_run']);out=a.output;frozen=json.loads((out/'frozen.json').read_text());assert sha(out/'frozen.pt')==frozen['sha256'];clf=SmileClassifier(a.device);images=load(out/'frozen.pt');rows=[]
    paths=sorted((parent/'jobs').glob('future_*/model.json'))
    for path in paths[int(a.job.split('_')[-1])::8]:
        meta=json.loads(path.read_text());assert sha(meta['checkpoint'])==meta['checkpoint_sha256'];projection=load(Path(meta['checkpoint']))['projection'].to(a.device)
        for name,item in images.items():
            with torch.no_grad(): p=clf.logits_features(features(clf,item['x'],a.device).to(a.device),projection).softmax(1)[:,0]
            for i,value in enumerate(p.tolist()):rows.append({'method':name,'test_index':i,**meta['scenario'],'probability':value,'future_validity':int(value>=.5),'hinge':max(0,.5-value),'checkpoint':meta['checkpoint']})
    pd.DataFrame(rows).to_csv(out/'jobs'/a.job/'future.csv',index=False)


def metrics(a,c):
    import lpips
    import cv2
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    idx=int(a.job.split('_')[-1]);out=a.output;parent=Path(c['parent_run']);facts=load(parent/'test_factuals.pt');methods=load(out/'frozen.pt');g=Generator(a.device);clf=SmileClassifier(a.device)
    # Reuse the official detector/recognizer with one CPU thread per GPU worker.
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    modelroot=out/'identity_assets/models/buffalo_l';modelroot.mkdir(parents=True,exist_ok=True)
    for name in ['det_10g.onnx','w600k_r50.onnx']:
        link=modelroot/name
        try:link.symlink_to(out/'checkpoints'/name)
        except FileExistsError:pass
    face=FaceAnalysis(name='buffalo_l',root=str(out/'identity_assets'),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'],sess_options=opts)
    face.prepare(ctx_id=-1,det_thresh=.5,det_size=(640,640))
    torch.hub.set_dir('/tmp/ffhq_torch_hub');lp=lpips.LPIPS(net='squeeze').to(a.device).eval()
    def embedding(x):
        rgb=((x[0].cpu().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8)
        faces=face.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        chosen=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None
        return (chosen.normed_embedding if chosen is not None else None),len(faces),rgb
    with torch.no_grad(): factual=g.decode(facts['w'][idx:idx+1].to(a.device)).clamp(-1,1)
    emb,nfaces,rgb=embedding(factual);Image.fromarray(rgb).save(out/'qualitative_examples'/f'{idx}_Factual.png');rows=[];embeddings={}
    if emb is not None:embeddings['Factual']=emb
    for name,item in methods.items():
        with torch.no_grad():
            native=g.decode(item['w'][idx:idx+1].to(a.device));x=clf.canonical(native)
            error=float((x.cpu()-item['x'][idx:idx+1]).abs().max());assert error<2e-5
            native=native.clamp(-1,1);percept=float(lp(factual,native));pixel=float(((native-factual)/2).flatten(1).norm(dim=1))
        cfemb,ncf,rgb=embedding(native);Image.fromarray(rgb).save(out/'qualitative_examples'/f'{idx}_{name}.png')
        identity=float(np.dot(emb,cfemb)) if emb is not None and cfemb is not None else np.nan
        if cfemb is not None:embeddings[name]=cfemb
        rows.append({'method':name,'test_index':idx,'identity_similarity':identity,'factual_faces_detected':nfaces,'cf_faces_detected':ncf,'LPIPS_squeeze':percept,'pixel_L2_1024':pixel,'decoded_image_max_error':error})
    pd.DataFrame(rows).to_csv(out/'jobs'/a.job/'metrics.csv',index=False);np.savez(out/'jobs'/a.job/'identity_embeddings.npz',**embeddings)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','gradient','basis','reconstruction','optimize','pool','select','freeze','future','metrics']);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.14);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.manual_seed(20260917)
    config=a.output/'resolved_config.json';c=json.loads((config if config.exists() else CONFIG).read_text());job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
    if (job/'complete.json').exists():return
    start=time.time();meta={'job':a.job,'phase':a.phase,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta)
    globals()[a.phase](a,c);torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9})

if __name__=='__main__':main()
