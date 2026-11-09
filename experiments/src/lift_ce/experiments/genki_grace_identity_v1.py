"""GENKI-head finite-bank GRACE in a frozen-content four-style realization."""
import argparse,hashlib,json,os,shutil,socket,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from ..ffhq.assets import Generator,GAN
from ..ffhq.genki_classifier import GenkiSmileClassifier
from ..ffhq.style_editor import StyleEditor,region_mask
from .ffhq_smile_editor_v1 import identity_models
from .ffhq_smile_editor_v2 import Measurement,passes_quality
from .ffhq_grace_v1 import shift
from .galaxy_grace_v1 import load,dump,sha,stamp,project_delta,paired_distance
from ..counterfactuals.geodesic_excess import exact_w2
from ..counterfactuals.minimal_validity_repair import solve_raw
ROOT=Path(__file__).resolve().parents[4]


def classifier(a):return GenkiSmileClassifier(a.output/'checkpoints/f0.pt',a.device)


def feature_batches(clf,x,device):
    with torch.no_grad():return torch.cat([clf.features(v.to(device)).cpu() for v in x.split(16)])


def head_update(clf,features,y,c,device):
    u=c['update'];head=nn.Linear(512,2).to(device);head.load_state_dict(clf.head.state_dict());opt=torch.optim.SGD(head.parameters(),lr=u['lr'])
    x=features.to(device);y=y.to(device);losses=[]
    for step in range(u['steps']):
        opt.zero_grad();loss=F.cross_entropy(head(x),y)+.5*u['l2']*head.weight.square().sum();loss.backward();opt.step()
        if step%20==0 or step==u['steps']-1:losses.append(float(loss.detach()))
    return {k:v.detach().cpu() for k,v in head.state_dict().items()},losses


def state_scores(features,states):
    weights=torch.stack([s['weight'][0]-s['weight'][1] for s in states]).to(features)
    bias=torch.stack([s['bias'][0]-s['bias'][1] for s in states]).to(features)
    return torch.einsum('nd,kd->kn',features,weights)+bias[:,None]


def population_risk(probabilities):return torch.relu(.5-probabilities).double().mean(1).max()


class GenkiMeasurement(Measurement):
    def __init__(self,a,c,w):
        import lpips
        self.a=a;self.c=c;self.g=Generator(a.device);self.editor=StyleEditor(self.g);self.clf=classifier(a)
        self.arc,self.mtcnn,self.facenet=identity_models(a);torch.hub.set_dir('/tmp/ffhq_torch_hub')
        self.metric=lpips.LPIPS(net='squeeze').to(a.device).eval();self.w=w;self.outside=1-region_mask(224,a.device)
        with torch.no_grad():
            self.s=self.editor.encode(w);self.native=self.g.decode(w);self.x=self.clf.canonical(self.native)
            self.p=float(self.clf(self.native).softmax(1)[0,0]);self.arc0,self.face0,_,_=self.identities(self.native)


def initialize(a,c):
    out=a.output;genki=Path(c['parent_genki']);editor=Path(c['parent_editor']);parent=Path(c['parent_ffhq'])
    for name in ['jobs','logs','checkpoints','figures','qualitative_examples']:(out/name).mkdir(exist_ok=True)
    dump(out/'resolved_config.json',c);dump(out/'construction_clock.json',{'start_unix':time.time(),'start_utc':stamp()})
    inputs=[genki/'checkpoints/selected.pt',genki/'dataset_manifest.csv',editor/'checkpoints/controls.pt',editor/'checkpoints/statistics.pt',parent/'training.pt',GAN]
    for src,name in [(inputs[0],'f0.pt'),(inputs[2],'controls.pt'),(inputs[3],'statistics.pt'),(Path('/tmp/ffhq_facenet_vggface2.pt'),'facenet_vggface2.pt')]:
        shutil.copyfile(src,out/'checkpoints'/name)
    clf=classifier(a);g=Generator(a.device);style=StyleEditor(g);indices=load(out/'checkpoints/controls.pt')['style4']['indices'];std=load(out/'checkpoints/statistics.pt')['style_std'][indices]
    train=load(parent/'training.pt');train_features=feature_batches(clf,train['x'],a.device)
    with torch.no_grad():p=clf.head(train_features.to(a.device)).softmax(1)[:,0].cpu();refidx=torch.nonzero(p>=.5).flatten()[:8];assert len(refidx)==8
    with torch.no_grad():refstyles=style.encode(train['w'][refidx].to(a.device)).cpu()
    torch.save({'u':refstyles[:,indices]/std,'w':train['w'][refidx],'features':train_features[refidx],'ids':[train['ids'][i] for i in refidx],
                'nominal_probabilities':p[refidx]},out/'references.pt')
    arc,mtcnn,facenet=identity_models(a);eligibility=[]
    import cv2
    for split in ['calibration','test']:
        paths=sorted((parent/'jobs').glob(f'sample_{split}_*/samples.pt'));inputs+=paths
        items=[load(p) for p in paths];ws=torch.cat([i['w'] for i in items]);xs=torch.cat([i['x'] for i in items]);ids=sum([i['ids'] for i in items],[])
        with torch.no_grad():probs=clf.head(feature_batches(clf,xs,a.device).to(a.device)).softmax(1)[:,0].cpu()
        chosen=[];images=[];coordinates=[]
        for i in range(len(ws)):
            record={'split':split,'pool_index':i,'id':ids[i],'nominal_probability':float(probs[i]),'selected':False}
            if probs[i]>=.5:record['reason']='nominal_smiling';eligibility.append(record);continue
            with torch.no_grad():native=g.decode(ws[i:i+1].to(a.device));x=clf.canonical(native).cpu()
            rgb=((native[0].clamp(-1,1).cpu().permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8)
            faces=arc.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));crop=mtcnn(Image.fromarray(rgb))
            if not faces or crop is None:record['reason']='identity_detection_missing';eligibility.append(record);continue
            with torch.no_grad():pn=float(clf(native).softmax(1)[0,0]);s=style.encode(ws[i:i+1].to(a.device)).cpu()
            if pn>=.5:record['reason']='native_nominal_smiling';eligibility.append(record);continue
            record.update(selected=True,reason='first_eligible',native_probability=pn);eligibility.append(record)
            j=len(chosen);Image.fromarray(rgb).save(out/'qualitative_examples'/f'{split}_{j}_factual.png')
            chosen.append(i);images.append(x);coordinates.append(s[:,indices]/std)
            if len(chosen)==8:break
        assert len(chosen)==8,f'{split}: insufficient eligible factuals'
        torch.save({'w':ws[chosen],'x':torch.cat(images),'u':torch.cat(coordinates),'ids':[ids[i] for i in chosen]},out/f'{split}_factuals.pt')
    pd.DataFrame(eligibility).to_csv(out/'factual_eligibility.csv',index=False)
    del g,arc,mtcnn,facenet;torch.cuda.empty_cache()
    dataset=pd.read_csv(genki/'dataset_manifest.csv');dataset=dataset[dataset.split=='train'].copy()
    images=torch.cat([clf.canonical(TF.to_tensor(Image.open(p).convert('RGB'))[None]*2-1) for p in dataset.image_path])
    feats=feature_batches(clf,images,a.device);y=torch.tensor(1-dataset.smile.to_numpy())
    torch.save({'x':images,'features':feats,'y':y,'ids':dataset['index'].tolist()},out/'training.pt')
    dataset[['index','smile','sha256','image_path']].to_csv(out/'training_manifest.csv',index=False)
    inputs+=list(map(Path,dataset.image_path));dump(out/'input_manifest.json',{str(p):sha(p) for p in inputs})
    dump(out/'protocol_frozen.json',{'utc':stamp(),'config_sha256':sha(out/'resolved_config.json'),'training_count':len(dataset),
                                  'selected_channels':indices.tolist(),'std':std.tolist(),'f0_sha256':sha(out/'checkpoints/f0.pt'),
                                  'generated_factuals_not_real_GENKI':True,'reference_smiles_not_guaranteed_after_content_transplant':True})
    dump(out/'hardware_manifest.json',{'hostname':socket.gethostname(),'gpu_ids':list(range(8)),
          'hardware':subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv'],text=True)})


def base(a,c):
    _,split,i=a.job.split('_');i=int(i);facts=load(a.output/f'{split}_factuals.pt');m=GenkiMeasurement(a,c,facts['w'][i:i+1].to(a.device))
    idx=load(a.output/'checkpoints/controls.pt')['style4']['indices'].to(a.device);std=load(a.output/'checkpoints/statistics.pt')['style_std'].to(a.device)
    q=torch.zeros(1,4,device=a.device);current,rgb,arc,face=m.measure(q,idx,std);assert passes_quality(current,c)
    step_size=c['base']['initial_step'];rows=[dict(step=-1,accepted=True,**current)];points=[q.cpu()];crossed=False
    for step in range(c['base']['steps']):
        v=q.detach().requires_grad_(True);score=m.clf(m.decode(v,idx,std)).softmax(1)[0,0];grad,=torch.autograd.grad(score,v)
        trial=(q+step_size*grad/(grad.norm()+1e-12)).clamp(-16,16).detach();row,rgb1,arc1,face1=m.measure(trial,idx,std)
        accept=passes_quality(row,c) and row['smile_score']>=current['smile_score']
        rows.append(dict(step=step,accepted=accept,**row));points.append(trial.cpu())
        if accept and row['smile_score']>=.5:
            lo=q.clone();hi=trial;hirow=row;hirgb=rgb1;hiarc=arc1;hiface=face1
            for _ in range(c['base']['bisection_steps']):
                mid=(lo+hi)/2;mr,mi,ma,mf=m.measure(mid,idx,std)
                if passes_quality(mr,c) and mr['smile_score']>=.5:hi=mid;hirow=mr;hirgb=mi;hiarc=ma;hiface=mf
                else:lo=mid
            q=hi;current=hirow;rgb=hirgb;arc=hiarc;face=hiface;crossed=True;break
        if accept:q=trial;current=row;rgb=rgb1;arc=arc1;face=face1;step_size=min(c['base']['maximum_step'],step_size*1.1)
        else:step_size=max(c['base']['minimum_step'],step_size*.5)
        if step%10==0:print(a.job,step,current['smile_score'],flush=True)
    with torch.no_grad():native=m.decode(q,idx,std);x=m.clf.canonical(native);f=m.clf.features(x)
    job=a.output/'jobs'/a.job
    torch.save({'u':facts['u'][i:i+1]+q.cpu(),'x':x.cpu(),'features':f.cpu(),'q':q.cpu()},job/'base.pt')
    torch.save(torch.cat(points),job/'proposals.pt');pd.DataFrame(rows).to_csv(job/'trace.csv',index=False)
    dump(job/'quality.json',{**current,'nominal_base_succeeded':crossed,'failed_base_retained':True})
    np.savez(job/'identity_embeddings.npz',arc_factual=m.arc0,arc_base=arc,facenet_factual=m.face0,facenet_base=face)
    Image.fromarray(rgb).save(a.output/'qualitative_examples'/f'{split}_{i}_Base.png')


def pack(a,c):
    split=a.job.split('_')[-1];items=[load(a.output/'jobs'/f'base_{split}_{i}'/'base.pt') for i in range(8)]
    torch.save({k:torch.cat([v[k] for v in items]) for k in ['u','x','features','q']},a.output/f'{split}_population.pt')
    dump(a.output/f'{split}_base_frozen.json',{'sha256':sha(a.output/f'{split}_population.pt'),'utc':stamp(),'bank_seen':False})


def attack(a,c):
    _,ri,restart=a.job.split('_');ri=int(ri);restart=int(restart);r=c['r_grid'][ri];s=c['seed']+1000+100*ri+restart
    torch.manual_seed(s);d=load(a.output/'training.pt');n=len(d['y']);settings=c['attack'];idx=torch.randperm(n,generator=torch.Generator().manual_seed(s))[:settings['mutable']]
    clf=classifier(a);x=d['x'][idx].to(a.device);y=d['y'][idx].to(a.device);clean=d['features'].to(a.device)
    delta=project_delta(x,torch.randn_like(x)*.01,r,n).requires_grad_(True);opt=torch.optim.Adam([delta],lr=settings['learning_rate'])
    qf=torch.cat([load(a.output/'calibration_population.pt')['features'],load(a.output/'references.pt')['features']]).to(a.device)
    trace=[]
    for step in range(settings['steps']):
        ii=torch.randperm(len(x))[:settings['batch_size']].to(a.device)
        w=clf.head.weight.detach().clone().requires_grad_(True);b=clf.head.bias.detach().clone().requires_grad_(True)
        loss=F.cross_entropy(F.linear(clean,w,b),d['y'].to(a.device))+.5*c['update']['l2']*w.square().sum()
        loss=loss+(F.cross_entropy(F.linear(clf.features(x[ii]+delta[ii]),w,b),y[ii],reduction='sum')-
                   F.cross_entropy(F.linear(clean[idx[ii.cpu()]],w,b),y[ii],reduction='sum'))/n
        gw,gb=torch.autograd.grad(loss,(w,b),create_graph=True)
        prob=F.linear(qf,w-settings['surrogate_lr']*gw,b-settings['surrogate_lr']*gb).softmax(1)[:,0]
        objective=(F.softplus(10*(.5-prob))/10).mean();grad,=torch.autograd.grad(objective,delta)
        assert torch.isfinite(grad).all();opt.zero_grad();delta.grad=-grad;opt.step()
        with torch.no_grad():delta.copy_(project_delta(x,delta,r,n))
        trace.append(dict(step=step,surrogate=float(objective.detach()),gradient_norm=float(grad.norm()),paired_bound=float(paired_distance(delta,n))))
    rows=[];job=a.output/'jobs'/a.job
    for fraction in settings['fractions']:
        change=project_delta(x,delta.detach()*fraction,r,n).cpu();bound=float(paired_distance(change,n));assert bound<=r+1e-6
        f=d['features'].clone();f[idx]=feature_batches(clf,d['x'][idx]+change,a.device)
        state,losses=head_update(clf,f,d['y'],c,a.device);path=a.output/'checkpoints'/f'{a.job}_{fraction}.pt';torch.save(state,path)
        dp=job/f'delta_{fraction}.pt';torch.save({'indices':idx,'delta':change},dp)
        rows.append({'candidate':path.stem,'checkpoint':str(path),'checkpoint_sha256':sha(path),'perturbation':str(dp),'perturbation_sha256':sha(dp),
                     'r':r,'paired_W_data_upper_bound':bound,'W_data_exact_computed':False,'fraction':fraction,'losses':losses})
    pd.DataFrame(trace).to_csv(job/'search.csv',index=False);dump(job/'models.json',rows)


def shifted_model(a,c):
    out=a.output;d=load(out/'training.pt');clf=classifier(a)
    if a.phase=='clean':scenario={'family':'brightness','strength':1.,'seed':0}
    elif a.phase=='calmodel':scenario=c['calibration_scenarios'][int(a.job.split('_')[-1])]
    else:
        _,family,si,rep=a.job.split('_');sev=c['future']['severity'][int(si)];scenario={'family':family,'severity':sev,'realization':int(rep),
            'strength':1-.2*sev if family=='brightness' else .04*sev if family=='noise' else .8*sev,'seed':c['future']['seed_base']+int(rep)}
        frozen=json.loads((out/'frozen.json').read_text());assert sha(out/'frozen.pt')==frozen['sha256']
    if a.phase=='clean':f=d['features']
    else:f=feature_batches(clf,shift(d['x'],scenario['family'],scenario['strength'],scenario['seed']),a.device)
    state,losses=head_update(clf,f,d['y'],c,a.device);path=out/'checkpoints'/f'{a.job}.pt';torch.save(state,path)
    record={'candidate':a.job,'checkpoint':str(path),'checkpoint_sha256':sha(path),'scenario':scenario,'losses':losses}
    if a.phase=='clean':record.update(paired_W_data_upper_bound=0.,r=0.)
    dump(out/'jobs'/a.job/'model.json',record)
    if a.phase=='future':
        rows=[]
        for name,item in load(out/'frozen.pt').items():
            p=state_scores(item['features'],[state]).sigmoid()[0]
            rows.append({'method':name,**scenario,'future_validity':float((p>=.5).double().mean()),'mean_hinge':float(torch.relu(.5-p).double().mean()),'checkpoint':str(path)})
        pd.DataFrame(rows).to_csv(out/'jobs'/a.job/'metrics.csv',index=False)


def records(out,r):
    rows=[json.loads((out/'jobs/clean/model.json').read_text())]
    for p in sorted((out/'jobs').glob('attack_*/models.json')):rows+=json.loads(p.read_text())
    return [v for v in rows if v['paired_W_data_upper_bound']<=r+1e-6]


class FeatureGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,bank):ctx.bank=bank;ctx.save_for_backward(u.detach());return bank.cached_features(u)
    @staticmethod
    def backward(ctx,upstream):
        u,=ctx.saved_tensors;grads=[]
        for i,(row,derivative) in enumerate(zip(u,upstream)):
            with torch.enable_grad():
                v=row[None].detach().requires_grad_(True);f=ctx.bank.clf.features(ctx.bank.clf.canonical(ctx.bank.decode(i,v)))
                grads.append(torch.autograd.grad((f*derivative).sum(),v)[0])
        return torch.cat(grads),None


class StyleBank:
    def __init__(self,a,c,split,r):
        self.device=a.device;self.g=Generator(a.device);self.editor=StyleEditor(self.g);self.clf=classifier(a)
        self.facts=load(a.output/f'{split}_factuals.pt');self.w=self.facts['w'].to(a.device)
        self.indices=load(a.output/'checkpoints/controls.pt')['style4']['indices'].to(a.device)
        self.std=load(a.output/'checkpoints/statistics.pt')['style_std'][self.indices.cpu()].to(a.device)
        with torch.no_grad():self.styles=self.editor.encode(self.w)
        self.states=[load(Path(v['checkpoint'])) for v in records(a.output,r)];self.count=len(self.states);self.cache={}
    def decode(self,i,u):
        styles=self.styles[i:i+1].clone();styles[:,self.indices]=u*self.std
        mask=torch.ones(styles.shape[1],dtype=torch.bool,device=self.device);mask[self.indices]=False
        assert torch.equal(styles[:,mask],self.styles[i:i+1,mask])
        return self.editor.decode(self.w[i:i+1],styles)
    def cached_features(self,u):
        key=hashlib.sha256(u.detach().cpu().numpy().tobytes()).hexdigest()
        if key not in self.cache:
            xs=[];fs=[]
            with torch.no_grad():
                for i,row in enumerate(u):
                    x=self.clf.canonical(self.decode(i,row[None].to(self.device)));xs.append(x.cpu());fs.append(self.clf.features(x).cpu())
            self.cache[key]={'points':u.detach().cpu().clone(),'x':torch.cat(xs),'features':torch.cat(fs)}
        return self.cache[key]['features'].to(self.device)
    def logits(self,u,start=0,end=None):
        f=FeatureGradient.apply(u,self) if torch.is_grad_enabled() and u.requires_grad else self.cached_features(u)
        return state_scores(f,self.states[start:end])
    def risk(self,u):
        with torch.no_grad():return float(population_risk(self.logits(u).sigmoid()))


def optimize(a,c):
    _,split,ri,bi=a.job.split('_');ri=int(ri);bi=int(bi);out=a.output;r=c['r_grid'][ri];beta=c['beta_grid'][bi]
    pop=load(out/f'{split}_population.pt');z=pop['u'].to(a.device);target=load(out/'references.pt')['u'].to(a.device)
    plan=exact_w2(z,target);D=plan.distance;perm=torch.empty(8,dtype=torch.long);perm[plan.source_index]=plan.target_index;target=target[perm.to(a.device)]
    bank=StyleBank(a,c,split,r);factual=bank.facts['u'].to(a.device);trace=[]
    for lam in c['geodesic_candidates']:bank.cached_features((1-lam)*z+lam*target)
    for name,initial in [('base',z),('near_base',.95*z+.05*target)]:
        _,history=solve_raw(bank,z,target,initial,D,beta*D,factual-16,factual+16,c['optimizer'],
            on_step=lambda row:print(a.job,name,row['step'],row['true_bank_hinge'],row['slack'],flush=True) if row['step']%4==0 else None)
        trace += [dict(start=name,**h) for h in history]
    candidates=list(bank.cache.values());rows=[]
    for i,v in enumerate(candidates):
        q=v['points'];length=exact_w2(z,q).distance+exact_w2(q,target).distance
        rows.append(dict(candidate=i,length=length,risk=float(population_risk(state_scores(v['features'],bank.states).sigmoid())),
                         pixel_cost=float((v['x']-bank.facts['x']).flatten(1).norm(dim=1).mean()),
                         box_feasible=bool(((q-bank.facts['u']).abs()<=16+1e-6).all())))
    job=out/'jobs'/a.job;torch.save(candidates,job/'candidates.pt');pd.DataFrame(rows).to_csv(job/'candidates.csv',index=False)
    pd.DataFrame(trace).to_csv(job/'trace.csv',index=False);dump(job/'meta.json',{'r':r,'beta':beta,'D':D,'bank_ids':[v['candidate'] for v in records(out,r)]})


def pool(a,c):
    _,split,ri=a.job.split('_');ri=int(ri);out=a.output;items=[];rows=[];seen=set()
    for bi in range(4):
        job=out/'jobs'/f'opt_{split}_{ri}_{bi}';candidates=load(job/'candidates.pt');table=pd.read_csv(job/'candidates.csv');D=json.loads((job/'meta.json').read_text())['D']
        for row in table.to_dict('records'):
            v=candidates[row['candidate']];key=hashlib.sha256(v['points'].numpy().tobytes()).hexdigest()
            if key in seen:continue
            seen.add(key);row['candidate']=len(items);items.append(v);rows.append(row)
    facts=load(out/f'{split}_factuals.pt');m=GenkiMeasurement(a,c,facts['w'][:1].to(a.device))
    idx=load(out/'checkpoints/controls.pt')['style4']['indices'].to(a.device);std=load(out/'checkpoints/statistics.pt')['style_std'].to(a.device)
    quality={};qualityrows=[];selected={};geometries=[]
    def passes(index):
        if index in quality:return quality[index]
        v=items[index];ok=True
        for i in range(8):
            with torch.no_grad():
                m.w=facts['w'][i:i+1].to(a.device);m.s=m.editor.encode(m.w);m.native=m.g.decode(m.w);m.x=m.clf.canonical(m.native)
                m.p=float(m.clf(m.native).softmax(1)[0,0]);m.arc0,m.face0,_,_=m.identities(m.native)
            row,_,_,_=m.measure((v['points'][i:i+1]-facts['u'][i:i+1]).to(a.device),idx,std)
            hit=passes_quality(row,c);qualityrows.append(dict(candidate=index,factual_index=i,quality_pass=hit,**row))
            if not hit:ok=False;break
        quality[index]=ok;return ok
    for beta in c['beta_grid']:
        eligible=sorted([r for r in rows if r['box_feasible'] and r['length']<=beta*D+1e-6*max(1,D)],key=lambda r:(r['risk'],r['pixel_cost'],r['candidate']))
        best=next((r for r in eligible if passes(r['candidate'])),None)
        assert best is not None,'Base should remain a geometry- and preservation-feasible candidate'
        selected[str(beta)]=items[best['candidate']]
        geometries.append({'r':c['r_grid'][ri],'beta_geo':beta,'D':D,'B_geo':beta*D,'total_length':best['length'],
                           'slack':beta*D-best['length'],'Phi_hat_r':best['risk'],'pixel_L2_cost':best['pixel_cost'],
                           'candidate':best['candidate'],'preservation_feasible':True,'geometry_feasible':True})
        print(a.job,beta,best,flush=True)
    job=out/'jobs'/a.job;torch.save(selected,job/'selected.pt');pd.DataFrame(geometries).to_csv(job/'geometry.csv',index=False)
    pd.DataFrame(qualityrows).to_csv(job/'quality_screen.csv',index=False);pd.DataFrame(rows).to_csv(job/'pooled_candidates.csv',index=False)


def select(a,c):
    rows=[];out=a.output;states=[load(out/'checkpoints'/f'calmodel_{i}.pt') for i in range(4)]
    for ri,r in enumerate(c['r_grid']):
        items=load(out/'jobs'/f'pool_calibration_{ri}'/'selected.pt');geometry=pd.read_csv(out/'jobs'/f'pool_calibration_{ri}'/'geometry.csv')
        for beta in c['beta_grid']:
            p=state_scores(items[str(beta)]['features'],states).sigmoid();row=geometry[geometry.beta_geo==beta].iloc[0].to_dict()
            rows.append({**row,'r_index':ri,'calibration_max_mean_hinge':float(population_risk(p)),
                         'calibration_mean_validity':float((p>=.5).double().mean())})
    pd.DataFrame(rows).to_csv(out/'calibration_results.csv',index=False)
    best=min(rows,key=lambda r:(round(r['calibration_max_mean_hinge'],6),r['pixel_L2_cost'],r['r'],r['beta_geo']))
    dump(out/'selected_configuration.json',{'r_star':best['r'],'r_index':best['r_index'],'beta_geo_star':best['beta_geo'],'utc':stamp(),'future_outcomes_used':False})


def freeze(a,c):
    out=a.output;spec=json.loads((out/'selected_configuration.json').read_text());ri=spec['r_index'];facts=load(out/'test_factuals.pt');basepop=load(out/'test_population.pt')
    candidates=load(out/'jobs'/f'pool_test_{ri}'/'selected.pt');methods={'Base':dict(points=basepop['u'],x=basepop['x'],features=basepop['features'])}
    methods.update({f'GRACE_beta{b:.2f}':candidates[str(b)] for b in c['beta_grid']})
    m=GenkiMeasurement(a,c,facts['w'][:1].to(a.device));idx=load(out/'checkpoints/controls.pt')['style4']['indices'].to(a.device)
    std=load(out/'checkpoints/statistics.pt')['style_std'].to(a.device);rows=[];emb={}
    states=[load(Path(v['checkpoint'])) for v in records(out,spec['r_star'])]
    for i in range(8):
        with torch.no_grad():
            m.w=facts['w'][i:i+1].to(a.device);m.s=m.editor.encode(m.w);m.native=m.g.decode(m.w);m.x=m.clf.canonical(m.native)
            m.p=float(m.clf(m.native).softmax(1)[0,0]);m.arc0,m.face0,_,_=m.identities(m.native)
        emb[f'arc_factual_{i}']=m.arc0;emb[f'facenet_factual_{i}']=m.face0
        for name,v in methods.items():
            row,rgb,arc,face=m.measure((v['points'][i:i+1]-facts['u'][i:i+1]).to(a.device),idx,std)
            assert passes_quality(row,c),row
            rows.append(dict(method=name,factual_index=i,quality_pass=True,**row));emb[f'arc_{name}_{i}']=arc;emb[f'facenet_{name}_{i}']=face
            Image.fromarray(rgb).save(out/'qualitative_examples'/f'test_{i}_{name}.png')
    np.savez(out/'identity_embeddings.npz',**emb);pd.DataFrame(rows).to_csv(out/'image_quality.csv',index=False)
    pd.DataFrame([{'method':name,'Phi_hat_r':float(population_risk(state_scores(v['features'],states).sigmoid()))} for name,v in methods.items()]).to_csv(out/'finite_bank_risk.csv',index=False)
    torch.save(methods,out/'frozen.pt');pd.DataFrame(records(out,max(c['r_grid']))).to_csv(out/'attack_bank_manifest.csv',index=False)
    dump(out/'frozen.json',{'sha256':sha(out/'frozen.pt'),'utc':stamp(),'selected_sha256':sha(out/'selected_configuration.json'),
                          'future_outcomes_seen':False,'factual_ids':facts['ids']})
    clock=json.loads((out/'construction_clock.json').read_text());clock['end_unix']=time.time();clock['wall_seconds']=clock['end_unix']-clock['start_unix'];dump(out/'construction_clock.json',clock)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','base','pack','attack','clean','calmodel','optimize','pool','select','freeze','future'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20260918);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    path=a.output/'resolved_config.json';c=json.loads((path if path.exists() else ROOT/'experiments/configs/genki_grace_identity_v1.json').read_text())
    torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction'])
    job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
    if (job/'complete.json').exists():return
    start=time.time();meta={'phase':a.phase,'job':a.job,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta)
    if a.phase in ['clean','calmodel','future']:shifted_model(a,c)
    else:globals()[a.phase](a,c)
    torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start,'peak_gpu_memory_GB':torch.cuda.max_memory_allocated()/1e9})


if __name__=='__main__':main()
