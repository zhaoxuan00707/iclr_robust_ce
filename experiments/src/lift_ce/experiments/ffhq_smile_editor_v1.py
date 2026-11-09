"""Predeclared training/calibration screen for local smile controls."""
import json,os,time,shutil
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from ..ffhq.assets import Generator,SmileClassifier
from ..ffhq.style_editor import StyleEditor,region_mask,apply_control
from .galaxy_grace_v1 import load,sha,dump,stamp
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/ffhq_smile_editor_v1.json'


def initialize(a,c):
    out=a.output;parent=Path(c['parent_run'])
    for name in ['jobs','logs','checkpoints','figures','qualitative_examples']:(out/name).mkdir(exist_ok=True)
    dump(out/'resolved_config.json',c)
    names=['training.pt','calibration_factuals.pt','calibration_factual_ids.npy','calibration_population.pt']
    dump(out/'input_manifest.json',{str(parent/name):sha(parent/name) for name in names})
    dump(out/'protocol_frozen.json',{'utc':stamp(),'config_sha256':sha(out/'resolved_config.json'),'final_test_accessed':False,'future_models_accessed':False})
    shutil.copyfile(parent/'calibration_factual_ids.npy',out/'factual_ids.npy')
    shutil.copyfile('/tmp/ffhq_facenet_vggface2.pt',out/'checkpoints/facenet_vggface2.pt')
    dump(out/'facenet_manifest.json',{'sha256':sha(out/'checkpoints/facenet_vggface2.pt'),'source':'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt','package':'facenet-pytorch2.6.0','embedding_dim':512,'input':'MTCNN160x160, (RGB-127.5)/128','role':'independent identity evaluation; not differentiated or used in channel discovery'})
    g=Generator(a.device);editor=StyleEditor(g);train=load(parent/'training.pt')
    with torch.no_grad():styles=editor.encode(train['w'].to(a.device)).cpu()
    std=styles.std(0).clamp_min(1e-6)
    probe=LogisticRegression(C=1,max_iter=2000,random_state=c['seed']).fit(train['w'].numpy(),(train['y'].numpy()==0).astype(int));direction=torch.from_numpy(probe.coef_[0]).float();direction/=direction.norm()
    torch.save({'style_std':std,'w_direction':direction},out/'checkpoints/statistics.pt')
    dump(out/'style_architecture.json',{'layer_names':editor.names,'sizes':editor.sizes,'offsets':editor.offsets,'total_style_dimensions':sum(editor.sizes),'input_transform':'original factual W, fixed','mutable_layers':c['eligible_layers']})
    # Style-space zero edit must exactly match original synthesis; directional
    # finite differences verify the post-affine gradient routing.
    w=train['w'][:1].to(a.device)
    with torch.no_grad():s=editor.encode(w);original=g.decode(w);reconstruction=editor.decode(w,s)
    error=float((original-reconstruction).abs().max());assert error<1e-6
    clf=SmileClassifier(a.device);s=s.detach().requires_grad_(True);x=clf.canonical(editor.decode(w,s));score=clf.logits_features(clf.features(x)).softmax(1)[0,0];grad=torch.autograd.grad(score,s)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    direction=grad/grad.norm();epsilon=.01
    with torch.no_grad():
        def value(z):return clf(editor.decode(w,z)).softmax(1)[0,0]
        fd=float((value(s+epsilon*direction)-value(s-epsilon*direction))/(2*epsilon))
    ad=float((grad*direction).sum());assert abs(fd-ad)<.1*abs(ad)+1e-3
    dump(out/'style_smoke.json',{'zero_edit_max_error':error,'autograd_directional_derivative':ad,'finite_difference_derivative':fd,'passed':True})


def discover(a,c):
    shard=int(a.job.split('_')[-1]);train=load(Path(c['parent_run'])/'training.pt');std=load(a.output/'checkpoints/statistics.pt')['style_std'].to(a.device);g=Generator(a.device);editor=StyleEditor(g);clf=SmileClassifier(a.device);rows=[];outside=1-region_mask(224,a.device)
    for i in range(shard,c['discovery_training_count'],8):
        w=train['w'][i:i+1].to(a.device)
        with torch.no_grad():s0=editor.encode(w)
        q=torch.zeros_like(s0,requires_grad=True)
        def output(q):return clf.canonical(editor.decode(w,s0+q*std))
        x=output(q);p=clf.logits_features(clf.features(x)).softmax(1)[0,0];grad=torch.autograd.grad(p,q)[0]
        energies=[]
        for probe in range(c['outside_jacobian_probes']):
            x=output(q);rng=torch.Generator(device=a.device).manual_seed(c['seed']+100*i+probe);noise=(torch.randint(0,2,x.shape,device=a.device,generator=rng)*2-1).float()*outside
            derivative=torch.autograd.grad((x*noise).sum()/torch.sqrt(outside.sum()*3),q)[0];energies.append(derivative.square())
        rows.append({'index':i,'score_gradient':grad.cpu(),'outside_energy':torch.stack(energies).mean(0).cpu()})
        print('discovery',i,flush=True)
    torch.save(rows,a.output/'jobs'/a.job/'discovery.pt')


def controls(a,c):
    rows=sum([load(p) for p in sorted((a.output/'jobs').glob('discover_*/discovery.pt'))],[]);assert len(rows)==c['discovery_training_count']
    grads=torch.cat([r['score_gradient'] for r in rows]);leakage=torch.cat([r['outside_energy'] for r in rows]);mean=grads.mean(0);agreement=(grads.sign()==mean.sign()).float().mean(0)
    arch=json.loads((a.output/'style_architecture.json').read_text());eligible=torch.zeros(len(mean),dtype=torch.bool)
    for j in c['eligible_layers']:eligible[arch['offsets'][j]:arch['offsets'][j+1]]=True
    eligible &= agreement>=c['minimum_gradient_sign_agreement']
    score=mean.abs()/(leakage.mean(0).sqrt()+1e-5);score[~eligible]=-float('inf');order=score.argsort(descending=True)
    assert eligible.sum()>=max(c['style_counts'])
    saved={};stats=load(a.output/'checkpoints/statistics.pt');table=[]
    for count in c['style_counts']:
        indices=order[:count];d=torch.zeros_like(mean);d[indices]=mean[indices];d/=d.norm();saved[f'style{count}']={'direction':d*stats['style_std'],'standardized_direction':d,'indices':indices,'mutable_channels':count}
    saved['w_linear']={'direction':stats['w_direction'],'mutable_channels':512}
    for i in order[eligible[order]].tolist():
        layer=next(j for j in range(len(arch['sizes'])) if arch['offsets'][j]<=i<arch['offsets'][j+1]);table.append({'index':i,'layer':arch['layer_names'][layer],'channel':i-arch['offsets'][layer],'score':float(score[i]),'mean_smile_gradient':float(mean[i]),'outside_jacobian_rms_estimate':float(leakage[:,i].mean().sqrt()),'sign_agreement':float(agreement[i])})
    pd.DataFrame(table).to_csv(a.output/'channel_ranking.csv',index=False);torch.save(saved,a.output/'checkpoints/controls.pt');dump(a.output/'controls_frozen.json',{'utc':stamp(),'sha256':sha(a.output/'checkpoints/controls.pt'),'training_indices':sorted(r['index'] for r in rows),'calibration_used':False,'test_used':False})


def identity_models(a):
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    from facenet_pytorch import MTCNN,InceptionResnetV1
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    arc=FaceAnalysis(name='buffalo_l',root='/tmp/ffhq_identity_assets',allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'],sess_options=opts);arc.prepare(ctx_id=-1,det_size=(320,320))
    mtcnn=MTCNN(image_size=160,margin=0,keep_all=False,select_largest=True,post_process=True,device=a.device)
    net=InceptionResnetV1(classify=True,num_classes=8631).to(a.device).eval().requires_grad_(False);net.load_state_dict(torch.load(a.output/'checkpoints/facenet_vggface2.pt',map_location='cpu',weights_only=True),strict=True);net.classify=False
    return arc,mtcnn,net


def evaluate(a,c):
    import lpips,cv2
    i=int(a.job.split('_')[-1]);facts=load(Path(c['parent_run'])/'calibration_factuals.pt');g=Generator(a.device);editor=StyleEditor(g);clf=SmileClassifier(a.device);controls=load(a.output/'checkpoints/controls.pt');arc,mtcnn,facenet=identity_models(a)
    torch.hub.set_dir('/tmp/ffhq_torch_hub');metric=lpips.LPIPS(net='squeeze').to(a.device).eval();w=facts['w'][i:i+1].to(a.device);outside=1-region_mask(224,a.device)
    def identities(native):
        rgb=((native[0].clamp(-1,1).cpu().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8);faces=arc.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));face=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None
        crop=mtcnn(Image.fromarray(rgb))
        f=facenet(crop[None].to(a.device))[0].cpu().numpy() if crop is not None else None
        return face.normed_embedding if face is not None else None,f,rgb,len(faces)
    rows=[];embeddings={};images=[]
    with torch.no_grad():
        s=editor.encode(w);native0=g.decode(w);x0=clf.canonical(native0);p0=float(clf(native0).softmax(1)[0,0]);arc0,face0,rgb0,n0=identities(native0)
        Image.fromarray(rgb0).save(a.output/'qualitative_examples'/f'{i}_factual.png')
        if arc0 is not None:embeddings['arc_factual']=arc0
        if face0 is not None:embeddings['facenet_factual']=face0
        for name,control in controls.items():
            direction=control['direction'].to(a.device)
            for j,alpha in enumerate(c['alpha_grid']):
                if name=='w_linear':native=g.decode(w+alpha*direction)
                else:
                    modified=apply_control(s,direction,alpha);indices=control['indices'].to(a.device);immutable=torch.ones(s.shape[1],dtype=torch.bool,device=a.device);immutable[indices]=False;assert torch.equal(modified[:,immutable],s[:,immutable]);native=editor.decode(w,modified)
                x=clf.canonical(native);prob=float(clf(native).softmax(1)[0,0]);arc1,face1,rgb,n1=identities(native)
                key=f'{name}_{j}';Image.fromarray(rgb).save(a.output/'qualitative_examples'/f'{i}_{key}.png')
                arc_sim=float(arc0@arc1) if arc0 is not None and arc1 is not None else np.nan;face_sim=float(face0@face1) if face0 is not None and face1 is not None else np.nan
                if arc1 is not None:embeddings['arc_'+key]=arc1
                if face1 is not None:embeddings['facenet_'+key]=face1
                lp=float(metric(native0.clamp(-1,1),native.clamp(-1,1)));rms=float((((x-x0).square()*outside).sum()/(3*outside.sum())).sqrt())
                rows.append({'calibration_index':i,'editor':name,'alpha':alpha,'alpha_index':j,'smile_score':prob,'factual_smile_score':p0,'smile_gain':prob-p0,'arcface_similarity':arc_sim,'facenet_similarity':face_sim,'LPIPS':lp,'outside_rms':rms,'pixel_L2_224':float((x-x0).flatten(1).norm(dim=1)),'factual_arc_faces':n0,'edited_arc_faces':n1,'mutable_channels':control['mutable_channels']})
                images.append({'key':key,'x':x.cpu(),'smile_score':prob})
    job=a.output/'jobs'/a.job;pd.DataFrame(rows).to_csv(job/'metrics.csv',index=False);np.savez(job/'embeddings.npz',**embeddings);torch.save(images,job/'canonical_images.pt')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','discover','controls','evaluate']);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.14);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.manual_seed(20260917)
    config=a.output/'resolved_config.json';c=json.loads((config if config.exists() else CONFIG).read_text());job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
    if (job/'complete.json').exists():return
    start=time.time();meta={'job':a.job,'phase':a.phase,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta);globals()[a.phase](a,c);torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9})

if __name__=='__main__':main()
