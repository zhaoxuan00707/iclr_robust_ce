"""Stronger scalar and independently movable local-style calibration controls."""
import json,os,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .ffhq_smile_editor_v1 import evaluate as scalar_evaluate,identity_models
from .galaxy_grace_v1 import load,dump,sha,stamp
from ..ffhq.assets import Generator,SmileClassifier
from ..ffhq.style_editor import StyleEditor,region_mask
from ..ffhq.supervised_smile import SupervisedSmile
ROOT=Path(__file__).resolve().parents[4];CONFIG=ROOT/'experiments/configs/ffhq_smile_editor_v2.json'


def passes_quality(row,c):
    return bool(row['arcface_similarity']>=c['arcface_min'] and row['facenet_similarity']>=c['facenet_min'] and row['LPIPS']<=c['lpips_max'] and row['outside_rms']<=c['outside_rms_max'])


def scalar_table(out,c):
    old=Path(c['previous_screen']);d=pd.read_csv(old/'all_calibration_metrics.csv');d=d[d.editor.str.startswith('style')].copy();d['image_path']=[str(old/'qualitative_examples'/f'{int(r.calibration_index)}_{r.editor}_{int(r.alpha_index)}.png') for r in d.itertuples()];d['source']='previous_scalar'
    tables=[d]
    for p in sorted((out/'jobs').glob('scalar_*/metrics.csv')):
        d=pd.read_csv(p);d['image_path']=[str(out/'qualitative_examples'/f'{int(r.calibration_index)}_{r.editor}_{int(r.alpha_index)}.png') for r in d.itertuples()];d['source']='extended_scalar';tables.append(d)
    return pd.concat(tables,ignore_index=True)


def gate_table(table,c):
    table=table.copy();table['quality_pass']=[passes_quality(row,c) for row in table.to_dict('records')];table['smile_pass']=(table.smile_score>=c['smile_score_min'])&(table.smile_gain>=c['smile_score_gain_min']);table['all_gates_pass']=table.quality_pass&table.smile_pass
    counts=table.groupby(['editor','calibration_index']).all_gates_pass.any().groupby('editor').sum()
    return table,counts


def initialize(a,c):
    out=a.output;old=Path(c['previous_screen']);parent=Path(c['parent_run'])
    for name in ['jobs','logs','checkpoints','qualitative_examples','figures']:(out/name).mkdir(exist_ok=True)
    dump(out/'resolved_config.json',c)
    controls={k:v for k,v in load(old/'checkpoints/controls.pt').items() if k.startswith('style')};torch.save(controls,out/'checkpoints/controls.pt')
    for name in ['statistics.pt','facenet_vggface2.pt']:(out/'checkpoints'/name).symlink_to(old/'checkpoints'/name)
    shutil.copyfile('/tmp/ffhq_lennon_state.pt',out/'checkpoints/supervised_smile.pt');shutil.copyfile('/tmp/ffhq_smile_model_readme.md',out/'supervised_smile_model_card.md')
    shutil.copyfile(old/'factual_ids.npy',out/'factual_ids.npy')
    inputs=[old/'checkpoints/controls.pt',old/'all_calibration_metrics.csv',parent/'calibration_factuals.pt',parent/'calibration_factual_ids.npy']
    dump(out/'input_manifest.json',{str(p):sha(p) for p in inputs});dump(out/'protocol_frozen.json',{'utc':stamp(),'config_sha256':sha(out/'resolved_config.json'),'controls_sha256':sha(out/'checkpoints/controls.pt'),'final_ffhq_test_accessed':False,'future_models_accessed':False})
    model=SupervisedSmile(out/'checkpoints/supervised_smile.pt');validation=Path('/tmp/ffhq_celeba_smile_audit');labels=json.loads((validation/'labels.json').read_text());scores=model.score([validation/f'{r["index"]}.jpg' for r in labels]);table=pd.DataFrame([{**r,**v} for r,v in zip(labels,scores)]);table.to_csv(out/'supervised_smile_source_validation.csv',index=False)
    dump(out/'supervised_smile_manifest.json',{**c['supervised_smile'],'converted_state_sha256':sha(out/'checkpoints/supervised_smile.pt'),'original_checkpoint_sha256':sha('/tmp/ffhq_lennon_smile.pt'),'conversion':'weights_only load with audited module/standard nn allowlist, then state_dict; inference keys loaded strictly','source_sample_accuracy':float((table.smiling==table.supervised_smile_label).mean()),'source_positive_count':int(table.smiling.sum()),'source_extra_class_predictions':int((table.supervised_raw_class>1).sum()),'small_nonindependent_source_audit_not_generalization_estimate':True,'validation_dataset_revision':'e9092e921951927418a9e8f87822020206b292ce'})
    factuals=[old/'qualitative_examples'/f'{i}_factual.png' for i in range(8)];pd.DataFrame([{'calibration_index':i,**v} for i,v in enumerate(model.score(factuals))]).to_csv(out/'supervised_factual_scores.csv',index=False)


def scalar(a,c):scalar_evaluate(a,c)


def scalar_gate(a,c):
    table,counts=gate_table(scalar_table(a.output,c),c);table.to_csv(a.output/'scalar_candidates.csv',index=False)
    dump(a.output/'scalar_gate.json',{'passing_factuals':{k:int(v) for k,v in counts.items()},'passed':bool((counts>=c['required_passing_factuals']).any()),'thresholds_unchanged':True})


class Measurement:
    def __init__(self,a,c,w):
        import lpips
        self.a=a;self.c=c;self.g=Generator(a.device);self.editor=StyleEditor(self.g);self.clf=SmileClassifier(a.device);self.arc,self.mtcnn,self.facenet=identity_models(a);torch.hub.set_dir('/tmp/ffhq_torch_hub');self.metric=lpips.LPIPS(net='squeeze').to(a.device).eval();self.w=w;self.outside=1-region_mask(224,a.device)
        with torch.no_grad():
            self.s=self.editor.encode(w);self.native=self.g.decode(w);self.x=self.clf.canonical(self.native);self.p=float(self.clf(self.native).softmax(1)[0,0]);self.arc0,self.face0,_,_=self.identities(self.native)
    def identities(self,native):
        import cv2
        rgb=((native[0].clamp(-1,1).cpu().permute(1,2,0).numpy()+1)*127.5).round().clip(0,255).astype(np.uint8);faces=self.arc.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));face=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None;crop=self.mtcnn(Image.fromarray(rgb));f=self.facenet(crop[None].to(self.a.device))[0].cpu().numpy() if crop is not None else None
        return face.normed_embedding if face is not None else None,f,rgb,len(faces)
    def decode(self,q,indices,std):
        styles=self.s.clone();styles[:,indices]=self.s[:,indices]+q*std[indices];mask=torch.ones(styles.shape[1],dtype=torch.bool,device=q.device);mask[indices]=False;assert torch.equal(styles[:,mask],self.s[:,mask]);return self.editor.decode(self.w,styles)
    @torch.no_grad()
    def measure(self,q,indices,std):
        n=self.decode(q,indices,std);x=self.clf.canonical(n);p=float(self.clf(n).softmax(1)[0,0]);arc,face,rgb,count=self.identities(n)
        row={'smile_score':p,'factual_smile_score':self.p,'smile_gain':p-self.p,'arcface_similarity':float(self.arc0@arc) if self.arc0 is not None and arc is not None else np.nan,'facenet_similarity':float(self.face0@face) if self.face0 is not None and face is not None else np.nan,'LPIPS':float(self.metric(self.native.clamp(-1,1),n.clamp(-1,1))),'outside_rms':float((((x-self.x).square()*self.outside).sum()/(3*self.outside.sum())).sqrt()),'pixel_L2_224':float((x-self.x).flatten(1).norm(dim=1)),'edited_arc_faces':count}
        return row,rgb,arc,face


def vector(a,c):
    _,count,i=a.job.split('_');count=int(count);i=int(i);settings=c['independent'];facts=load(Path(c['parent_run'])/'calibration_factuals.pt');w=facts['w'][i:i+1].to(a.device);measurement=Measurement(a,c,w);control=load(a.output/'checkpoints/controls.pt')[f'style{count}'];indices=control['indices'].to(a.device);std=load(a.output/'checkpoints/statistics.pt')['style_std'].to(a.device)
    scalar=pd.read_csv(a.output/'scalar_candidates.csv');subset=scalar[(scalar.editor==f'style{count}')&(scalar.calibration_index==i)&scalar.quality_pass];best=subset.sort_values(['smile_score','alpha'],ascending=[False,True]).iloc[0];q=(float(best.alpha)*control['standardized_direction'][control['indices']]).to(a.device)[None];current,rgb,arc,face=measurement.measure(q,indices,std);assert passes_quality(current,c)
    job=a.output/'jobs'/a.job;rows=[];points=[];embeddings={};step_size=settings['initial_step_size']
    def save(step,q,row,rgb,arc,face,accepted):
        path=a.output/'qualitative_examples'/f'{a.job}_{step}.png';Image.fromarray(rgb).save(path);rows.append({**row,'calibration_index':i,'editor':f'independent{count}','step':step,'step_size':step_size,'accepted':bool(accepted),'mutable_channels':count,'image_path':str(path),'source':'independent'});points.append(q.cpu())
        if arc is not None:embeddings[f'arc_{step}']=arc
        if face is not None:embeddings[f'facenet_{step}']=face
    save(-1,q,current,rgb,arc,face,True)
    if measurement.arc0 is not None:embeddings['arc_factual']=measurement.arc0
    if measurement.face0 is not None:embeddings['facenet_factual']=measurement.face0
    for step in range(settings['steps']):
        variable=q.detach().requires_grad_(True);n=measurement.decode(variable,indices,std);score=measurement.clf(n).softmax(1)[0,0];grad=torch.autograd.grad(score,variable)[0];assert torch.isfinite(grad).all();trial=(q+step_size*grad/(grad.norm()+1e-12)).clamp(-settings['coordinate_bound'],settings['coordinate_bound']).detach();row,rgb,arc,face=measurement.measure(trial,indices,std)
        accept=passes_quality(row,c) and row['smile_score']>=current['smile_score']-1e-7
        save(step,trial,row,rgb,arc,face,accept)
        if accept:q=trial;current=row;step_size=min(settings['maximum_step_size'],step_size*settings['step_growth'])
        else:step_size=max(settings['minimum_step_size'],step_size*settings['rejection_factor'])
        if step%10==0:print(a.job,step,current['smile_score'],step_size,flush=True)
    pd.DataFrame(rows).to_csv(job/'metrics.csv',index=False);torch.save(torch.cat(points),job/'coordinates.pt');np.savez(job/'embeddings.npz',**embeddings)


def supervised(a,c):
    model=SupervisedSmile(a.output/'checkpoints/supervised_smile.pt');tables=[pd.read_csv(a.output/'scalar_candidates.csv')]
    for p in sorted((a.output/'jobs').glob('vector_*/metrics.csv')):tables.append(pd.read_csv(p))
    d=pd.concat(tables,ignore_index=True);scores=[]
    for start in range(0,len(d),32):scores.extend(model.score(d.image_path.iloc[start:start+32].tolist()))
    d=pd.concat([d.reset_index(drop=True),pd.DataFrame(scores)],axis=1);d.to_csv(a.output/'all_candidates_with_supervised_smile.csv',index=False)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','scalar','scalar_gate','vector','supervised']);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.14);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.manual_seed(20260917)
    path=a.output/'resolved_config.json';c=json.loads((path if path.exists() else CONFIG).read_text());job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
    if (job/'complete.json').exists():return
    start=time.time();meta={'job':a.job,'phase':a.phase,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':stamp()};dump(job/'started.json',meta);globals()[a.phase](a,c);torch.cuda.synchronize();dump(job/'complete.json',{**meta,'end_utc':stamp(),'wall_seconds':time.time()-start,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9})

if __name__=='__main__':main()
