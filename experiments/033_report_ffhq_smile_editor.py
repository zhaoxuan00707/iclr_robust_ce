"""Audit the frozen calibration gates and report the local editor screen."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.experiments.galaxy_grace_v1 import sha,load,dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);out=parser.parse_args().output;c=json.loads((out/'resolved_config.json').read_text());controls=load(out/'checkpoints/controls.pt')
    for path,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==digest
    manifest=json.loads((out/'controls_frozen.json').read_text());assert sha(out/'checkpoints/controls.pt')==manifest['sha256'];assert manifest['training_indices']==list(range(32))
    jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')];assert len(jobs)==18 and {int(j['physical_gpu']) for j in jobs}==set(range(8));assert all(j['start_utc']>manifest['utc'] for j in jobs if j['phase']=='evaluate')
    d=pd.concat([pd.read_csv(p) for p in sorted((out/'jobs').glob('evaluate_*/metrics.csv'))],ignore_index=True);assert len(d)==8*4*6
    for count in c['style_counts']:
        ctrl=controls[f'style{count}'];assert int((ctrl['direction']!=0).sum())==count
    for i in range(8):
        emb=np.load(out/'jobs'/f'evaluate_{i}'/'embeddings.npz')
        for row in d[d.calibration_index==i].itertuples():
            for field,prefix in [('arcface_similarity','arc'),('facenet_similarity','facenet')]:
                left=f'{prefix}_factual';right=f'{prefix}_{row.editor}_{row.alpha_index}'
                if left in emb and right in emb:assert abs(float(emb[left]@emb[right])-getattr(row,field))<1e-6
                else:assert pd.isna(getattr(row,field))
    quality=(d.arcface_similarity>=c['arcface_min'])&(d.facenet_similarity>=c['facenet_min'])&(d.LPIPS<=c['lpips_max'])&(d.outside_rms<=c['outside_rms_max'])
    d['quality_pass']=quality;d['smile_pass']=(d.smile_score>=c['smile_score_min'])&(d.smile_gain>=c['smile_score_gain_min']);d['all_gates_pass']=d.quality_pass&d.smile_pass
    zero=d[d.alpha==0];assert zero.LPIPS.abs().max()<1e-6 and zero.outside_rms.max()<1e-6 and zero.smile_gain.abs().max()<1e-5
    assert zero.arcface_similarity.min()>.999 and zero.facenet_similarity.min()>.999
    d.to_csv(out/'all_calibration_metrics.csv',index=False)
    selected=[]
    for (name,i),rows in d.groupby(['editor','calibration_index']):
        hits=rows[rows.all_gates_pass]
        if len(hits):best=hits.sort_values('alpha').iloc[0]
        else:
            safe=rows[rows.quality_pass];best=(safe if len(safe) else rows[rows.alpha==0]).sort_values(['smile_score','alpha'],ascending=[False,True]).iloc[0]
        selected.append(best.to_dict())
    selected=pd.DataFrame(selected);selected.to_csv(out/'selected_examples.csv',index=False)
    summaries=[]
    for name,rows in selected.groupby('editor'):
        summaries.append({'editor':name,'mutable_channels':int(rows.mutable_channels.iloc[0]),'passing_factuals':int(rows.all_gates_pass.sum()),'total_factuals':8,'numerical_gate_passed':bool(rows.all_gates_pass.sum()>=c['required_passing_factuals']),'mean_smile_gain':rows.smile_gain.mean(),'mean_smile_score':rows.smile_score.mean(),'mean_arcface':rows.arcface_similarity.mean(),'min_arcface':rows.arcface_similarity.min(),'mean_facenet':rows.facenet_similarity.mean(),'min_facenet':rows.facenet_similarity.min(),'mean_LPIPS':rows.LPIPS.mean(),'mean_outside_rms':rows.outside_rms.mean()})
    summary=pd.DataFrame(summaries);summary.to_csv(out/'editor_comparison.csv',index=False)
    eligible=summary[summary.editor.str.startswith('style')&summary.numerical_gate_passed]
    chosen=eligible.sort_values(['passing_factuals','min_arcface','mutable_channels'],ascending=[False,False,True]).iloc[0].editor if len(eligible) else None
    gate={'numerical_gate_passed':chosen is not None,'selected_editor':chosen,'required_passing_factuals':c['required_passing_factuals'],'qualitative_review_required':True,'final_test_accessed':False,'future_models_accessed':False,'GRACE_started':False,'thresholds_relaxed':False}
    dump(out/'editor_gate.json',gate)
    fig,axes=plt.subplots(8,5,figsize=(11,17));methods=['style1','style4','style8','w_linear']
    for i in range(8):
        axes[i,0].imshow(Image.open(out/'qualitative_examples'/f'{i}_factual.png'));axes[i,0].set_title(f'Calibration {i}',fontsize=9)
        for j,name in enumerate(methods,1):
            row=selected[(selected.editor==name)&(selected.calibration_index==i)].iloc[0]
            axes[i,j].imshow(Image.open(out/'qualitative_examples'/f'{i}_{name}_{int(row.alpha_index)}.png'))
            axes[i,j].set_title(f'{name}, α={row.alpha:g}, '+('PASS' if row.all_gates_pass else 'FAIL'),fontsize=8)
        for ax in axes[i]:ax.axis('off')
    fig.tight_layout(pad=.3);fig.savefig(out/'figures/calibration_editor_comparison.pdf',dpi=180);fig.savefig(out/'figures/calibration_editor_comparison.png',dpi=100);plt.close(fig)
    # Global mean curves show all amplitudes, including rejected high-change edits.
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    for name,rows in d.groupby('editor'):
        curve=rows.groupby('alpha').mean(numeric_only=True)
        axes[0].plot(curve.index,curve.smile_gain,'o-',label=name);axes[1].plot(curve.index,curve.arcface_similarity,'o-',label=name);axes[2].plot(curve.index,curve.outside_rms,'o-',label=name)
    for ax,title in zip(axes,['Smile-score gain','ArcFace cosine similarity','Outside-region RMS']):ax.set_title(title);ax.set_xlabel('Edit strength α');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures/editor_strength_curves.pdf');fig.savefig(out/'figures/editor_strength_curves.png',dpi=160);plt.close(fig)
    review={'status':'passed','jobs':18,'calibration_images':8,'evaluated_edits':192,'gpu_ids':list(range(8)),'source_hashes_unchanged':True,'training_only_controls':True,'numerical_gate_recomputed':True,'identity_cosines_recomputed':True,'zero_edit_identity_check_passed':True,'missing_arcface_pairs':int(d.arcface_similarity.isna().sum()),'missing_facenet_pairs':int(d.facenet_similarity.isna().sum()),'final_test_accessed':False,'future_models_accessed':False}
    dump(out/'completion_review.json',review)
    report=['# Calibration-only localized smile-editor screen',f'Raw run: `{out}`. Read-only parent: `{c["parent_run"]}`. All eight GPUs used. Final test images/results and future models were not used.',
      '## Question and approach','The previous29-dimensional smile-gradient subspace did not preserve identity. This screen tests whether a much smaller StyleGAN3 post-affine control can increase the original smile score while retaining facial appearance. This is a new adaptation inspired by [StyleSpace](https://arxiv.org/abs/2011.12799), not reuse of pretrained StyleGAN2 channel IDs.',
      '## Training-only control discovery','Use all256 saved training W codes for per-channel standard deviations and a512-input logistic smile-direction baseline. On the first32 training codes, differentiate the original smiling softmax score with respect to standardized style channels. Estimate outside-region image-Jacobian RMS using three fixed-seed Rademacher projections per example. Rank eligible channels by absolute mean smile gradient divided by estimated outside-Jacobian RMS plus1e-5, require75% gradient-sign agreement, and select1/4/8 channels. Layer indices3–11 are eligible; other modulation channels and the factual Fourier input transform remain fixed. A scalar α moves the selected channels along the normalized mean-gradient direction. The W baseline is a single unit-norm logistic-regression direction and is not eligible as the hard-style-bottleneck realization.',
      '## Predeclared calibration gates',json.dumps(c,indent=2),
      'These are engineering screen thresholds, not validated same-person verification operating points. The fixed lower-face ellipse is a coarse localization proxy, not a semantic mouth mask. A smile score is the original CLIP probability for “smiling face” versus “face”; it cannot establish a real smile on its own. A visual check must accompany the numerical gate. Missing detections fail.',
      '## Results',summary.to_markdown(index=False),json.dumps(gate,indent=2),
      'Per factual, use the smallest edit strength passing every gate. For failed cases the figure shows the strongest smile score among quality-preserving candidates, or the zero edit if no quality-preserving candidate exists. FAIL labels remain visible. Thus the panel is a disclosed calibration selection, not a test-performance figure. All192 candidates and every rejection remain in all_calibration_metrics.csv.',
      '## Independent identity evaluation','ArcFace buffalo_l uses the previously audited320px detector and five-landmark alignment. Independent FaceNet uses [facenet-pytorch2.6.0](https://github.com/timesler/facenet-pytorch), VGGFace2 InceptionResnetV1, MTCNN160 crops and512-dimensional cosine similarity. FaceNet is neither differentiated nor used in channel discovery. Its calibration threshold participates in editor screening, so it is not an untouched final test metric. Checkpoint hashes and acquisition source are retained in facenet_manifest.json.',
      '## Next-stage rule','Run GRACE only if a localized control passes the frozen numerical gates and qualitative review. Otherwise stop after reporting this screen. No threshold relaxation or repeated test-guided search is allowed. Any later hard identity constraint changes the domain-specific feasible set; it does not modify max_model mean_CF hinge or data-side WRO.',
      '## Execution provenance','The initial attempt preserved successful style discovery but failed at FaceNet checkpoint loading. The checkpoint uses legacy serialization and includes a classification head. The final run loads every key strictly, then switches to embedding mode. Completed discovery/controls are reused read-only from20260917_calibration_v1; failed evaluation attempts v1/v2 remain unchanged. The18 recorded successful jobs comprise10 reused setup/discovery jobs and8 new calibration-evaluation jobs. See discovery_reuse.json. No calibration outcomes existed in the failed attempts.',
      '## Checks',json.dumps(review,indent=2),json.dumps(json.loads((out/'style_smoke.json').read_text()),indent=2),
      'The style wrapper is checked against original synthesis at zero edit; its gradients are checked by finite differences and a checkpointed toy-network test. Immutable channels are asserted exactly equal on every style edit. Calibration thresholds, controls and input hashes were frozen before evaluation. No existing experiment output was overwritten.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n');dest=ROOT/'results/016_ffhq_smile_editor_v1';dest.mkdir(exist_ok=True);figdest=ROOT/'figures/ffhq_smile_editor_v1';figdest.mkdir(exist_ok=True)
    for name in ['report.md','editor_comparison.csv','selected_examples.csv','all_calibration_metrics.csv','editor_gate.json','completion_review.json','resolved_config.json','style_smoke.json','style_architecture.json','channel_ranking.csv','facenet_manifest.json','controls_frozen.json']:
        shutil.copyfile(out/name,dest/name)
    for path in (out/'figures').glob('*'):shutil.copyfile(path,figdest/path.name)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
