"""Audit extended calibration editor gates and independent smile diagnostics."""
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
from lift_ce.experiments.ffhq_smile_editor_v2 import gate_table,passes_quality
from lift_ce.ffhq.supervised_smile import SupervisedSmile


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output;c=json.loads((out/'resolved_config.json').read_text());old=Path(c['previous_screen']);frozen=json.loads((out/'protocol_frozen.json').read_text())
    for path,digest in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==digest
    assert sha(out/'resolved_config.json')==frozen['config_sha256'];assert sha(out/'checkpoints/controls.pt')==frozen['controls_sha256']
    jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')];assert {int(j['physical_gpu']) for j in jobs}==set(range(8));assert len([j for j in jobs if j['phase']=='scalar'])==8
    for j in jobs:
        if j['phase']!='initialize':assert j['start_utc']>frozen['utc']
    d=pd.read_csv(out/'all_candidates_with_supervised_smile.csv');d,counts=gate_table(d,c)
    assert len(d[d.source=='extended_scalar'])==8*3*4
    scalar=json.loads((out/'scalar_gate.json').read_text());vector_jobs=[j for j in jobs if j['phase']=='vector'];assert len(vector_jobs)==(0 if scalar['passed'] else 16)
    # Recompute scalar/new-vector identity similarities from independently saved embeddings.
    max_error=0.
    for i in range(8):
        emb=np.load(out/'jobs'/f'scalar_{i}'/'embeddings.npz')
        for row in d[(d.source=='extended_scalar')&(d.calibration_index==i)].itertuples():
            for field,prefix in [('arcface_similarity','arc'),('facenet_similarity','facenet')]:
                if f'{prefix}_factual' in emb and f'{prefix}_{row.editor}_{int(row.alpha_index)}' in emb:
                    value=float(emb[f'{prefix}_factual']@emb[f'{prefix}_{row.editor}_{int(row.alpha_index)}']);max_error=max(max_error,abs(value-getattr(row,field)))
                else:assert pd.isna(getattr(row,field))
    for job in vector_jobs:
        path=out/'jobs'/job['job'];table=pd.read_csv(path/'metrics.csv');points=load(path/'coordinates.pt');assert len(points)==41 and float(points.abs().max())<=c['independent']['coordinate_bound']+1e-6
        accepted=table[table.accepted];assert all(passes_quality(r,c) for r in accepted.to_dict('records'));assert (accepted.smile_score.diff().dropna()>=-1e-7).all()
        emb=np.load(path/'embeddings.npz')
        for row in table.itertuples():
            for field,prefix in [('arcface_similarity','arc'),('facenet_similarity','facenet')]:
                if f'{prefix}_factual' in emb and f'{prefix}_{row.step}' in emb:
                    value=float(emb[f'{prefix}_factual']@emb[f'{prefix}_{row.step}']);max_error=max(max_error,abs(value-getattr(row,field)))
                else:assert pd.isna(getattr(row,field))
    assert max_error<1e-6
    # Verify every independent-model prediction with a fresh model instance.
    model=SupervisedSmile(out/'checkpoints/supervised_smile.pt');supervised_error=0.
    for start in range(0,len(d),32):
        block=d.iloc[start:start+32];fresh=model.score(block.image_path.tolist());supervised_error=max(supervised_error,max(abs(v['supervised_smile_probability']-float(row.supervised_smile_probability)) for v,row in zip(fresh,block.itertuples())))
    assert supervised_error<1e-7
    factual=pd.read_csv(out/'supervised_factual_scores.csv');factual_map=factual.set_index('calibration_index').supervised_smile_label.to_dict()
    d['supervised_factual_label']=d.calibration_index.map(factual_map);d['supervised_transition']=(d.supervised_factual_label==0)&(d.supervised_smile_label==1)
    d.to_csv(out/'audited_candidates.csv',index=False)
    selected=[]
    for (editor,i),rows in d.groupby(['editor','calibration_index']):
        hits=rows[rows.all_gates_pass]
        if len(hits):best=hits.sort_values(['pixel_L2_224','LPIPS']).iloc[0]
        else:
            safe=rows[rows.quality_pass];assert len(safe)>0;best=safe.sort_values(['smile_score','pixel_L2_224'],ascending=[False,True]).iloc[0]
        selected.append(best.to_dict())
    selected=pd.DataFrame(selected);selected.to_csv(out/'selected_examples.csv',index=False)
    summaries=[]
    for editor,rows in selected.groupby('editor'):
        summaries.append({'editor':editor,'passing_factuals':int(rows.all_gates_pass.sum()),'numerical_gate_passed':bool(rows.all_gates_pass.sum()>=c['required_passing_factuals']),'mean_CLIP_smile':rows.smile_score.mean(),'mean_CLIP_gain':rows.smile_gain.mean(),'mean_ArcFace':rows.arcface_similarity.mean(),'minimum_ArcFace':rows.arcface_similarity.min(),'mean_FaceNet':rows.facenet_similarity.mean(),'mean_LPIPS':rows.LPIPS.mean(),'mean_outside_RMS':rows.outside_rms.mean(),'supervised_positive_outputs':int(rows.supervised_smile_label.sum()),'supervised_smile_transitions':int(rows.supervised_transition.sum()),'CLIP_pass_and_supervised_positive':int((rows.all_gates_pass&(rows.supervised_smile_label==1)).sum())})
    summary=pd.DataFrame(summaries);summary.to_csv(out/'editor_comparison.csv',index=False)
    decision={'numerical_gate_passed':bool(summary.numerical_gate_passed.any()),'passing_counts':{r.editor:int(r.passing_factuals) for r in summary.itertuples()},'supervised_factual_positive_count':int(factual.supervised_smile_label.sum()),'qualitative_review_required':True,'GRACE_started':False,'final_FFHQ_test_accessed':False,'future_models_accessed':False,'preservation_limits_relaxed':False}
    dump(out/'editor_gate.json',decision)
    methods=summary.editor.tolist();fig,axes=plt.subplots(8,len(methods)+1,figsize=(2.2*(len(methods)+1),17))
    for i in range(8):
        axes[i,0].imshow(Image.open(old/'qualitative_examples'/f'{i}_factual.png'));axes[i,0].set_title(f'Calibration {i}; supervised={factual_map[i]}',fontsize=8)
        for j,name in enumerate(methods,1):
            row=selected[(selected.editor==name)&(selected.calibration_index==i)].iloc[0];axes[i,j].imshow(Image.open(row.image_path));axes[i,j].set_title(f'{name}: '+('PASS' if row.all_gates_pass else 'FAIL')+f'\nCLIP {row.smile_score:.2f}; supervised {row.supervised_smile_probability:.2f}',fontsize=8)
        for ax in axes[i]:ax.axis('off')
    fig.tight_layout(pad=.3);fig.savefig(out/'figures/extended_editor_comparison.pdf',dpi=160);fig.savefig(out/'figures/extended_editor_comparison.png',dpi=100);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    for editor,rows in d[d.source!='independent'].groupby('editor'):
        curve=rows.groupby('alpha').mean(numeric_only=True)
        for ax,key in zip(axes,['smile_score','arcface_similarity','outside_rms']):ax.plot(curve.index,curve[key],'o-',label=editor)
    for ax,label in zip(axes,['Original CLIP smile score','ArcFace similarity','Outside-region RMS']):ax.set_xlabel('α');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[1].axhline(c['arcface_min'],ls='--',color='gray');axes[2].axhline(c['outside_rms_max'],ls='--',color='gray');axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures/extended_strength_curves.pdf');fig.savefig(out/'figures/extended_strength_curves.png',dpi=160);plt.close(fig)
    pd.DataFrame(jobs).to_csv(out/'runtime.csv',index=False)
    review={'status':'passed','jobs':len(jobs),'eight_gpus_used':True,'previous_scalar_candidates_reused':144,'new_scalar_candidates':96,'independent_candidates':int((d.source=='independent').sum()),'all_accepted_vector_iterates_preserve_gates':True,'maximum_identity_cosine_error':max_error,'supervised_prediction_recompute_error':supervised_error,'missing_ArcFace_pairs':int(d.arcface_similarity.isna().sum()),'missing_FaceNet_pairs':int(d.facenet_similarity.isna().sum()),'source_hashes_unchanged':True,'final_FFHQ_test_accessed':False,'future_models_accessed':False};dump(out/'completion_review.json',review)
    manifest=json.loads((out/'supervised_smile_manifest.json').read_text())
    report=['# Extended calibration-only smile-editor comparison',f'Raw run: `{out}`. Previous screen: `{old}`. Stronger scalar settings and independent-channel optimization were authorized after inspecting the first screen, so this is a sequential calibration study, not independent confirmation. The FFHQ final test set and all future models remain untouched.',
      '## What changed','Reuse the same training-selected1/4/8 style channels. Add scalar strengths6,8,12,16; retain old0–4 candidates without regeneration. If scalar controls fail the6/8 gate, allow the selected4 or8 standardized channel coefficients to move independently. Start from each factual’s best preservation-feasible scalar edit, maximize original CLIP smile probability using40 normalized-gradient steps with adaptive step size, and bound each coefficient to±16. Accept a trial only when its original score is nondecreasing and every measured preservation gate passes. Rejected trials remain saved. This is constrained editor construction, not GRACE or a current-validity repair branch.',
      '## Unchanged limits','Every accepted edit requires ArcFace≥0.8, FaceNet≥0.8, LPIPS≤0.15 and outside-region RMS≤0.04. Smile success still requires original CLIP probability≥0.5 and increase≥0.05. Six of eight calibration factuals must pass. The fixed mouth/cheek ellipse remains a proxy, not segmentation. Nonselected styles and the input transform stay factual. No identity penalty was added.',
      '## Results',summary.to_markdown(index=False),json.dumps(decision,indent=2),
      'Selected examples use the minimum pixel cost among candidates passing all original gates; failures display the highest-score preservation-feasible candidate. This is explicitly post-selection on calibration data. The complete candidate table contains rejected large edits and preserves all failures.',
      '## Independent supervised smile check',json.dumps(manifest,indent=2),
      'The [released LeNNon model](https://huggingface.co/AiresPucrs/LeNNon-Smile-Detector) was trained on CelebA smile labels. Architecture and upstream preprocessing were audited; it resizes RGB to100×100 with values[0,1]. The checkpoint retains10 logits despite the binary task. We preserve upstream argmax>0 mapping and report1−P(class0), raw class and extra-class mass. Only classes0/1 were predicted on the32 annotated source-audit examples. The source check scored31/32 but training overlap is unknown; it is not a generalization claim or calibration of probabilities. This small supervised model is a diagnostic, not a replacement for the original classifier or a ground-truth annotator. Disagreements and visually ambiguous smiles must be reported.',
      factual.to_markdown(index=False),
      'Source-audit examples are the first32 test rows from eurecom-ds/celeba, an external dataset—not the FFHQ final test cohort. Neither these examples nor the independent smile scores affect channel discovery, the optimization objective, or the original numerical gates.',
      '## Interpretation rule','Only proceed to GRACE if an editor achieves the numerical gate and the qualitative/independent smile check supports meaningful edits. Do not relax identity limits or claim success solely from larger classifier scores. If original source labels already disagree with visible/independent smiles, report the cohort/label limitation rather than silently replacing labels or changing success thresholds.',
      '## Reproducibility and checks',json.dumps(c,indent=2),json.dumps(review,indent=2),
      'Every independent-channel accepted iterate is checked against the true measured preservation gates; missing detections fail. All identity similarities are recomputed from saved embeddings, and independent smile predictions are re-evaluated with a fresh model instance. Raw coordinates, rejected candidates, PNGs, commands, seed, model hashes and phase timing are retained. Previous outputs and the manuscript are unchanged.']
    (out/'report.md').write_text('\n\n'.join(report)+'\n');dest=ROOT/'results/017_ffhq_smile_editor_v2';dest.mkdir(exist_ok=True);figdest=ROOT/'figures/ffhq_smile_editor_v2';figdest.mkdir(exist_ok=True)
    for name in ['report.md','editor_comparison.csv','selected_examples.csv','editor_gate.json','completion_review.json','resolved_config.json','supervised_smile_manifest.json','supervised_smile_source_validation.csv','supervised_factual_scores.csv','runtime.csv','scalar_gate.json']:
        shutil.copyfile(out/name,dest/name)
    for path in (out/'figures').glob('*'):shutil.copyfile(path,figdest/path.name)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
