"""Independently verify the GENKI run and save compact summaries and figures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/genki_ffhq_report_mpl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lift_ce.experiments.genki_ffhq_classifier_v1 import features,metrics,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    torch.set_num_threads(1)
    dest=ROOT/'results/019_genki_ffhq_classifier';figdest=ROOT/'figures/genki_ffhq_classifier'
    assert not dest.exists() and not figdest.exists()
    cfg=json.loads((out/'resolved_config.json').read_text());table=pd.read_csv(out/'dataset_manifest.csv')
    assert len(table)==4000 and table.groupby('pixel_sha256').split.nunique().max()==1
    for r in table.itertuples():assert sha(r.image_path)==r.sha256
    for path,value in json.loads((out/'input_manifest.json').read_text()).items():assert sha(path)==value
    jobs=[json.loads(p.read_text()) for p in (out/'jobs').glob('*/complete.json')]
    assert len(jobs)==18
    for phase in ['extract','fit']:
        assert {int(j['physical_gpu']) for j in jobs if j['phase']==phase}==set(range(8))
    frozen=json.loads((out/'selection_frozen.json').read_text());assert sha(out/'checkpoints/selected.pt')==frozen['checkpoint_sha256']
    evaluate=json.loads((out/'jobs/evaluate/started.json').read_text());assert evaluate['start_utc']>frozen['utc']
    x=features(out);assert x.shape==(4000,512) and torch.isfinite(x).all()
    assert float((x.norm(dim=1)-1).abs().max())<1e-5
    va=table.split=='validation';valrows=[]
    for i in range(8):
        checkpoint=torch.load(out/'checkpoints'/f'candidate_{i}.pt',map_location='cpu',weights_only=True)
        head=nn.Linear(512,2);head.load_state_dict(checkpoint['head_state'])
        with torch.no_grad():p=head(x[va.to_numpy()]).softmax(1)[:,0].numpy()
        fresh=metrics(table.loc[va,'smile'].to_numpy(),p)
        recorded=json.loads((out/'jobs'/f'fit_{i}'/'validation.json').read_text())
        assert abs(fresh['balanced_accuracy']-recorded['balanced_accuracy'])<1e-8
        assert abs(fresh['log_loss']-recorded['log_loss'])<1e-5
        valrows.append(recorded)
    selected=sorted(valrows,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],-r['l2']))[0]
    assert selected==frozen['selected']
    checkpoint=torch.load(out/'checkpoints/selected.pt',map_location='cpu',weights_only=True)
    head=nn.Linear(512,2);head.load_state_dict(checkpoint['head_state']);te=table.split=='test'
    with torch.no_grad():probs=head(x[te.to_numpy()]).softmax(1)[:,0].numpy()
    saved=pd.read_csv(out/'test_predictions.csv');assert saved['index'].tolist()==table.loc[te,'index'].tolist()
    assert np.max(np.abs(probs-saved.smile_probability))<1e-7
    test=metrics(saved.smile.to_numpy(),probs);recorded=json.loads((out/'test_metrics.json').read_text())
    assert test==recorded
    original_checks=pd.read_csv(out/'adapter_checks.csv');assert len(original_checks)==8 and (original_checks.interface_error<1e-6).all()
    precision=Path('/tmp/genki_adapter_precision_audit_20260917_v2')
    precision_review=json.loads((precision/'checks.json').read_text())
    assert precision_review['checkpoint_sha256']==frozen['checkpoint_sha256']
    assert precision_review['primary_pass_count']==precision_review['smaller_epsilon_pass_count']==8
    assert sha(ROOT/'experiments/audits/genki_adapter_precision.py')==precision_review['script_sha256']
    checks=pd.read_csv(precision/'gradient_checks.csv');checks=checks[checks.epsilon==.001]
    assert len(checks)==8 and checks.passed.all()
    transfer=pd.read_csv(out/'transfer_predictions.csv');assert len(transfer)==16
    factual=transfer[transfer.kind=='factual'].set_index('index');edited=transfer[transfer.kind=='edited'].set_index('index')
    summary={'factual_positive':int(factual.genki_smile_label.sum()),'edited_positive':int(edited.genki_smile_label.sum()),
             'negative_to_positive_pairs':int(((factual.genki_smile_label==0)&(edited.genki_smile_label==1)).sum()),
             'mean_paired_probability_change':float((edited.genki_smile_probability-factual.genki_smile_probability).mean()),
             'generated_face_ground_truth_available':False,'old_edits_selected_using_original_CLIP':True}
    splits=json.loads((out/'split_summary.json').read_text())
    review={'status':'passed_with_documented_precision_followup','completed_worker_jobs':len(jobs),'attempted_worker_jobs':19,'eight_gpus_used':True,'dataset_images':len(table),
            'all_dataset_and_parent_hashes_verified':True,'no_exact_duplicate_cross_split_overlap':True,
            'selection_recomputed_using_validation_only':True,'test_evaluation_after_selection_freeze':True,
            'held_out_test_predictions_independently_recomputed':True,'adapter_gradient_checks_passed':8,
            'maximum_adapter_gradient_relative_error':float(checks.relative_error.max()),
            'original_cuda_float32_gradient_checks_passed':int(original_checks.passed.sum()),
            'supplemental_cpu_float64_gradient_checks_passed':8,
            'original_evaluation_failure_preserved':True,
            'focused_tests_passed':3,'new_counterfactuals_generated':False,'FFHQ_test_accessed':False,
            'GRACE_started':False,'semantic_transfer_accuracy_established':False}
    dest.mkdir();figdest.mkdir()
    for name in ['resolved_config.json','split_summary.json','input_manifest.json','validation_candidates.csv','selection_frozen.json',
                 'test_metrics.json','test_predictions.csv','transfer_predictions.csv','adapter_checks.csv']:
        (dest/name).write_bytes((out/name).read_bytes())
    for name in ['gradient_checks.csv','precision_scores.csv','checks.json']:
        (dest/('precision_'+name)).write_bytes((precision/name).read_bytes())
    table[['index','smile','split','sha256','pixel_sha256']].to_csv(dest/'split_manifest.csv',index=False)
    pd.DataFrame(jobs).to_csv(dest/'runtime.csv',index=False)
    (dest/'completion_review.json').write_text(json.dumps(review,indent=2)+'\n')
    (dest/'transfer_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    cm=np.array(test['confusion_matrix_labels_0_1']);axes[0].imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['Not smiling','Smiling'],yticklabels=['Not smiling','Smiling'],xlabel='Predicted label',ylabel='GENKI annotation',title=f'GENKI held-out test (n={len(saved)})')
    v=pd.DataFrame(valrows).sort_values('l2');axes[1].semilogx(v.l2,v.balanced_accuracy,'o-');axes[1].set(xlabel='L2 penalty',ylabel='Validation balanced accuracy',title='Selection uses GENKI validation only');axes[1].grid(alpha=.3)
    fig.tight_layout();fig.savefig(figdest/'genki_classifier_evaluation.pdf');fig.savefig(figdest/'genki_classifier_evaluation.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));idx=np.arange(8)
    ax.bar(idx-.18,factual.genki_smile_probability,.36,label='Factual');ax.bar(idx+.18,edited.genki_smile_probability,.36,label='Existing 4-channel edit')
    ax.axhline(.5,color='gray',ls='--');ax.set(xticks=idx,xlabel='Previous FFHQ calibration index',ylabel='GENKI-head smile probability',ylim=(0,1),title='Transfer predictions; generated-face labels unavailable');ax.legend();fig.tight_layout()
    fig.savefig(figdest/'ffhq_transfer_predictions.pdf');fig.savefig(figdest/'ffhq_transfer_predictions.png',dpi=140);plt.close(fig)
    report=f'''# GENKI-4K smile classifier on the FFHQ infrastructure

The supervised classifier and differentiable FFHQ adapter are implemented and
evaluated. The original CLIP image encoder is frozen; a new binary linear head
learns GENKI human smile labels. It uses no text prompts. This is a supervised
linear probe, not end-to-end encoder fine-tuning or generator training.

Raw run: `{out}`.
Selected head: `{out}/checkpoints/selected.pt`.
Original encoder: `{cfg['clip_checkpoint']}`; its hash is saved in the manifest
and head checkpoint. Original StyleGAN3 and identity assets remain reusable and
unchanged. All eight GPUs were used for feature extraction and candidate fitting.

## GENKI results

- Training images: {int((table.split=='train').sum())}; validation: {int(va.sum())}; held-out test: {len(saved)}.
- Selected L2 penalty: {selected['l2']}; validation balanced accuracy: {selected['balanced_accuracy']:.4%}.
- Test accuracy: **{test['accuracy']:.4%}**.
- Test balanced accuracy: **{test['balanced_accuracy']:.4%}**.
- Test AUROC: **{test['auroc']:.6f}**; log loss: **{test['log_loss']:.6f}**.
- Confusion matrix, rows true and columns predicted, order [not smiling, smiling]: `{test['confusion_matrix_labels_0_1']}`.

These are image-level results from one predeclared split, not a repeated-seed or
subject-disjoint estimate. The archive contains {splits['duplicate_images']} repeated decoded images;
exact duplicate groups were kept together, causing small deviations from the
nominal 2,800/600/600 image counts. Source annotations define the evaluation
labels; remaining label noise is possible. The release provides no subject IDs,
and near-duplicates were not exhaustively audited. Frozen CLIP pretraining overlap
with these web images is unknown; the held-out split excludes head-training data,
not necessarily pretrained-encoder exposure.

## FFHQ integration and transfer diagnostic

`lift_ce.ffhq.genki_classifier.GenkiSmileClassifier` accepts native generator RGB
in [-1,1] and returns logits in the existing FFHQ order [smiling, not_smiling].
Resize/crop, normalization and feature computation remain differentiable.
All eight image-gradient checks pass in the CPU float64 diagnostic; maximum
relative finite-difference error is {checks.relative_error.max():.4%}. Tests also cover input-gradient flow, class-order
rejection and keeping duplicate groups out of multiple partitions.

The original CUDA float32 evaluation saved predictions and test metrics, then
stopped because three of eight finite-difference checks exceeded 3% tolerance
(largest discrepancy {original_checks.relative_error.max():.2%}). Those failed checks and logs
remain preserved. A separate CPU float64 diagnostic uses the same selected head,
the original epsilon 0.001 and unchanged 3% tolerance; all eight checks pass at
both epsilon 0.001 and 0.0001. CPU64 and CUDA32 prediction probabilities differ
by at most {precision_review['maximum_cpu64_cuda32_probability_difference']:.6g} on the 16 saved images.
This supports a numerical precision explanation for the finite-difference
failure; it does not establish every GPU derivative at every input. No checkpoint
was retrained, no hyperparameter changed, and no saved evaluation output was
overwritten during this diagnostic. Exact native/canonical adapter logits agree
on all eight original interface checks.
For the float64 audit only, CLIP's LayerNorm wrapper is evaluated using the same
formula without its forced float32 cast. The initial precision-audit attempt
stopped at a mixed-dtype error before producing predictions; it is preserved.

```python
from lift_ce.ffhq.genki_classifier import GenkiSmileClassifier
classifier = GenkiSmileClassifier("{out}/checkpoints/selected.pt", device="cuda:0")
smile_probability = classifier(native_generated_images).softmax(1)[:, 0]
```

On the same eight existing FFHQ calibration pairs, the new head predicts
{summary['factual_positive']}/8 factuals and {summary['edited_positive']}/8 existing edits as smiling,
with {summary['negative_to_positive_pairs']}/8 negative-to-positive transitions.
These are **transfer predictions, not FFHQ accuracy or semantic success**.
The images and edits were previously selected using the original CLIP classifier;
no new counterfactuals were optimized with this head. Human labels for generated
faces are still unavailable, so GENKI test performance alone does not establish
reliable transfer to FFHQ.

GENKI images are used for classifier training/evaluation. We have not inverted
real GENKI photographs into StyleGAN3, so this is not yet a counterfactual study
whose factuals are real GENKI images. Such a study would need image inversion and
an explicit reconstruction audit before attributing changes to recourse.

The current four-channel editor and preservation limits are unchanged. Previous
Base CFs, the conditional flow, attack banks and future-model checkpoints use the
old classifier/labels and cannot be interpreted as results for this new head.
An end-to-end GENKI-head CF/GRACE experiment requires new compatible construction
and evaluation assets. No such robustness result is claimed here.

## Protocol and reproducibility

Official archive: [{cfg['dataset_citation']}]({cfg['dataset_url']}).
The included README maps the Nth label row to fileNNNN.jpg and specifies 1=smile,
0=non-smile. We verified all 4,000 images and labels. Pose labels are retained in
the external manifest and not used as prediction targets.

Frozen split seed: {cfg['seed']}. Use decoded-image hashes as groups, stratify
unique groups by smile label, then split groups 70/15/15. Class 0 of the FFHQ
adapter corresponds to dataset smile label 1. Input is RGB, tensor bicubic
antialiased resize of the short side to 224, center crop, original CLIP
normalization; features are L2-normalized 512-dimensional image embeddings.
No augmentation is used for this initial linear probe.

Fit eight predeclared L2 strengths using full-batch float64 LBFGS, maximum 300
iterations, strong-Wolfe search, zero-initialized weights/bias. Minimize mean cross
entropy plus half L2 times squared weight norm; bias is unpenalized. Rank on GENKI
validation balanced accuracy, then log loss, then stronger regularization. Freeze
the chosen checkpoint before evaluating test labels or FFHQ transfer predictions.
All scores use the fixed 0.5 decision threshold; probabilities are not additionally
calibrated. Selection is reproduced independently from persisted features/heads.

```sh
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/038_run_genki_ffhq.py --output {out}
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/audits/genki_adapter_precision.py --run {out} --output /tmp/genki_adapter_precision_audit_20260917_v2
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/039_report_genki_ffhq.py --output {out}
```

The controller invokes `experiments/run.py` for each worker, with source
fingerprints, explicit GPU assignment, a 14% per-worker memory cap and immutable
completion markers. Commands are saved in raw logs (the initial command is in
`/tmp/genki_ffhq_initialize.log`). Dataset, features and checkpoints remain
outside Git. Previous raw outputs and manuscript claims are preserved. The FFHQ
final test and old future-model bank were not accessed.

## Independent checks

Eighteen worker jobs completed; the nineteenth saved evaluation results but
stopped at its CUDA numerical-gradient assertion. The separate precision audit
completed the integration check without rerunning training or overwriting that
worker. Dataset/parent checksums, split isolation, eight-GPU use,
feature norms, validation selection, test prediction reproduction, selection
before evaluation, and eight adapter gradients passed. Three focused tests
passed. Summary: `completion_review.json`.
'''
    (dest/'report.md').write_text(report)
    s=yaml.safe_load((ROOT/'state.yaml').read_text())
    entry=dict(task='tasks/006_genki_ffhq_classifier.md',phase='complete',run=str(out),
               report='results/019_genki_ffhq_classifier/report.md',gpu_ids=list(range(8)),completed_jobs=18,attempted_jobs=19,
               precision_followup_completed=True,original_evaluation_failure_preserved=True,
               selected_checkpoint=str(out/'checkpoints/selected.pt'),test_accuracy=test['accuracy'],
               test_balanced_accuracy=test['balanced_accuracy'],semantic_FFHQ_transfer_validated=False,
               FFHQ_test_accessed=False,GRACE_started=False,new_counterfactuals_generated=False,
               focused_tests_passed=3,updated_utc=datetime.now(timezone.utc).isoformat())
    s.update(active_task=entry['task'],status='complete',current_work=entry,
             next_actions=['Validate the GENKI-head predictions on independently labeled generated faces.',
                           'For an end-to-end new-classifier experiment, construct compatible Base CFs and rebuild model banks; do not reuse old classifier-dependent assets.'])
    s.setdefault('completed_work',{})['genki_ffhq_classifier_v1']=entry
    (ROOT/'state.yaml').write_text(yaml.safe_dump(s,sort_keys=False))
    with (ROOT/'notes/decisions.md').open('a') as f:
        f.write(f'\n\n## 2026-09-17: GENKI-4K supervised head on FFHQ encoder\n\nUser authorized GENKI-4K on the FFHQ infrastructure. Trained a supervised two-class linear head on frozen CLIP image embeddings using official GENKI smile annotations. Selected L2={selected["l2"]} on validation only; held-out GENKI accuracy {test["accuracy"]:.4%}, balanced accuracy {test["balanced_accuracy"]:.4%}. Exact duplicate groups isolated; subject identities and pretraining overlap unknown. Eight GPUs, 18 completed workers; the evaluation worker saved results but failed three CUDA float32 finite-difference checks. A separate CPU float64 audit passes all eight at the same tolerance and two step sizes; original outputs preserved. Three focused tests pass. Differentiable FFHQ adapter retains smile at output index0. Old generator, editor and identity limits preserved. No inversion of real GENKI photographs, new CFs, new bank, or GRACE result. Generated-face transfer is diagnostic without human labels. Report: results/019_genki_ffhq_classifier/report.md. Raw: {out}.\n')
    print(json.dumps({'test':test,'transfer':summary,'review':review},indent=2))


if __name__=='__main__':main()
