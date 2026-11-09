# Smile-label and CLIP-gradient audit

The automated audit is complete. Human validation on generated faces remains
pending. The evidence supports prompt sensitivity and a cohort-label concern;
it does not identify a validated replacement smile classifier.

## Results

All eight factual-image gradient checks passed. The largest relative error
between autograd and central finite differences was **0.7455%**, below the
predeclared 3% tolerance. These checks differentiate the continuous original
CLIP smile probability with respect to canonical image pixels, along the
normalized gradient direction with epsilon 0.001. They complement the previous
style-coordinate gradient check; they do not test every direction or imply that
the hard class decision is differentiable.

| Diagnostic prompt pair | Smiling factuals /8 | Smiling edits /8 | Negative-to-positive pairs /8 | Correct on 32 source examples |
|---|---:|---:|---:|---:|
| smiling face / face (original) | 0 | 6 | 6 | 26 |
| smiling face / non-smiling face | 0 | 2 | 2 | 28 |
| a person who is smiling / a person who is not smiling | 4 | 5 | 1 | 26 |

The first prompt in each pair defines the positive class. Every diagnostic uses
the frozen original CLIP checkpoint, its logit scale, and a 0.5 threshold; these
softmax scores are not established calibrated smile probabilities. Transitions
in this table mean crossing that threshold, not the previous full editor gate
with its additional score-gain requirement. All eight frozen independent4 edits
already satisfy the unchanged preservation limits in the parent run.

All prompt pairs were specified before this audit's scoring. The explicit-face
pair scores 28/32 on the reused CelebA source diagnostic versus 26/32 for the
original pair, but this small difference is not sufficient to select a prompt.
The supervised diagnostic previously scored 31/32 there; its training overlap
is unknown. These source examples cannot establish accuracy on generated faces.

The factual cohort was selected using the original classifier, which explains
why all original factual predictions are negative by construction. The third
prompt marks four of those factuals positive. This is classifier disagreement,
not proof of incorrect ground-truth labels. The 16 generated images comprise
eight related factual/edit pairs selected during sequential calibration, not
16 independent observations or an untouched test sample.

Scoring saved PNGs instead of native floating-point generator outputs changes
the original probability by at most **0.003614**. The original threshold-positive
edit count remains six. Quantization/preprocessing effects are disclosed rather
than presenting this audit as bitwise reproduction of native-image inference.

## Annotation package and remaining validation

The score-free, shuffled package contains all eight factuals and all eight
selected independent4 edits, with no cherry-picking. The HTML is standalone,
contains 384px previews, requires an explicit expression label for each image,
and exports CSV. The PDF contains the same 16 IDs on two pages. Original full
resolution image paths and checksums remain in the separate analyst mapping.

- `figures/ffhq_smile_label_audit/smile_annotation_v2.html`
- `figures/ffhq_smile_label_audit/smile_annotation_blinded_v2.pdf`
- `annotation_template.csv`
- `annotation_key_do_not_show_raters.csv` (analyst only)

Two human raters should independently label each item as `smiling`,
`not_smiling`, or `uncertain`, before viewing predictions, pairings, or this
report. Exposed teeth alone do not establish a smile. Use `uncertain` when the
expression is ambiguous or obscured. The interface does not infer or preselect
labels. Similar faces may be recognizable across items; hiding methods and
scores does not make the observations independent.

Retain both rater files. Report agreement and uncertain counts; adjudicate
disagreements without showing model scores. For each classifier, report the
confusion matrix on adjudicated unambiguous items and the excluded count. For
counterfactual semantics, require an adjudicated non-smiling factual and smiling
edit, retaining the existing preservation gates. Report factual smiles and
uncertain pairs separately; do not silently replace the original cohort or
retroactively change its success labels.

No human labels were supplied or fabricated. No generated-face accuracy or
semantic success rate is reported. Assistant visual descriptions are not used
as ground truth. The next milestone is this label audit, followed by an
independent, larger generated calibration cohort if classifier development is
needed. Changing f0 would require a new, explicitly named experiment with
compatible training labels and rebuilt model banks; existing results remain
results for the original f0.

## Reproducibility and preservation

Run from `/home/xu.zhao/iclr_robust_ce`:

```sh
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/audits/ffhq_smile_labels.py
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/audits/ffhq_smile_labels_finalize.py
```

This audit refuses to overwrite its output directories. It uses four CPU threads
for 48 saved images; no GPU training or image regeneration was needed.
Configuration: `experiments/configs/ffhq_smile_label_audit.json`, seed 20260917.
The code uses repository-local CLIP source, float32 inference, tensor bicubic
antialiased resize to 224, center crop, and original CLIP normalization.

Read-only raw parents:

- `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v2/20260917_extended_v1`
- `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3`

Checkpoint: `/tmp/ViT-B-32.pt`. The exact script/config/checkpoint and all 48
source image checksums are recorded in `input_manifest.json` and verified
unchanged after the audit. Raw images and checkpoints were not copied into Git;
only compact annotation figures and numerical tables were saved here.

The final annotation package is v2: it adds PDF row spacing and uses a more
compatible CSV string-escaping operation. Initial drafts remain preserved; no
classifier inference was repeated for the layout fix. Four focused existing
editor tests passed using the documented isolated FFHQ dependency paths.

Source labels and provenance are reused from the parent's
`supervised_smile_source_validation.csv` and
`supervised_source_audit_manifest.json`: the first 32 rows of the pinned CelebA
source audit, not the FFHQ final test set. Independent-model predictions are
reused from the previously audited parent, not newly optimized or treated as
labels. No editor, identity limit, classifier, bank, manuscript claim, final
FFHQ test image, or future model was changed or accessed for this audit.
