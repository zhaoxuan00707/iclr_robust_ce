# GENKI-4K smile classifier on the FFHQ infrastructure

The supervised classifier and differentiable FFHQ adapter are implemented and
evaluated. The original CLIP image encoder is frozen; a new binary linear head
learns GENKI human smile labels. It uses no text prompts. This is a supervised
linear probe, not end-to-end encoder fine-tuning or generator training.

Raw run: `/home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1`.
Selected head: `/home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1/checkpoints/selected.pt`.
Original encoder: `/tmp/ViT-B-32.pt`; its hash is saved in the manifest
and head checkpoint. Original StyleGAN3 and identity assets remain reusable and
unchanged. All eight GPUs were used for feature extraction and candidate fitting.

## GENKI results

- Training images: 2798; validation: 600; held-out test: 602.
- Selected L2 penalty: 0.0003; validation balanced accuracy: 94.9704%.
- Test accuracy: **94.8505%**.
- Test balanced accuracy: **94.6528%**.
- Test AUROC: **0.987100**; log loss: **0.195743**.
- Confusion matrix, rows true and columns predicted, order [not smiling, smiling]: `[[254, 21], [10, 317]]`.

These are image-level results from one predeclared split, not a repeated-seed or
subject-disjoint estimate. The archive contains 14 repeated decoded images;
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
relative finite-difference error is 1.4310%. Tests also cover input-gradient flow, class-order
rejection and keeping duplicate groups out of multiple partitions.

The original CUDA float32 evaluation saved predictions and test metrics, then
stopped because three of eight finite-difference checks exceeded 3% tolerance
(largest discrepancy 22.69%). Those failed checks and logs
remain preserved. A separate CPU float64 diagnostic uses the same selected head,
the original epsilon 0.001 and unchanged 3% tolerance; all eight checks pass at
both epsilon 0.001 and 0.0001. CPU64 and CUDA32 prediction probabilities differ
by at most 0.000667874 on the 16 saved images.
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
classifier = GenkiSmileClassifier("/home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1/checkpoints/selected.pt", device="cuda:0")
smile_probability = classifier(native_generated_images).softmax(1)[:, 0]
```

On the same eight existing FFHQ calibration pairs, the new head predicts
1/8 factuals and 4/8 existing edits as smiling,
with 3/8 negative-to-positive transitions.
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

Official archive: [http://mplab.ucsd.edu, The MPLab GENKI-4K Dataset.](https://mplab.ucsd.edu/398/media/genki4k.tar).
The included README maps the Nth label row to fileNNNN.jpg and specifies 1=smile,
0=non-smile. We verified all 4,000 images and labels. Pose labels are retained in
the external manifest and not used as prediction targets.

Frozen split seed: 20260917. Use decoded-image hashes as groups, stratify
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
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/038_run_genki_ffhq.py --output /home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/audits/genki_adapter_precision.py --run /home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1 --output /tmp/genki_adapter_precision_audit_20260917_v2
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/039_report_genki_ffhq.py --output /home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1
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
