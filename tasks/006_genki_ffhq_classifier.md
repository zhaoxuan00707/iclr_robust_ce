# GENKI-4K supervised smile classifier on FFHQ infrastructure

User request: use GENKI-4K on the existing FFHQ infrastructure. Reuse original
frozen CLIP ViT-B/32 image encoder, StyleGAN3 generator and identity evaluation;
replace the text-prompt decision with a supervised two-class linear head.
This task implements and evaluates that new classifier, not a retrained generator.

Download the official UCSD GENKI-4K archive. Verify 4,000 images and smile labels
(1=smile); preserve exact duplicate groups across a stratified 70/15/15 split.
No subject IDs are provided, so do not claim identity-disjoint evaluation.
Freeze split seed 20260917, eight L2 penalties and selection before training.
Extract frozen image features on eight GPUs, train one head per penalty, select
by validation balanced accuracy, then log loss, then stronger regularization.
Evaluate the selected classifier on the GENKI test partition once, after freeze.
No FFHQ image or test label participates in hyperparameter selection.

Retain FFHQ class order [smiling, not_smiling] for the differentiable adapter.
Audit score/gradient compatibility on the 16 saved FFHQ calibration images.
Generated-face ground truth is still absent: report transfer predictions only.
Keep all identity limits fixed and old raw runs immutable. Old CLIP-labeled flow,
Base CFs, model banks and future models are not results for the new classifier.
Do not silently plug the head into those artifacts or claim GRACE performance.

Store raw dataset, features and checkpoints outside Git; summaries under
results/019_genki_ffhq_classifier, figures under figures/genki_ffhq_classifier.
Record provenance, commands, seed, split IDs, metrics and limitations. Run focused
tests and update state.yaml and notes/decisions.md.
