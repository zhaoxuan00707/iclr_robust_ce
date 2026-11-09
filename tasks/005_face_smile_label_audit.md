# FFHQ smile-label audit

Authorized after the user asked whether CLIP was differentiable and approved
validating the smile labels/classifier while retaining the four-channel editor.

Use only the eight existing calibration factuals and their frozen independent4
edits, plus the previously downloaded 32 labeled CelebA diagnostic examples.
Do not regenerate images, optimize new edits, change identity gates, access the
FFHQ final test/future models, or replace f0 or its model bank.

Before scoring, freeze three diagnostic prompt pairs: original smiling face/face,
smiling face/non-smiling face, and a person who is smiling/a person who is not
smiling. Report every pair; do not select a winner from eight calibration cases.
Check CLIP image gradients against finite differences and distinguish that check
from the previously completed StyleGAN style-gradient check.

Create a randomized score-free annotation PDF and standalone HTML with CSV export
for all 16 generated images. Labels are smiling, not_smiling, uncertain. Request
two independent human raters; disagreements require adjudication. Never substitute
assistant descriptions or model predictions for human ground truth. Save the
mapping separately from the annotation interface. These are paired, calibration-
selected images, not 16 independent test examples.

Save numerical diagnostics and a reproducible command under results/018, figures
under figures/ffhq_smile_label_audit, and update state.yaml. The automated audit
can finish while domain validation remains pending human labels. No FFHQ accuracy
or validated semantic counterfactual claim is allowed without those labels.
