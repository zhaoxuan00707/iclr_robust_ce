# User-approved localized-editor calibration extension

Extend the existing frozen controls to alpha6,8,12,16, preserving the original
ArcFace0.8, FaceNet0.8, LPIPS0.15 and outside-RMS0.04 gates. Do not regenerate
previous scalar edits. Compare on the same8 calibration factuals; final FFHQ test
and future models remain untouched.

Audit a supervised smile model independently of original CLIP. Preserve original
classifier/objective and report disagreements; do not treat either model as a
semantic ground-truth annotation. Validate preprocessing and labels on annotated
external examples and inspect calibration images. Model/data overlap limits must
be disclosed.

If stronger scalar edits fail the predeclared6/8 success gate, optimize the selected
4/8 channel coefficients independently, freezing every other style and factual
input transform. Use40 normalized-gradient ascent steps, bounded coefficients,
and true preservation-gate checks before accepting each iterate. No soft identity
penalty or GRACE validity-repair branch is added; this is an editor feasibility
screen, not a robustness experiment. Do not relax preservation thresholds.

Only consider GRACE after quantitative and qualitative smile/identity checks pass.
Save all candidates, rejected steps, independent predictions and provenance.
Config: experiments/configs/ffhq_smile_editor_v2.json.
