# User-approved FFHQ localized smile-editor screen

Follow-up to the failed task002 identity bottleneck. User approved a calibration-only comparison of localized smile controls before another GRACE run.

Reuse original StyleGAN3; discover controls on training only. Compare1/4/8 selected post-affine style channels with a one-direction W baseline. Keep all other style channels and the factual Fourier input transform fixed. Use original smile classifier, ArcFace, independent FaceNet and a coarse mouth/cheek localization diagnostic. Predeclare thresholds and amplitudes before evaluation. Do not inspect final test or future outcomes.

Proceed to GRACE only if a localized editor passes the numerical calibration gate and qualitative inspection. Otherwise save the negative screen and do not start a full robustness sweep. Any subsequent hard identity constraint changes the domain-specific feasible set, not the robust objective, and infeasibility must be reported.

Exact gates, mask, selection and provenance: experiments/configs/ffhq_smile_editor_v1.json. Gate requires6/8 calibration factuals with smiling probability≥0.5 and gain≥0.05, ArcFace and FaceNet cosines≥0.8, LPIPS≤0.15 and outside-mask RMS≤0.04. These are engineering screening gates, not validated identity verification thresholds. Missing detections fail.
