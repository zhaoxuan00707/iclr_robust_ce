# GENKI-head GRACE with frozen content and measured preservation limits

User authorized GRACE with the GENKI-trained classifier and requested identity
and background preservation. Use eight GPUs and save numerical/qualitative results.

Keep the existing four training-selected style channels; freeze all other styles
and the factual Fourier input. Coordinates u are absolute selected styles divided
by the frozen training standard deviations. Use the current GRACE objective
max_model mean_CF [0.5-p(smile)]_+ and exact Euclidean W2 in u for all three
geometry terms, with beta=[1,1.05,1.1,1.2] and radii [1,2]. Reuse solve_raw;
include all its evaluated iterates in the finite candidate pool. Final candidates
must satisfy geometry, coefficient displacement <=16, ArcFace>=.8, FaceNet>=.8,
LPIPS<=.15 and outside-mask RMS<=.04 for every factual. No nominal-validity repair
or penalty enters GRACE. The mask is the previous coarse mouth/cheek ellipse.

References are the first eight generated training faces classified smiling by
the new head, encoded into u. Reference smiles refer to their original content;
transplanting u into a different factual does not guarantee a smile. This is the
previous task-space/frozen-content realization, not full-W geometry or proof of
disentanglement. Identity gates apply to returned faces, not reference images.

Choose eight factuals per calibration/test pool in seeded order, requiring nominal
non-smile and detectable faces by both identity models, before CF generation.
Record eligibility exclusions. These are existing generated FFHQ pools, not real
GENKI photographs or a new untouched image benchmark. Retain nominal Base failures.
Generate Base using nominal gradient ascent in the four coordinates with true
preservation checks; freeze the first .5-crossing after bisection, or the best
preservation-feasible point after 40 steps if it fails. Freeze Base before banks.

Use only the GENKI training partition for Update. Encoder fixed; deterministic
full-batch SGD updates of the two-class head, 120 steps, lr .1 and original L2
.0003, starting from the frozen selected head. Radii use RGB224 pixel W2; certify
attack feasibility by the class-preserving paired displacement upper bound.
Four pixel attacks per radius, 32 mutable images, 12 ascent steps, one-step
hypergradient surrogate, fractions .5/1; save each exact full-update head. Include
a clean full update. All Q use the same radius-compatible bank. No weight attacks.

Choose r/beta on four declared calibration shifts. Freeze all test outputs before
30 future shifted-training updates (brightness/noise/blur, five severities, two
realizations). Report deterministic duplicate future models. Preserve all old
outputs and disclose optimization/domain/finite-bank limitations and failed cases.
