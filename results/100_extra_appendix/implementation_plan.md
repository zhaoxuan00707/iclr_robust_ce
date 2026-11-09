# Task100 inspection and implementation plan — 2026-09-25

User override: one experiment seed only; eight GPUs. No multi-seed reruns.

1. MNIST `mnist_update_aware.update_model` delegates to `update_aware.unroll`,
   full-batch BCE SGD, T3 eta.01, eval-mode BN/dropout. `mnist_small_radius.futures`
   creates the held-out distributions using `stylemnist_grace_v1.shifted_images`.
   Reuse task016 p05001 (current visual/table source), seed0,64 factuals/64 latent
   supports, original frozen CNN/64D decoder and saved r.05/beta1.02 output.
   All36 existing test scenarios retained: three families x four severities x three
   deterministic derived realizations. These are not additional experiment seeds.
   One trajectory per scenario to40, save3/5/10/20/40; compare step3 with old saved
   checkpoint. Do not tune or regenerate Base. Preserve known repeated shifts.
2. `mnist_low_margin_base.optimize` passes r to `surrogate`; beta multiplies
   D=exact_w2(Base,target). That routine runs exactly800 support updates, primal20.
3. Disable ONLY `f+dual*violation+.5*rho*relu(violation)^2` augmentation and dual
   update for the explicit ablation. Retain diagnostic geometry, initialization,
   decoder, assignment, full smooth objective, and800 updates. No replacement
   regularizer/domain clamp. Add opt-in `disable_geometry`; default unchanged.
   Reuse constrained output rather than overwrite or rerun the completed solve.
4. Table1 in paper/Sections/04_experiments.tex is a hard-coded matched-cost LIFT-CF
   table, not a six-method per-dataset GRACE table. No source script/reference to
   raw runs accompanies it. Historical `main_tabular_geo_uniform_benchmark` note
   describes consolidation of Base/T-Rex/ROAR adaptation/Uniform-LIFT/Geo-LIFT,
   not the revised Method. Its source is absent from the imported core. Existing
   `heloc_diabetes_official_lift_v1` supports HELOC/Diabetes but uses historical
   transport, not revised GRACE. Cannot claim a direct Table1 reproduction from it.
5. `logistic_margin_anchor_benchmark.py` exists: torch Linear(d,1), AdamW1e-3,
   weight_decay1e-4, >=1000 training updates; only HELOC/Diabetes and old anchor
   interpolation, not revised GRACE or all four official baseline adapters.
6. No DiCE implementation found in local experiment/vendor code; dice_ml not
   installed in wm_gpu. Installing an official DiCE dependency is possible once
   the authoritative Table1 preprocessing/classifier/update path is identified.
7. Historical main/logistic datasets use seed0; revised HELOC task013 reuses seed1;
   MNIST current task016 uses seed0 plus recorded scenario offsets71000/81000.
   Preserve MNIST seed0. Do not silently call seed1 HELOC a seed0 reproduction.

Proceed with MNIST experiments1/2 now. Trace the Table1 source while these run.
An asynchronous clarification requests the exact current four-dataset source.
FICO is named separately in the task but local HELOC metadata calls itself FICO
HELOC; do not silently duplicate that dataset as a fourth benchmark. Document
unsupported cells and provenance gaps before proposing a replacement protocol.

## Authorized execution reconstruction after FICO download
User requested proceeding after dictionary-backed FICO validation. Use seed0,
existing60/20/20 loader, MLP64/32 AdamW40epochs and existing logistic >=1000-update
fit. Cap each cal/test cohort at128 unfavorable factuals (all if fewer). Preserve
all failures. Eight workers correspond to four datasets x two classifiers.
Use45 calibration +45 future scenarios: mean/scale/tail x .5/1/2.5 x5; train-only
predictive numeric features, same shifts and models across methods. Derive b_ref
from existing bootstrap utility on calibration feature distribution. Adopt verified
HELOC full10-step SGD.01 for every tabular variant; this is a documented diagnostic
choice because the PDF does not fully specify the other update operators.
GRACE grid r/b_ref{0,.25,1,3}, beta{1.05,1.1,1.2},800iteration cap. Final iterate
feasibility enforced, no best-iterate replacement or validity correction.
Mixed categorical update uncertainty uses logit-softmax (tau1, epsilon1e-4
interior smoothing); geometry includes standardized categorical coordinates.
Support domain uses exact simplex projection with affine normalization; evaluate
hard-category projection separately without silently replacing relaxed output.
Rebuild official baseline adapters for zero-hidden-layer logistic wrappers and
account for standardized one-hot transforms in native ElliCE. Unsupported mixed
ROAR/PROPLACE implementations get explicit status, never continuous results
presented as legal categorical CFs. Baseline grids in appendix_tabular.json;30s
per factual cap recorded with failed rows retained, not fabricated as success.
DiCE official0.12 random generator, four candidates, sample_size5000, minimum
valid normalizedL2 selection; no handcrafted substitute. Train/calibration and
model loading excluded from10CF runtime; CPU baseline and GPU GRACE labeled.

Categorical execution clarification (user authorized PDF treatment or better
preprocessing): retain the audited train-only encoded/standardized representation;
use A.5 logit-softmax training perturbations, and weighted simplex projection for
support updates in standardized coordinates. Score hard projection separately.
Tail resampling moves (x,y) pairs together: the old helper accidentally kept labels
in their old row order. This explicit correction is part of the new diagnostic
protocol, not a retrospective change to historical runs. Mean and scale follow
existing definitions; repeated sign/scale shifts will be counted and disclosed.
Preflight v2 completed on all8 GPUs. Official DiCE runs after isolated jsonschema
installation and float64 pandas input compatibility. PROPLACE unnamed output
columns are positional. MLP PROPLACE has observed timeouts and restricted-Gurobi
license failures; these are recorded, not replaced with another algorithm.

Separate hard-category diagnostic, declared before held-out scoring: for German
and mixed FICO, perturb each training categorical variable independently with
probability .05/.10/.20, replacing it by a train-empirical hard category; retain
numerical features and labels. Five derived realizations per probability, seed
300000+100*level+realization. Apply the same fixed10-step update, after all CFs are
frozen, and evaluate relaxed and separately hardened supports. These15 models
are an extra diagnostic, never calibration inputs or replacements for the45
main numerical-shift scenarios. Label preservation here defines an intervention
stress test, not a causally validated joint-data shift.
