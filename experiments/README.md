> **Research workflow archive.** For the paper overview and current entry points,
> start with the [project README](../README.md) and [reproduction guide](../docs/reproducibility.md).
> The records below include earlier experiments and retain their original provenance.

# Repository-local core experiments

## GENKI editor capacity and target ablation

`tasks/010_genki_expanded_editor.md` specifies the matched 4/8/16-channel by
0.5/0.7-target experiment. Run `049_run_genki_expanded.py` with a new external
`genki_expanded_editor/<run_id>` output path; workers use `experiments/run.py`.
`050_report_genki_expanded.py` evaluates the frozen PNGs uniformly and produces
all-case figures. Configuration: `configs/genki_expanded_editor.json`.
The original four channels are retained, with training-only nested discovery
of additional channels. All six arms share 16-D geometry and identical feasible
search settings. The classification threshold remains 0.5 in every evaluation.
Large raw proposals remain external; never resume after changing core source
or overwrite completed runs.

## GENKI-head GRACE with fixed face content

`tasks/007_genki_grace_identity.md` runs actual finite-bank GRACE using the
GENKI-supervised head, newly generated nominal Base CFs, new pixel-side GENKI
attacks, and new shifted-training future models. The controller is
`041_run_genki_grace.py`; report/audit is `042_report_genki_grace.py`; configuration
is `configs/genki_grace_identity_v1.json`. Every worker uses `experiments/run.py`
and an isolated `genki_grace_identity_v1/<run_id>` external output directory.

Only four fixed post-affine style channels move; all other factual styles and
the input transform stay fixed. Exact W2 geometry is evaluated in standardized
task coordinates. All returned candidates must pass the existing ArcFace,
FaceNet, LPIPS and outside-region RMS limits. Candidate proposals use the
existing geometry-only solver and are ranked by max_model mean_CF hinge;
nominal-validity repair is not used. The finite candidate search and projected
reference-label limitations are documented in the task and result report.

The base/head assets and prior runs remain external and immutable. This is a
generated-face experiment; it does not invert real GENKI photographs. The
final qualitative PDF includes every fixed test factual, Base and selected
GRACE output, with an additional figure for all geometry budgets.

## GENKI-4K smile head on FFHQ features

`tasks/006_genki_ffhq_classifier.md` uses official GENKI smile labels to train a
binary linear head over the frozen original CLIP image encoder. The new module
`genki_ffhq_classifier_v1` runs through the local launcher and has its own external
output root. Controller: `038_run_genki_ffhq.py`; independent report:
`039_report_genki_ffhq.py`. Configuration: `configs/genki_ffhq_classifier_v1.json`.

The differentiable `lift_ce.ffhq.genki_classifier.GenkiSmileClassifier` accepts
native generator RGB in [-1,1] and returns [smiling, not_smiling] logits. Use the
selected head checkpoint from the raw run. It shares the FFHQ image encoder and
preprocessing but does not evaluate text prompts. Old CLIP-conditioned flows,
Base CFs and robustness banks are not compatible results for this new classifier.
GENKI photographs are not inverted into StyleGAN by this classifier experiment.

Three focused tests:

```sh
PYTHONPATH=experiments/src /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python -m unittest discover -s experiments/tests -p 'test_genki_classifier.py'
```

The existing source audits separately check full CLIP input gradients. Follow
the result report for the preserved float32 CUDA check failure and supplemental
float64 numerical check; do not rerun a completed model-training phase to repair
an evaluation-only diagnostic.

`experiments/src/lift_ce/` is the authoritative source for the imported
HELOC/DIABETES workflows. Make future core changes and run commands from
`/home/xu.zhao/iclr_robust_ce/`. The source in `/home/xu.zhao/robust_ce/`
remains an unchanged historical reference. This is a scoped migration of the
core workflows and their import dependencies, not every prototype in that tree.

## Commands

Use the existing environment; no global editable-package replacement is needed:

```bash
cd /home/xu.zhao/iclr_robust_ce
PY=/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python
$PY experiments/run.py list
$PY experiments/run.py check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY experiments/run.py test
$PY experiments/run.py run single_budget_geodesic_wro_v1 -- --help
```

The launcher explicitly loads this checkout's `lift_ce`, `robustx` and `ellice`.
It does not execute the old checkout. `pyproject.toml` declares core dependencies;
vendored baseline source and licenses live under `vendor/`. Optional baseline
execution also needs solver packages/licensing and LIME. The current environment
is recorded in `results/003_core_migration_environment.json`. Do not install the
historical vendor dependency pins blindly into an existing environment.

New large outputs belong outside Git, under
`/home/xu.zhao/robust_ce_results/iclr_runs/<unique_run_id>/`. For example, this
**dry run writes nothing and starts no experiment**:

```bash
$PY experiments/run.py run single_budget_geodesic_wro_v1 \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/example_new_run \
  --dry-run -- prepare --device cpu
```

Remove `--dry-run` only when starting an intended experiment. Later phases of
that same run require `--resume` with the same output path. Resume checks the
repository, experiment module, and source fingerprint. Preserve the phase,
worker, device, and configuration arguments in each run's resolved metadata.
The existing worker flags are unchanged; GPU workers must be launched explicitly.

Use the launcher for execution. Imported dependency modules include historical
entry points that are not supported standalone launch targets and may retain old
output defaults. The ten supported core CLIs also require explicit `--output`
when invoked directly, but only the launcher enforces the protected output root.
`fixed_instance_stability_audit` is retained as historical source, not exposed as
a launch target.

## Frozen inputs and provenance

Historical checkpoints, populations, perturbations and raw results remain at
`/home/xu.zhao/robust_ce_results/`. Downloaded data remains under
`/home/xu.zhao/robust_ce_data/`. These are external data dependencies, not source
code dependencies. Existing drivers intentionally retain their frozen parent-run
references. A fresh machine needs those artifacts staged at the declared paths;
this migration does not make the historical datasets/checkpoints portable.

`provenance/core_import.json` records every imported file, original and local
SHA-256, vendor revisions/local changes, and administrative edits. Historical
HELOC/DIABETES YAMLs are copied verbatim into `configs/historical/`; they are not
newly resolved configurations for the fixed-instance experiments. Those drivers
retain their predeclared settings and save resolved configs in each run.

No models, datasets or experiment results were regenerated. No scientific
algorithm, production default, frozen bank or manuscript claim was changed.
Task 002 remains pending its missing objective/representation specification.

## Task 003: favorable-region direct WRO

`src/lift_ce/experiments/favorable_region_geodesic_wro_v1.py` is an isolated
HELOC experiment. It optimizes the control distribution on favorable graph
nodes using true population-level bank hinge and an exact discrete graph-OT
budget check. It does not use task 002's path objective or a validity penalty.
The predeclared configuration is `configs/favorable_region_geodesic_wro_v1.yaml`.

The new graph audit is at
`/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_graph_audit/`.
The full run is at
`/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1/`.
Historical data, models and controls remain external references.

The initial preparation command, already executed for this run, is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/run.py run \
  favorable_region_geodesic_wro_v1 \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1 \
  -- prepare
```

`experiments/012_run_favorable_task.py` orchestrates smoke, caching, optimization,
path freezing, fresh attacks and matched evaluation, and updates `state.yaml`.
It uses the four exposed devices (physical IDs 1, 2, 3, 5). Do not start a second
controller for an active run. Per-phase commands are recorded in process logs;
all phases support the local launcher with `--resume` and unchanged code.
`experiments/013_review_favorable_task.py` performs independent completion
checks and writes only a lightweight check summary into `results/`.

## Task 004: continuous WRO with minimal validity restoration

The isolated entry point is `minimal_validity_repair_wro_v1`; its core is
`src/lift_ce/counterfactuals/minimal_validity_repair.py`. The raw optimizer receives
no current model and retains the existing continuous Euclidean OT geometry.
Only invalid frozen supports are restored, using first-crossing search along
their corresponding target segment. Valid supports are unchanged. Both current
validity and exact post-repair geometry are checked independently.

Configuration: `configs/minimal_validity_repair_wro_v1.yaml`.
Main run:
`/home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2/`.
Initial smoke-only artifacts remain in sibling `20260916_minimal_v1/`.

Initial command (already executed):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/run.py run \
  minimal_validity_repair_wro_v1 \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2 \
  -- prepare
```

Controller: `experiments/015_run_minimal_repair_task.py`. It records exact commands
and environment settings in `orchestration/commands.jsonl`, updates `state.yaml`,
and uses four exposed GPUs. Do not launch duplicate controllers.
Independent completion review: `experiments/016_review_minimal_repair_task.py`.
The separately labeled historical final-output repair diagnostic receives no
own targeted fresh search and is not the primary learned-control path.

## Task 001: MNIST 4→9 with style shifts (GRACE)

User-confirmed task identity: existing MNIST with style shifts, not a separate
StyleMNIST dataset or six classifier IDs. Frozen SmallCNN and AE64 weights remain
external; see `results/008_stylemnist_asset_inventory.md`.

Predeclared config: `configs/stylemnist_grace_v1.json`.
Run: `/home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1/`.
This task has its own allowed external output root in `run.py`.

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/017_run_stylemnist.py \
  --output /home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1
```

The controller uses one worker per physical GPU 1,2,3,5, phase barriers, immutable
configuration/source identity, and completion markers. Do not launch duplicate
controllers; rerunning after the controller exits skips completed jobs. Every
worker runs through the repository launcher. Preparation, attack, optimization,
freeze, future and report are independently resumable phases. The raw solver
never receives f0; no validity restoration is called. The final candidate pool is
selected against the same complete attack bank before future evaluation starts.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/018_review_stylemnist.py \
  --output /home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1
```

The report explicitly distinguishes original base images from their encoder/decoder
realization, describes truncated attack hypergradients, and labels the proposed
severity/update extension and paper-inspired baselines. No latent validity is used
as a substitute for decoded-image classifier evaluation.

Completed-run interpretation (after the independent review):

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/019_summarize_stylemnist.py \
  --output /home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1
```

Results: `results/009_stylemnist_grace_v1/`; PDFs:
`figures/stylemnist_grace_v1/`. The report discloses the Base round-trip gap,
beta1's tolerance-sized geometry excess, and nondeterministic decoder kernels.

## Task 001 Galaxy: original multiclass LeapFactual assets

Task: `tasks/001_galaxy_task.md`. Assets are inventoried in
`results/010_galaxy_asset_inventory.md`; numerical strict-load/gradient checks are
in adjacent JSON files. The original VGG20, WAE64 and conditional flow are reused,
with inference source provenance in `provenance/galaxy_import.json`.
Ten target-label tasks preserve the original multiclass classifier.

Configuration: `configs/galaxy_grace_v1.json`. Additional original-ODE runtime
dependencies are pinned in `requirements-galaxy.txt` and installed with `--no-deps`
into `/tmp/galaxy_grace_deps` for this run (the existing environment supplies
PyTorch/torchvision/scipy and related dependencies). Recreate that directory on a
fresh machine and set PYTHONPATH accordingly; do not retrain existing models.

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/020_run_galaxy.py \
  --output /home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_full_v1
```

The controller uses physical GPUs0–7, one worker per GPU, a12.8GiB PyTorch allocator
cap, per-job directories, completion markers, source/config identity checks and
phase barriers. It updates `state.yaml`. Do not start a duplicate active controller.
After an exited controller is resumed, successful jobs are skipped. Each worker
runs through `experiments/run.py`; the Galaxy-specific external root is allowed
without weakening output protections for other experiments.

`021_report_galaxy.py` independently verifies persisted models/perturbations,
configuration selection, exact latent geometry, costs, all future-validity values,
shared-bank risks and freeze ordering, then creates publication PDFs/PNGs and CSVs.
Results will be stored in `results/011_galaxy_grace_v1/` and
`figures/galaxy_grace_v1/`. Calibration and future stress are expressly new protocols;
all adaptation/representation limits are disclosed in the final report.

### FFHQ generated-face GRACE

Original StyleGAN3/CLIP/flow inference assets are audited in `results/013_ffhq_asset_inventory.md`. This is generated FFHQ-style data with CLIP pseudo-labels; the fixed update adapts the CLIP visual projection only. See `configs/ffhq_grace_v1.json` and `requirements-ffhq.txt`. Original pure inference sources are preserved under `vendor/ffhq`; provenance is `provenance/ffhq_import.json`. Install dependencies in `/tmp/ffhq_grace_deps` alongside `/tmp/galaxy_grace_deps`; isolated CUDA build artifacts are in `/tmp/ffhq_cuda` and `/tmp/ffhq_torch_extensions`. The official CLIP checkpoint has its full checksum recorded in the asset checks; SqueezeNet LPIPS uses the official torchvision backbone cached under `/tmp/ffhq_torch_hub`.

Launch: `/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/024_run_ffhq.py --output /home/xu.zhao/robust_ce_results/ffhq_grace_v1/<new_run_id>`.
The controller invokes `experiments/run.py`, queues independent jobs across all8 GPUs, resumes only matching source fingerprints, updates state, and runs `025_report_ffhq.py` for independent validation before completion.

### Task002: FFHQ frozen-content bottleneck realization

`tasks/002_face_baseline.md` is implemented by
`src/lift_ce/experiments/ffhq_grace_identity_v1.py` and
`src/lift_ce/ffhq/bottleneck.py`. Configuration:
`configs/ffhq_grace_identity_v1.json`. The source FFHQ run is a read-only parent;
its generated IDs, Base CFs, attack bank and shifted-update checkpoints are reused.
The fallback decomposition is a training-only, uncentered score-gradient SVD with
95% energy. Only task coordinates are optimized, paired with each original
factual's fixed complementary content. There is no identity penalty or
current-validity restoration branch. `Conditional_Base` exposes any image change
caused by the restricted realization itself.

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/028_run_ffhq_identity.py \
  --output /home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1
```

The controller attaches eight GPUs, caps each worker's PyTorch memory at14% of
the device, and runs workers through the repository launcher. Resume skips
completed jobs only with matching source/configuration. Do not run duplicate
controllers. Report/checks: `029_report_ffhq_identity.py`; repository summaries:
`results/015_ffhq_identity_v1/`; figures: `figures/ffhq_grace_identity_v1/`.

New identity evaluation uses official InsightFace buffalo_l detector and ArcFace
weights stored outside Git; see the inventory and identity-model manifest for
source URL, hashes and alignment. Isolated dependencies in
`/tmp/ffhq_identity_deps` precede the existing FFHQ dependency directories.
Install the exact versions in `requirements-ffhq-identity.txt` with `--no-deps`
into that directory; existing PyTorch, scipy, sklearn and OpenCV are reused.
The final qualitative figures show all eight prespecified IDs.

### FFHQ local smile-editor calibration screen

Task `tasks/003_face_editor_screen.md` implements the user-approved next step
following the unsuccessful gradient-bottleneck experiment. Core source:
`src/lift_ce/ffhq/style_editor.py` and
`src/lift_ce/experiments/ffhq_smile_editor_v1.py`. Config:
`configs/ffhq_smile_editor_v1.json`. It changes only1/4/8 post-affine modulation
channels, freezing all other channels and the factual synthesis-input transform.
Training-only ranking uses smile sensitivity and outside-region Jacobian probes.
The eight calibration factuals receive a fixed six-strength sweep, measured by
original CLIP, ArcFace, independent FaceNet, LPIPS and outside-region RMS.
Final test data and future models are excluded from the screen.

Controller: `032_run_ffhq_smile_editor.py`; report: `033_report_ffhq_smile_editor.py`.
Eight-GPU run: `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3`.
Its completed setup/discovery outputs are read-only references to v1; v1/v2
failed at legacy FaceNet checkpoint loading before producing calibration metrics.
Every checkpoint key is now strictly loaded with mmap disabled, including the
classification head, then embedding mode is enabled. Earlier runs are preserved.

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/032_run_ffhq_smile_editor.py \
  --output /home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3 \
  --reuse-discovery-from /home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v1
```

The optional reuse flag creates a fresh source-fingerprinted run and records all
reused paths; it does not replace the original run marker or rewrite old jobs.
Additional dependency: `facenet-pytorch==2.6.0 --no-deps --target /tmp/ffhq_editor_deps`;
the existing FFHQ and identity dependencies remain unchanged. FaceNet VGGFace2
checkpoint is external, with source and checksum in `facenet_manifest.json`.

### Extended local-editor screen and independent coefficients

Task `tasks/004_face_editor_extension.md` extends the frozen scalar controls to
alpha6/8/12/16, without relaxing any preservation gate. If fewer than6/8 calibration
factuals pass,4/8 selected standardized style coordinates are optimized independently
with40 constrained gradient-ascent steps. Actual ArcFace/FaceNet/LPIPS/locality
measurements gate every accepted iterate. This is editor construction, not GRACE.
Previous scalar images are reused; final FFHQ test and future models are excluded.

Run: `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v2/20260917_extended_v1`.
Controller: `035_run_ffhq_smile_editor_v2.py`; worker:
`src/lift_ce/experiments/ffhq_smile_editor_v2.py`; report:
`036_report_ffhq_smile_editor_v2.py`; config: `configs/ffhq_smile_editor_v2.json`.

An independent LeNNon CelebA smile classifier is evaluated diagnostically with
its original100px RGB preprocessing and argmax>0 label mapping. The released head
has10 logits; this is disclosed, and extra-class probability/predictions are saved.
Its whole-module checkpoint is converted with weights-only loading using audited
class allowlists, then a strict state-dict loader is used. Source revision and
checksums are frozen in the raw run. The32-example annotated CelebA check is a
small source-data sanity test with unknown training overlap, not a generalization
estimate. This external sample is unrelated to the FFHQ final test set.


## MNIST update-aware latent method (tasks011/012)

The authoritative method is PDF Section3 plus the user's smooth-hinge clarification.
`counterfactuals/update_aware.py` differentiates the prescribed full T-step update,
using empirical-RMS dual normalization sqrt(N). New task012 explicitly passes
softplus hinge temperature0.05; the helper's default hard hinge remains for
historical compatibility only. Keep the Base generator's returned `CEResult.latents`
with its image: encoding the image again is not equivalent.

Task011 and the invalid intermediate task012 attempts are preserved diagnostics.
Run the corrected task with `056_run_mnist_latent_base_fix.py --output <new iclr_runs
subdirectory>`; it uses `experiments/run.py`, eight GPUs, original future checkpoints,
and separate calibration selection. `057_report_mnist_smooth.py --run <run>` writes
summary CSVs and PDFs. Config: `configs/mnist_latent_base_fix.json`. Result directory:
`results/026_mnist_smooth_hinge/`. A small successive objective change is the algorithm's
stopping criterion, not a proof of a stationary or globally optimal solution.

## Galaxy update-aware directed pairs (task017)

`tasks/017_galaxy_update_aware.md` reuses the original Galaxy VGG/WAE and actual
nominal Base generator codes for Merging→Round and Round→Cigar. Config:
`configs/galaxy_update_aware.json`. This uses the revised smooth update-aware
objective, not the old Galaxy finite-bank solver.

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/069_run_galaxy_update_aware.py \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/<new_run_id>
```

The controller attaches eight GPUs and executes each phase through `run.py`.
It requires a fresh output root. `--audit-only` stops after preparation and both
Galaxy derivative checks; such outputs remain separate from full experiments.
Do not rerun a completed root or launch a second controller for a live run.

The prescribed model update changes only the last linear layer, for three
full-batch SGD steps on40 images. Pixel uncertainty is still measured through
the complete frozen image feature extractor. The custom pullback-norm operator
streams Jacobian actions and implements its exact first derivative with respect
to the feature adjoint; it does not promise higher-order derivatives through
that operator or updates of the frozen feature extractor. Its focused tests are
`tests/test_frozen_feature_sensitivity.py`.

After the controller exits successfully:

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/070_report_galaxy_update_aware.py \
  --run /home/xu.zhao/robust_ce_results/iclr_runs/galaxy_update_aware/<run_id>
```

The reporter uses a GPU to re-decode supports and independently recompute future
predictions, verifies exact OT/assignment/cost/selection, and writes all-case
PDFs, metrics and the result report. Original TF32 images and float32 replays
are scored separately. The best feasible iterate is retained, without any
current-validity gate; this is not a convergence or global robustness certificate.

## FFHQ original-CLIP update-aware full-W run (task018)

Task018 uses original CLIP prompts and full W512, following the user's explicit
removal of fixed-content constraints. Config `configs/ffhq_update_aware.json`;
core `src/lift_ce/experiments/ffhq_update_aware.py`; controller072; reporter073.
The original StyleGAN3/CLIP/nominal Base codes remain external and immutable.
Dependencies are the existing `/tmp/ffhq_grace_deps` and repository FFHQ vendor
sources; no model download or retraining of the original classifier is required.

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/072_run_ffhq_update_aware.py \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/ffhq_update_aware/<new_run_id>
```

Use `--audit-only` for a separate fresh preparation/gradient-audit root. Workers
run through the source-fingerprinted launcher, on eight GPUs with6.5% memory caps.
Do not restart a live controller or edit core source while its workers run.
The complete prescribed update is three projection-SGD steps, not the historical
AdamW bank. Classifier labels remain the original prompt-relative pseudo-labels.
Both pixel sensitivity and the first Q derivative use exact streamed chain rules.

After the full controller succeeds:

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/073_report_ffhq_update_aware.py \
  --run /home/xu.zhao/robust_ce_results/iclr_runs/ffhq_update_aware/<run_id>
```

The GPU reporter independently decodes W supports and rescoring uses every saved
future projection. It checks OT, assignment, costs, selection/freeze ordering and
hashes, then writes all-case and ranked heatmaps and completion state. Heatmaps
are abs(r.25-r0) and abs(r0-Base) at matched beta, using a shared RGB224 scale.
No identity/background constraints, content editing mask or validity repair are
imposed; avoid interpreting classifier validity as human smile/identity evidence.

## Text toxicity discrete adaptation (task019)

Task019 uses a frozen MiniLM sentence representation, fitted logistic head and
fitting-only FLAN-T5-small rewrites. It searches actual sentence-codebook supports;
no gradient through token decoding or arbitrary-embedding inverse is assumed.
Controller076 completed all phases on the original unique external run. Do not
rerun it. Independent reporter081 verifies raw outputs without modifying them:

```bash
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/081_report_text_toxicity.py
```

It is intentionally guarded against overwriting completed verification. Report:
`results/033_text_toxicity_update_aware/report.md`. Frozen source/config/task is
in `experiments/snapshots/text019_v1/`; dependencies in the resume record. This
pilot has saturated test FV for Base and all methods and establishes no robustness
improvement. All-eight PDF/HTML explicitly show unchanged sentences.

## Controlled population-versus-individual adversary ablation (task102)

`mixed_update_aware.mixed_surrogate` and the existing `appendix_tabular.grace`
accept `adversary_mode='population'` (unchanged default) or `'individual'`, with
`sensitivity_chunk_size=1`. Individual mode averages empirical-RMS dual norms
of per-CF training-input pullbacks through one shared differentiable SGD unroll.
It does not split geometry into single-CF optimizations.

The isolated `population_individual_adversary` runner reuses Task100's frozen
MLP, train/reference data, stored Wachter CFs and exact45 future checkpoints.
The seeds below subsample populations; they are not classifier-training seeds.
Each run must use a fresh external output path.

```bash
PY=/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python
$PY experiments/106_run_adversary_ablation.py --stage debug \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/NEW_DEBUG
$PY experiments/106_run_adversary_ablation.py --stage main \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/NEW_MAIN
$PY experiments/107_report_adversary_ablation.py \
  --run /home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/NEW_MAIN \
  --output results/102_population_individual_adversary/NEW_REPORT
```

A single explicitly configured job uses the normal launcher:

```bash
CUDA_VISIBLE_DEVICES=3 $PY experiments/run.py run population_individual_adversary \
  --output /home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/NEW_SINGLE \
  -- --dataset german --seed 3 --n 32 --r-ratio 3 --beta 1.1 \
  --adversary-mode individual --sensitivity-chunk-size 1
```

Tests (include local vendored imports for the legacy-optimizer regression):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/task102_mpl \
PYTHONPATH=experiments/src:experiments/vendor/RobustX:experiments/vendor/ellice \
$PY -m unittest discover -s experiments/tests -p test_sensitivity_aggregation.py
```

Do not relaunch completed directories. Task/config/report preserve the r0 HELOC
identity control separately from the predeclared positive-radius diagnostic.
