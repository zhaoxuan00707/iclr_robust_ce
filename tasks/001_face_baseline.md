Implement and run an FFHQ extension of the current GRACE method by reusing the
existing face / smiling-face assets from:

    /mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/

The target recourse task is expected to be:

    non-smiling face -> smiling face

but DO NOT assume the exact classifier/task definition from folder names.
Audit the existing LeapFactual code and checkpoints first.

Use the current GRACE method WITHOUT the current-validity repair/check branch.

Use all available GPU servers / GPUs safely and in parallel.

==================================================
0. AUDIT EXISTING FFHQ / FACE ASSETS FIRST
==================================================

Before editing code or launching expensive runs, inspect:

    /mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/

Locate all assets related to:

    FFHQ
    faces
    smiling / smile attribute
    latent-space editing
    encoder
    decoder / generator
    classifier
    counterfactual generation

Document:

1. Exact FFHQ dataset / subset used.
2. Exact train / calibration / validation / test splits.
3. Image resolution and preprocessing.
4. Smile-classifier architecture and checkpoint.
5. Class convention:
       smiling = ?
       non-smiling = ?
6. Decision threshold tau.
7. Encoder checkpoint.
8. Decoder / generator checkpoint.
9. Exact latent representation:
       z / w / w+ / VAE latent / other.
10. Latent dimensionality.
11. Any latent normalization.
12. Existing face-to-smile CF pipeline.
13. Existing latent interpolation / editing code.
14. Existing factual IDs and target/reference IDs.
15. Existing future-shift / retraining code.
16. Existing SLURM / multi-GPU launch infrastructure.

Create:

    ffhq_asset_inventory.md

with exact paths and compatibility notes.

Do NOT retrain any good existing classifier / encoder / decoder / generator
unless no compatible checkpoint exists.

==================================================
1. PRESERVE THE CURRENT GRACE FORMULATION
==================================================

Use exactly the current GRACE formulation.

For base classifier f_0 and base CF generator G:

    z_i = G(x_i)

and define:

    P_CF

as the empirical distribution of base face-to-smile counterfactuals.

Let:

    P_1

be the empirical distribution of favorable / smiling reference faces.

Data-side uncertainty:

    U_r
      =
      {
        P :
        W_data(P, P_tr_hat) <= r
      }.

Each shifted training distribution induces:

    f_P = Update(f_0, P).

Use invalidity loss:

    ell_f(z) = [tau - f(z)]_+.

The ideal robust risk remains:

    Phi_r(Q)
      =
      sup_{P in U_r}
      E_{z~Q} ell_{f_P}(z).

The practical finite-bank approximation remains:

    Phi_hat_r(Q)
      =
      max_{f in A_r}
      E_{z~Q} [tau - f(z)]_+.

IMPORTANT:

    max_model mean_CF hinge

NOT:

    mean_CF max_model hinge.

==================================================
2. PRESERVE THE GEOMETRY BUDGET
==================================================

Let:

    D = W_CF(P_CF,P_1).

Set:

    B_geo = beta_geo * D.

Use:

    G_B
      =
      {
        Q :
        W_CF(P_CF,Q)
        +
        W_CF(Q,P_1)
        <= B_geo
      }.

Use:

    beta_geo in {1.00, 1.05, 1.10, 1.20}.

Interpretation:

    beta_geo = 1
        shortest / zero-detour geometry

    beta_geo > 1
        bounded robustness-driven detour.

Do NOT add:

    current-validity repair
    current-validity penalties
    waypoint penalties
    spline penalties
    extra target penalties.

==================================================
3. RUN GRACE IN THE EXISTING FACE LATENT SPACE
==================================================

Reuse the validated LeapFactual latent representation.

Do NOT assume whether this is W, W+, VAE latent, etc.
Determine it from the code.

Let:

    h_i = Enc(x_i)

or use the corresponding existing inversion representation.

Represent:

    P_CF^h
    P_1^h
    Q^h

in the same latent space.

Optimize latent support coordinates:

    Q^h = sum_j b_j delta_{h_j}.

Decode / generate images only when classifier evaluation or image metrics are
needed.

If the existing pipeline uses:

    Generator(h)

rather than an explicit Decoder, preserve that interface.

==================================================
4. DEFINE W_CF CONSISTENTLY
==================================================

First inspect whether LeapFactual already defines a latent-space distance.

IF a validated latent metric already exists:
    reuse it.

OTHERWISE use:

    d_CF(h,h') = ||h-h'||_2.

Use the SAME CF-side metric for:

    D
    W_CF(P_CF,Q)
    W_CF(Q,P_1).

Do NOT mix latent-space and pixel-space Wasserstein distances.

Recourse cost must remain separate.

==================================================
5. BASE COUNTERFACTUAL POPULATION
==================================================

Reuse the existing LeapFactual face-to-smile CF generator if possible.

The base CF generator must be frozen before GRACE.

For each fixed factual face x_i:

    base CF = G(x_i)

under the existing smile target.

Use exactly the same base CF population for every:

    r
    beta_geo
    attack-bank run.

Save:

    factual_ids.npy
    factual_images.*
    base_cf_images.*
    base_cf_latents.*

Do not regenerate base CFs independently for different beta values.

==================================================
6. TARGET ANCHOR P_1
==================================================

Construct P_1 from smiling training faces.

Use the exact classifier/task convention found during asset audit.

Use a fixed target-reference subset if needed for computational reasons.

Selection must be frozen before final future-shift evaluation.

Save:

    target_ids.npy
    target_latents.npy

==================================================
7. ATTACK BANK
==================================================

Reuse the current GRACE attack-bank implementation.

The adversary acts on TRAINING DATA rather than directly perturbing the
classifier parameters.

Each feasible shifted distribution:

    P in U_r

must produce:

    f_P = Update(f_0,P).

Persist every accepted full-update model.

Construct:

    A_r.

Every candidate Q must be evaluated against the full compatible shared bank.

Do not use only source-specific adversaries.

==================================================
8. MODEL UPDATE
==================================================

Reuse an existing LeapFactual smile-classifier fine-tuning / retraining
procedure if available.

Otherwise define one fixed protocol before experiments.

Freeze:

    optimizer
    learning rate
    epochs
    batch size
    augmentation
    initialization
    seed policy.

Do not change Update settings across beta_geo.

Do not directly perturb network weights for the primary GRACE uncertainty
experiment.

==================================================
9. CALIBRATE r AND beta_geo
==================================================

Use calibration data only to select:

    r_star
    beta_geo_star.

Use a predefined grid for r based on the existing data-side reference scale.

Use:

    beta_geo in {1.00,1.05,1.10,1.20}.

Record:

    calibration score
    Phi_hat_r
    recourse cost
    selected r
    selected beta.

Save:

    calibration_results.csv
    selected_configuration.json

The final report MUST explicitly show:

    r_star
    beta_geo_star.

Do NOT recalibrate per final test shift severity.

==================================================
10. FFHQ FUTURE MODEL SHIFTS
==================================================

First inspect LeapFactual for an existing FFHQ future-shift protocol.

IF one exists:
    reuse it.

IF none exists:
    construct a small, predefined EXTERNAL stress suite based on realistic
    image / acquisition distribution changes.

Possible candidates, ONLY if compatible with the existing pipeline:

    brightness / exposure
    contrast
    blur
    image noise
    color-temperature / channel shift
    crop / scale
    mild pose distribution shift
    latent style shift

Prefer 3 distinct shift families rather than many redundant ones.

For each shift family, use ordered severity levels.

The protocol must be:

    shifted training images
        -> fine-tune / update f_0
        -> future classifier.

The final counterfactual population must be frozen BEFORE evaluating these
future models.

IMPORTANT:

External FFHQ stress severity is NOT the same as GRACE internal radius r.

==================================================
11. PRIMARY FFHQ EXPERIMENT
==================================================

Run:

    Base CF
    GRACE beta=1.00
    GRACE beta=1.05
    GRACE beta=1.10
    GRACE beta=1.20.

For each configuration report:

    Future Validity
    Recourse Cost
    Phi_hat_r
    Runtime
    geometry feasibility.

Then report the calibration-selected:

    GRACE(r_star,beta_geo_star).

==================================================
12. FUTURE VALIDITY
==================================================

For future classifiers f^(k), define empirical Future Validity as the fraction
of refined faces still classified as smiling, averaged over:

    factual examples
    future models.

Use the same final future models for every configuration.

==================================================
13. RECOURSE / IMAGE QUALITY
==================================================

Do NOT use only latent-space W_CF as recourse cost.

Report separately:

1. Pixel-space L2 or the face-domain cost already used in LeapFactual.
2. Latent displacement.
3. Existing perceptual metric if LeapFactual already uses one.

If LPIPS / identity distance is already available in the old pipeline, reuse it.

For faces, also inspect whether the existing project contains an identity
preservation metric.

If an existing identity metric is available, report it as a secondary metric.

Do NOT invent a new identity metric solely for this experiment.

==================================================
14. IDENTITY PRESERVATION
==================================================

Because the task is:

    non-smiling -> smiling

the refined CF should ideally preserve person identity.

If LeapFactual already uses a face embedding / identity model:

    reuse it.

Report cosine similarity or the existing identity-preservation score between:

    factual image
    returned CF.

This is a secondary diagnostic, not part of the GRACE optimization objective.

==================================================
15. GEOMETRY-BUDGET ABLATION
==================================================

Produce:

    beta_geo
    Future Validity
    Cost
    Phi_hat_r
    Identity score (if available)
    Runtime

for:

    1.00
    1.05
    1.10
    1.20.

The central question is:

Does additional geometric freedom reduce finite-bank robust risk and improve
Future Validity while preserving realistic faces and identity?

==================================================
16. QUALITATIVE FIGURE
==================================================

Produce the FFHQ analogue of the successful StyleMNIST figure.

Columns:

    Factual
    Base
    beta=1.00
    beta=1.05
    beta=1.10
    beta=1.20

Use fixed factual examples chosen BEFORE final evaluation.

Do not cherry-pick based on final Future Validity.

Prefer a diverse fixed set of factual IDs from the test population.

Save:

    figures/ffhq_qualitative.pdf
    figures/ffhq_qualitative.png

Images should be large enough to inspect:

    smile intensity
    identity
    artifacts
    background changes
    pose changes.

==================================================
17. OPTIONAL LATENT TRAJECTORY VISUALIZATION
==================================================

If inexpensive, for a few fixed examples show intermediate decoded samples
along:

    Base -> beta-selected GRACE output

or the learned latent displacement.

This is visualization only.

Do not introduce a new trajectory constraint or optimization objective.

==================================================
18. SHIFT-SEVERITY CURVE
==================================================

If external future shifts have ordered severity:

Plot:

    x-axis:
        external FFHQ shift severity

    y-axis:
        Future Validity (%).

Use the SAME frozen:

    r_star
    beta_geo_star

for all severity levels.

Do not tune GRACE separately at each severity.

==================================================
19. RUNTIME
==================================================

Measure runtime starting from:

    pretrained f_0
    pretrained encoder / generator

and ending at final robust CF outputs.

Exclude:

    shared base-classifier pretraining
    generator / encoder pretraining
    final future-shift evaluation.

For GRACE include:

    base CF generation
    attack-bank construction
    population refinement
    calibration.

Report:

    total population runtime
    runtime per CF
    GPU-hours.

==================================================
20. MULTI-GPU / ALL SERVER EXECUTION
==================================================

Use all available GPU servers safely.

First record:

    hostname
    GPU type
    GPU count
    free GPU memory
    CUDA_VISIBLE_DEVICES
    scheduler availability.

Create:

    hardware_manifest.json

before launch.

Parallelize independent units over:

    r
    beta_geo
    attack restart
    external future-shift realization
    random seed

where safe.

Do NOT let multiple workers mutate the same:

    bank
    checkpoint
    CSV
    output directory.

Use per-job directories and merge afterward.

Use SLURM arrays if available.
Otherwise build a GPU-aware launcher.

All jobs must be resume-safe.

==================================================
21. BASELINES
==================================================

First inspect LeapFactual for existing compatible face counterfactual baselines.

If compatible robust-CF methods already exist, run them.

If the tabular robust-CF baselines from the main paper do not naturally support
high-dimensional FFHQ images, do NOT force artificial ports solely to fill the
table.

For FFHQ, the important extension is:

    Base CF vs GRACE

plus qualitative / robustness / identity evidence.

Document baseline compatibility honestly.

==================================================
22. OUTPUT DIRECTORY
==================================================

Write to:

    robust_ce_results/ffhq_grace_v1/<run_id>/

Required:

    ffhq_asset_inventory.md
    resolved_config.json
    dataset_manifest.json
    model_manifest.json
    hardware_manifest.json
    factual_ids.npy
    target_ids.npy
    selected_configuration.json
    base_cf_images.*
    base_cf_latents.*
    attack_bank_manifest.csv
    calibration_results.csv
    geometry_ablation.csv
    future_validity.csv
    image_quality.csv
    identity_metrics.csv
    runtime.csv
    shift_severity.csv
    qualitative_examples/
    figures/
    checkpoints/
    logs/
    report.md

Do NOT overwrite any LeapFactual assets.

==================================================
23. REQUIRED TABLES
==================================================

TABLE A — FFHQ main result

    Method
    Future Validity
    Recourse Cost
    Identity Score
    Runtime

TABLE B — Selected configuration

    Task
    r_star
    beta_geo_star

TABLE C — Geometry-budget ablation

    beta_geo
    Future Validity
    Cost
    Phi_hat_r
    Identity Score
    Runtime

==================================================
24. REQUIRED FIGURES
==================================================

FIGURE A — Qualitative refinement

    Factual | Base | beta=1.00 | beta=1.05 | beta=1.10 | beta=1.20

FIGURE B — Future Validity vs external FFHQ shift severity

Optional:

FIGURE C — latent-space visualization

only if meaningful and easy to interpret.

==================================================
25. SCIENTIFIC RULES
==================================================

Do NOT:

- add current-validity repair/checks;
- retrain good LeapFactual encoders / generators / classifiers unnecessarily;
- select checkpoints using final test outcomes;
- directly perturb classifier weights for the primary future-model protocol;
- equate external stress severity with r;
- equate latent W_CF with individual recourse cost;
- tune beta independently for each final test shift;
- cherry-pick qualitative examples using final Future Validity;
- claim global WRO optimality;
- claim identity preservation unless actually measured.

==================================================
26. FINAL REPORT
==================================================

At completion report:

ASSET REUSE
Exact classifier, encoder, generator/decoder checkpoints and paths.

TASK
Exact non-smiling -> smiling classifier definition.

LATENT SPACE
Representation type and dimension.

CALIBRATION
r_star and beta_geo_star.

MAIN RESULT
Future Validity, Cost, Identity, Runtime.

GEOMETRY ABLATION
beta=1.00 / 1.05 / 1.10 / 1.20.

QUALITATIVE RESULT
Whether increased geometry freedom improves robustness without obvious face
artifacts or identity drift.

SHIFT ROBUSTNESS
Behavior under stronger external FFHQ shifts.

COMPUTE
Servers, GPUs, wall time, GPU-hours.

FAILURES
Any incompatible checkpoints, failed jobs, attack-bank failures, or decoder
artifacts.

LIMITATIONS
Finite attack bank, local optimization, latent-representation dependence.

Do not make claims unsupported by the completed run.
