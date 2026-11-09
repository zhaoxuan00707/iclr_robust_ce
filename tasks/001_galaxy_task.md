Implement and run a GALAXY-domain replication of the current GRACE method,
reusing the existing Galaxy assets from the previous LeapFactual project:

    /mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/

The goal is to reproduce the current GRACE experimental protocol on the
Galaxy image dataset WITHOUT redesigning the method.

Use the current no-current-validity-check version of GRACE.

Use all available GPU servers / GPUs safely and in parallel.

==================================================
0. FIRST: AUDIT THE EXISTING LEAPFACTUAL GALAXY ASSETS
==================================================

Before changing code or launching expensive experiments, inspect:

    /mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/

and identify all Galaxy-related assets.

Find and document:

1. Galaxy dataset loader(s).
2. Exact prediction task:
   - class names,
   - binary vs multiclass,
   - favorable / target class,
   - train / validation / test split.
3. Image preprocessing:
   - crop / resize,
   - normalization,
   - channels,
   - augmentation.
4. Existing trained classifier checkpoint(s).
5. Existing encoder checkpoint(s).
6. Existing decoder / autoencoder / VAE checkpoint(s).
7. Latent dimensionality and latent normalization.
8. Reconstruction code and reconstruction quality.
9. Existing Galaxy counterfactual-generation code.
10. Existing LeapFactual latent manipulation / interpolation code.
11. Any existing future-shift / corruption / retraining pipeline.
12. Any existing classifier-update code.
13. Existing multi-GPU / SLURM launch infrastructure.
14. Existing factual IDs / target-class reference IDs that can be reused.

Do NOT assume the task definition from folder names.

Create:

    galaxy_asset_inventory.md

with exact paths, model classes, checkpoint names, shapes, preprocessing, and
compatibility.

Do not retrain classifiers / encoders / decoders if compatible high-quality
checkpoints already exist.

==================================================
1. PRESERVE THE CURRENT GRACE FORMULATION
==================================================

Use the current GRACE objective exactly.

For base classifier f_0 and base CF generator G:

    z_i = G(x_i)

and

    P_CF = empirical distribution of base CFs.

Let P_1 be the empirical target / favorable reference distribution.

Data-side uncertainty:

    U_r
      =
      {
        P :
        W_data(P, P_tr_hat) <= r
      }.

A shifted training distribution induces:

    f_P = Update(f_0, P).

The deployment robust risk is:

    Phi_r(Q)
      =
      sup_{P in U_r}
      E_{z~Q} [tau - f_P(z)]_+.

The practical finite-bank approximation is:

    Phi_hat_r(Q)
      =
      max_{f in A_r}
      E_{z~Q} [tau - f(z)]_+.

IMPORTANT:

    max_model mean_CF hinge

NOT:

    mean_CF max_model hinge.

==================================================
2. PRESERVE THE GEOMETRIC DECISION REGION
==================================================

Let:

    D = W_CF(P_CF, P_1).

Use:

    B_geo = beta_geo * D,

and:

    G_B
      =
      {
        Q :
        W_CF(P_CF,Q)
        +
        W_CF(Q,P_1)
        <= B_geo
      }.

Primary beta grid:

    beta_geo in {1.00, 1.05, 1.10, 1.20}.

Interpretation:

    beta_geo = 1.00
        -> zero-detour / shortest-geodesic restriction

    beta_geo > 1
        -> bounded robustness-driven detour.

Do NOT introduce waypoint penalties, spline penalties, delta/epsilon lens
parameters, or current-validity repair constraints.

==================================================
3. USE THE EXISTING GALAXY LATENT SPACE
==================================================

Prefer the existing LeapFactual encoder / decoder for Galaxy.

If the existing autoencoder / decoder reconstructs Galaxy images faithfully,
run the GRACE decision geometry in latent space.

For image x:

    h = Enc(x)

and:

    x_hat = Dec(h).

Define:

    P_CF^h = Enc_# P_CF
    P_1^h  = Enc_# P_1.

Represent the refined distribution as:

    Q^h = sum_j b_j delta_{h_j}.

The optimization variables are latent supports h_j.

Classifier evaluation must be performed on decoded images:

    x_j = Dec(h_j)

followed by:

    f(x_j).

Use the same frozen encoder / decoder throughout the whole experiment.

==================================================
4. DEFINE W_CF CONSISTENTLY
==================================================

Inspect LeapFactual for an existing latent metric.

IF a validated Galaxy latent metric already exists:
    reuse it.

OTHERWISE use Euclidean latent distance:

    d_CF(h,h') = ||h-h'||_2.

Then compute all of:

    W_CF(P_CF,Q)
    W_CF(Q,P_1)
    D

in the SAME latent geometry.

Do NOT mix:

    latent W_CF for one term

with

    pixel W_CF for another.

Keep recourse cost separate from W_CF.

==================================================
5. BASE COUNTERFACTUALS
==================================================

Reuse the existing Galaxy CF generator from LeapFactual if compatible.

Priority order:

1. Existing verified LeapFactual Galaxy CF pipeline.
2. Existing latent counterfactual optimizer.
3. Only if neither exists, implement the minimal latent CF generator needed
   for the current classifier.

Freeze the base CF generator before GRACE.

Use the exact same factual population for every beta_geo and every comparison.

Save:

    factual_ids.npy
    base_cf_images.npy
    base_cf_latents.npy

Do not regenerate different base CFs per beta.

==================================================
6. TARGET REFERENCE DISTRIBUTION
==================================================

Construct P_1 from target / favorable training samples under the exact Galaxy
task discovered in Step 0.

Use a fixed target-reference subset if the full target class is too large.

The target subset must be selected before final future-shift evaluation.

Encode:

    P_1^h = Enc_# P_1.

Save:

    target_ids.npy
    target_latents.npy

==================================================
7. DATA-SIDE ADVERSARIAL MODEL BANK
==================================================

Reuse the current GRACE adversarial-search implementation.

The adversary acts on TRAINING DATA, not directly on classifier parameters.

Each accepted shifted training distribution P must satisfy:

    W_data(P, P_tr_hat) <= r

up to the existing declared numerical tolerance.

Each accepted P produces:

    f_P = Update(f_0,P).

Persist every accepted full-update model.

Construct a shared attack bank:

    A_r.

Every candidate Q must be cross-evaluated against every compatible model in
A_r.

Do NOT use candidate-specific adversaries only.

==================================================
8. MODEL UPDATE RULE
==================================================

Inspect LeapFactual for the existing Galaxy classifier update / fine-tuning
procedure.

Reuse it if available.

Otherwise define ONE fixed update protocol before experiments.

Freeze:

    optimizer
    learning rate
    epochs
    batch size
    initialization
    augmentation
    random seed policy.

Do not change update hyperparameters across beta_geo.

==================================================
9. CALIBRATION OF r AND beta_geo
==================================================

GRACE has two method-level configuration parameters:

    r
        data-side future uncertainty

    beta_geo
        CF-side geometric freedom.

Use calibration data only to select:

    r_star
    beta_geo_star.

Do NOT use final future-shift outcomes for this selection.

Record the complete calibration grid and scores.

Save:

    calibration_results.csv
    selected_configuration.json

The report MUST explicitly state:

    r_star
    beta_geo_star.

==================================================
10. GALAXY FUTURE-SHIFT EVALUATION
==================================================

First inspect whether LeapFactual already contains a Galaxy future-shift
protocol.

IF a verified Galaxy shift protocol exists:
    reuse it exactly.

IF NOT:
    create a separate external stress-test protocol, but clearly mark it as
    newly introduced for GRACE evaluation.

Prefer realistic image / acquisition / style changes rather than direct weight
perturbations.

Potential Galaxy stress factors, ONLY if compatible with the dataset and old
project:

    brightness / exposure
    contrast
    Gaussian or Poisson imaging noise
    blur / PSF variation
    color-channel shift
    background intensity
    crop / scale
    rotation / orientation
    latent acquisition-style perturbation

Do not arbitrarily include all of them.

Choose a minimal, predeclared subset supported by the Galaxy pipeline.

The future-model protocol must be:

    shifted training data
        -> fixed Update(f_0, shifted data)
        -> future classifier.

Final CFs are frozen before these shifts are evaluated.

==================================================
11. PRIMARY EXPERIMENT
==================================================

For each valid Galaxy task, run:

    Base CF
    GRACE beta=1.00
    GRACE beta=1.05
    GRACE beta=1.10
    GRACE beta=1.20

Use the same:

    factual IDs
    target population
    f_0
    encoder / decoder
    attack bank protocol
    final future shifts.

Report:

    Future Validity
    Recourse Cost
    Phi_hat_r
    Runtime
    r_star
    beta_geo_star.

==================================================
12. RECOURSE COST
==================================================

Do NOT use the latent Wasserstein geometry as the only recourse-cost metric.

Report image-domain recourse separately.

At minimum report:

    pixel L2

between factual x_i and returned CF x_cf,i.

If LeapFactual already uses a perceptual / latent recourse metric, also report
that metric.

If available and validated, LPIPS / perceptual distance may be reported as a
secondary image-quality measure.

Clearly distinguish:

    W_CF
        distribution-level geometry

from:

    Cost
        factual-to-CF recourse magnitude.

==================================================
13. GEOMETRY-BUDGET ABLATION
==================================================

Generate:

    beta_geo
    Future Validity
    Cost
    Phi_hat_r
    Runtime

for:

    1.00
    1.05
    1.10
    1.20.

This is the direct Galaxy analogue of the HELOC geometry-budget ablation.

Check whether achieved finite-bank risk decreases as beta increases.

If not monotone, report numerical optimization behavior honestly.

==================================================
14. QUALITATIVE GALAXY FIGURE
==================================================

Generate a qualitative grid analogous to the successful StyleMNIST figure.

Use fixed factual IDs chosen BEFORE final evaluation.

Columns:

    Factual
    Base CF
    beta=1.00
    beta=1.05
    beta=1.10
    beta=1.20

Optionally add:

    nearest target reference

only if visually useful.

Use identical rows across all beta columns.

Do NOT cherry-pick based on future validity.

Save both:

    figures/galaxy_qualitative.pdf
    figures/galaxy_qualitative.png

The figure should make it possible to visually inspect whether increasing
geometry freedom introduces unrealistic galaxy morphology.

==================================================
15. IMAGE QUALITY DIAGNOSTICS
==================================================

Because this is an image-domain experiment, also compute basic plausibility
diagnostics.

At minimum:

    reconstruction error
    pixel-space cost
    latent-space displacement.

If LeapFactual already provides a perceptual quality / reconstruction metric,
reuse it.

Do not invent a new metric solely because it makes GRACE look better.

==================================================
16. SHIFT-SEVERITY CURVE
==================================================

If the external Galaxy stress protocol has ordered severity levels, produce:

    Future Validity vs external shift severity.

IMPORTANT:

external severity != internal Wasserstein radius r.

Use the calibration-selected:

    r_star
    beta_geo_star

for ALL final severity levels.

Do not recalibrate GRACE separately at each test severity.

==================================================
17. BASELINES
==================================================

Inspect whether LeapFactual already contains compatible robust-CF baselines.

If existing compatible implementations exist, reuse them.

Do NOT spend large engineering effort porting tabular-only methods if they are
not naturally compatible with Galaxy images.

If only Base CF vs GRACE is practical, this is acceptable for the Galaxy
extension provided the main tabular baseline experiment remains the primary
comparison.

Document all compatibility decisions.

==================================================
18. RUNTIME
==================================================

Report end-to-end robust-CF construction runtime starting from a pretrained
base classifier.

Exclude:

    shared base-classifier training
    final future-shift evaluation.

For GRACE include:

    base CF generation
    attack-bank construction
    latent population refinement
    calibration / operating configuration selection.

Report:

    total population runtime
    runtime per CF
    GPU-hours.

==================================================
19. MULTI-GPU / MULTI-SERVER EXECUTION
==================================================

Use all available GPU servers safely.

First detect:

    hostname
    nvidia-smi
    CUDA_VISIBLE_DEVICES
    GPU memory
    scheduler / SLURM state.

Create:

    hardware_manifest.json

before launching jobs.

Parallelize independent jobs over:

    task / class pair
    r
    beta_geo
    adversarial restart
    future-shift realization
    random seed

where safe.

Do NOT allow multiple processes to mutate the same:

    attack bank
    result CSV
    checkpoint
    output directory.

Use one directory per job and merge after completion.

If SLURM exists:
    use job arrays.

Otherwise:
    implement a GPU-aware launcher with one worker per GPU.

Every job must log:

    hostname
    GPU ID
    seed
    task
    r
    beta_geo
    start time
    end time
    git commit
    resolved config.

Make all jobs resume-safe.

==================================================
20. REPRODUCIBILITY
==================================================

Freeze and record:

    Galaxy dataset split
    factual IDs
    target IDs
    classifier checkpoint
    encoder checkpoint
    decoder checkpoint
    base CFs
    attack-bank models
    r grid
    beta grid
    Update hyperparameters
    external future-shift scenarios
    selected r_star
    selected beta_geo_star.

No final future-shift result may affect method selection.

==================================================
21. OUTPUT DIRECTORY
==================================================

Write to:

    robust_ce_results/galaxy_grace_v1/<run_id>/

Required outputs:

    galaxy_asset_inventory.md
    resolved_config.json
    hardware_manifest.json
    dataset_manifest.json
    model_manifest.json
    selected_configuration.json
    factual_ids.npy
    target_ids.npy
    base_cf_images.*
    base_cf_latents.*
    attack_bank_manifest.csv
    calibration_results.csv
    geometry_ablation.csv
    future_validity.csv
    runtime.csv
    image_quality.csv
    shift_severity.csv
    qualitative_examples/
    checkpoints/
    figures/
    logs/
    report.md

Do NOT overwrite old LeapFactual or GRACE results.

==================================================
22. REQUIRED TABLES
==================================================

TABLE A — Main Galaxy result

    Method
    Future Validity
    Cost
    Runtime

TABLE B — GRACE selected configuration

    Task
    r_star
    beta_geo_star

TABLE C — Geometry-budget ablation

    beta_geo
    Future Validity
    Cost
    Phi_hat_r
    Runtime

==================================================
23. REQUIRED FIGURES
==================================================

FIGURE A:
    Galaxy qualitative refinements

    Factual | Base | beta=1.00 | beta=1.05 | beta=1.10 | beta=1.20

FIGURE B:
    Future Validity vs external shift severity
    if ordered stress levels are available.

Optionally:

FIGURE C:
    2D latent visualization of
    P_CF, learned Q, and P_1

only if a faithful visualization can be produced without changing the method.

==================================================
24. SCIENTIFIC RULES
==================================================

Do NOT:

- add current-validity repair/checks;
- retrain existing good encoder / decoder / classifier unnecessarily;
- choose checkpoints based on final test Future Validity;
- identify external shift severity with r;
- identify W_CF with individual recourse cost;
- directly perturb model parameters for the primary future-model experiment;
- tune beta separately at final-test severity levels;
- silently change the Galaxy task from the LeapFactual project;
- invent missing asset metadata;
- claim exact WRO optimality.

==================================================
25. FINAL REPORT
==================================================

At completion, summarize:

ASSET REUSE
Exactly which LeapFactual classifier, encoder, decoder, and CF components were
reused, with paths.

TASK
Exact Galaxy prediction / recourse task.

REPRESENTATION
Latent dimensionality and W_CF definition.

CALIBRATION
r_star and beta_geo_star.

MAIN RESULT
Future Validity, Cost, and Runtime.

GEOMETRY ABLATION
Results across beta_geo.

QUALITATIVE RESULT
Whether increasing beta visibly preserves plausible galaxy morphology.

SHIFT ROBUSTNESS
Behavior under stronger external Galaxy shifts.

COMPUTE
Servers, GPUs, wall time, GPU-hours.

FAILURES
Missing checkpoints, failed jobs, incompatible assets, solver failures.

LIMITATIONS
Finite attack bank, local optimization, latent representation dependence.

Do not make claims unsupported by the completed experiment.