Implement and run a full StyleMNIST replication of the current GRACE main
experiment.

The goal is NOT to invent a new StyleMNIST-specific method. Reuse the current
GRACE formulation and experimental protocol as closely as possible, while
adapting the CF representation and geometry to the existing StyleMNIST
classifier / autoencoder assets already available on the server.

Use ALL available GPU servers / GPUs safely and in parallel.

==================================================
0. FIRST: DISCOVER EXISTING STYLEMNIST ASSETS
==================================================

Before changing code or launching experiments, inspect the repository,
checkpoints, experiment directories, and server-visible model storage.

Find and report:

1. StyleMNIST dataset loaders and preprocessing.
2. Existing trained classifier checkpoints, especially the assets referred to
   as classifiers 4 through 9.
3. What "classifier 4--9" actually means in this repository:
   - checkpoint IDs,
   - random seeds,
   - architectures,
   - training domains,
   - style conditions,
   - or something else.
   Do NOT assume.
4. Existing autoencoder / VAE checkpoints.
5. Reconstruction quality / latent dimensionality / preprocessing of each
   autoencoder.
6. Existing CF-generation code for StyleMNIST.
7. Existing StyleMNIST future-shift / corruption / retraining code, if any.
8. Existing multi-GPU / SLURM / tmux / launcher infrastructure.

Prefer existing verified assets over retraining.

Create:

    stylemnist_asset_inventory.md

containing exact paths, checkpoint names, architectures, and compatibility.

Do not begin the expensive run until this inventory is complete.

==================================================
1. METHOD MUST MATCH CURRENT GRACE
==================================================

Use the current no-current-validity-check GRACE formulation.

For a fixed base classifier f_0 and base CF generator G:

    z_i = G(x_i)

and

    P_CF = empirical distribution of base CFs.

Let P_1 denote the empirical favorable / target-class reference distribution.

The data-side robust risk remains

    Phi_r(Q)
      =
      sup_{P in U_r}
      E_{z~Q} [tau - f_P(z)]_+,

with

    f_P = Update(f_0,P).

Use a finite bank A_r of feasible full-update models:

    Phi_hat_r(Q)
      =
      max_{f in A_r}
      E_{z~Q} [tau - f(z)]_+.

IMPORTANT aggregation:

    max_model mean_CF hinge

NOT:

    mean_CF max_model hinge.

The CF-side geometry remains

    D = W_CF(P_CF,P_1)

and

    G_B
      =
      {
        Q:
        W_CF(P_CF,Q)
        +
        W_CF(Q,P_1)
        <= B_geo
      },

with

    B_geo = beta_geo * D.

Use the same diagnostic beta grid:

    beta_geo in {1.00, 1.05, 1.10, 1.20}.

The practical optimization remains:

    minimize_Q Phi_hat_r(Q)

subject to

    Q in G_B.

Do NOT add current-validity repair/checks in this branch.

==================================================
2. STYLEMNIST REPRESENTATION
==================================================

Use the best existing compatible autoencoder already present on the server.

Do NOT retrain an autoencoder unless no usable checkpoint exists.

Preferred representation:

    h = Enc(x)
    x = Dec(h).

Run GRACE in latent space if reconstruction and classifier compatibility are
adequate.

Define:

    P_CF^h = Enc_# P_CF
    P_1^h  = Enc_# P_1.

Use latent support coordinates h_j as the optimization variables and decode
them only when classifier predictions are needed:

    x_j = Dec(h_j).

The CF-side ground cost should be the existing latent-space metric used by the
StyleMNIST codebase if one already exists.

Otherwise use Euclidean latent distance as the primary controlled experiment:

    d_CF(h,h') = ||h-h'||_2.

Then W_CF is the corresponding empirical 2-Wasserstein distance in latent
space.

Do NOT silently mix pixel-space and latent-space W_CF.

Record clearly which space is used for:

    optimization,
    Wasserstein geometry,
    classifier evaluation,
    recourse cost.

==================================================
3. BASE CLASSIFIER / CLASSIFIER 4--9
==================================================

Inspect the discovered classifiers 4--9 before deciding their role.

If they are multiple independently trained StyleMNIST classifiers compatible
with the same task and preprocessing:

- choose one predefined classifier as f_0 using repository convention;
- use the others only in roles justified by the existing experiment design;
- do NOT choose f_0 based on final future-validity performance.

If classifier 4--9 correspond to target labels/classes rather than model
replicates, preserve the repository's task definition and run each compatible
task independently.

Do not guess what 4--9 mean.

Save the final mapping in:

    classifier_role_manifest.json

==================================================
4. BASE COUNTERFACTUAL GENERATION
==================================================

Reuse an existing verified StyleMNIST CF generator if present.

If the repository already has a latent-space counterfactual optimizer
compatible with the autoencoder, prefer it.

The base CF generator must be frozen before GRACE refinement.

Generate the same fixed factual set for all comparisons.

Save:

    factual_ids.npy
    base_cf.npy
    base_cf_latents.npy

Do not regenerate different base CFs for different beta_geo values.

==================================================
5. TARGET REFERENCE DISTRIBUTION
==================================================

Construct P_1 using favorable / target-class training examples under the same
task definition as the base classifier.

Encode them using the same frozen encoder:

    P_1^h = Enc_# P_1.

Use a fixed, reproducible subset if the full target population is too large.

The subset must be chosen before looking at future-shift outcomes.

==================================================
6. DATA-SIDE UNCERTAINTY / ATTACK BANK
==================================================

Reuse the current GRACE adversarial-search logic as closely as possible.

The uncertainty acts on the TRAINING DATA, not directly on network weights.

Each feasible adversarial training distribution P produces:

    f_P = Update(f_0,P)

using a fixed update protocol.

Construct a shared attack bank A_r by repeated adversarial search and multiple
restarts.

Every accepted model must be persisted.

Save for every candidate:

    perturbation parameters
    shifted training data identifier
    W_data distance
    feasibility status
    full-update checkpoint
    source restart
    source beta if applicable

All candidate Q distributions must be cross-evaluated against the shared bank.

==================================================
7. FUTURE-SHIFT EVALUATION
==================================================

First inspect whether the repository already contains a StyleMNIST future-shift
evaluation protocol.

IF AN EXISTING VERIFIED STYLEMNIST SHIFT PROTOCOL EXISTS:
    reuse it exactly.

IF NO SUCH PROTOCOL EXISTS:
    do NOT silently invent one and present it as established.
    Implement a clearly separated proposed StyleMNIST stress suite only after
    documenting that it is newly introduced for this experiment.

If a new suite is required, prefer changes to image STYLE rather than direct
weight perturbations, for example only if supported naturally by StyleMNIST:

    stroke thickness / width
    slant / affine style
    contrast / intensity
    background / texture
    style-code perturbation

but keep the suite minimal and predeclared.

Future models must be obtained by changing training data and applying a fixed
model update rule, mirroring the main-paper evaluation philosophy.

Do NOT tune GRACE on final future-shift outcomes.

==================================================
8. PRIMARY STYLEMNIST EXPERIMENT
==================================================

Run the analogue of the current main experiment.

For each relevant StyleMNIST task / classifier setup:

A. Base CFs
B. beta_geo = 1.00
C. beta_geo = 1.05
D. beta_geo = 1.10
E. beta_geo = 1.20

Report:

    Future Validity
    Recourse Cost
    finite-bank robust risk Phi_hat_r
    runtime
    geometry feasibility

If existing robust-CF baselines have compatible StyleMNIST implementations,
also run them.

Do NOT spend substantial engineering effort porting incompatible tabular-only
baselines merely to populate a table. Record incompatibility explicitly.

==================================================
9. RECOURSE COST
==================================================

Report at least one user-interpretable image-domain cost.

Primary candidate:

    pixel-space L2 between factual image and returned CF.

Also report latent L2 separately if useful:

    ||Enc(x) - Enc(x_cf)||_2.

Do not call latent Wasserstein geometry "recourse cost".

Keep:

    geometry budget

and

    factual-to-CF cost

as separate quantities, consistent with the main paper.

==================================================
10. GEOMETRY-BUDGET ABLATION
==================================================

Produce the StyleMNIST analogue of the HELOC geometry-budget table:

    beta_geo
    Future Validity
    Cost
    Phi_hat_r
    Runtime

for:

    {1.00, 1.05, 1.10, 1.20}.

Check whether:

    beta_geo increases
        -> achieved Phi_hat_r decreases

up to numerical optimization error.

Do NOT claim empirical monotonicity is guaranteed by the solver.

==================================================
11. SHIFT-SEVERITY CURVE
==================================================

If the StyleMNIST future-shift protocol has ordered severity levels, produce:

    x-axis: external shift severity
    y-axis: Future Validity

for GRACE and any compatible baselines.

Severity must be defined by the EXTERNAL stress protocol and must not be
identified with GRACE's internal Wasserstein radius r.

==================================================
12. VISUAL QUALITATIVE PANEL
==================================================

For a fixed set of factual examples, save a compact qualitative grid:

    factual
    base CF
    beta=1.00
    beta=1.05
    beta=1.10
    beta=1.20
    target/reference example if useful

Use the SAME factual examples across columns.

Do not cherry-pick based on final future-model performance.

Prefer examples fixed by index / seed before evaluation.

==================================================
13. MULTI-GPU EXECUTION
==================================================

Use all available GPUs, but avoid multiple workers writing to the same output
directory.

First detect:

    nvidia-smi
    CUDA_VISIBLE_DEVICES
    hostname
    scheduler environment
    available GPU memory.

Create one immutable experiment manifest before launching jobs.

Parallelize over independent units such as:

    classifier/task
    beta_geo
    adversarial restart
    future-shift realization
    random seed

as appropriate.

Do NOT parallelize two processes that mutate the same attack bank or checkpoint.

Use per-job directories and merge only after completion.

If SLURM exists, generate and use a job array.

If no scheduler exists, create a launcher assigning one worker per GPU.

Each process must log:

    hostname
    GPU ID
    CUDA device name
    seed
    task
    start/end time
    git commit
    resolved config

Implement resume-safe execution:
completed jobs must not be rerun unless explicitly requested.

==================================================
14. REPRODUCIBILITY
==================================================

Use fixed seeds.

Freeze and record:

    dataset split
    factual IDs
    target reference IDs
    f_0 checkpoint
    encoder checkpoint
    decoder checkpoint
    base CFs
    attack-bank models
    beta grid
    update hyperparameters
    future-shift scenarios

No final-test outcome may affect method configuration.

==================================================
15. OUTPUT STRUCTURE
==================================================

Write everything under:

    robust_ce_results/stylemnist_grace_v1/<run_id>/

Required files:

    stylemnist_asset_inventory.md
    resolved_config.json
    classifier_role_manifest.json
    hardware_manifest.json
    factual_ids.npy
    target_ids.npy
    base_cf.*
    attack_bank_manifest.csv
    geometry_ablation.csv
    future_validity.csv
    runtime.csv
    geometry_feasibility.csv
    shift_severity.csv
    qualitative_examples/
    figures/
    checkpoints/
    logs/
    report.md

Do not overwrite previous StyleMNIST experiments.

==================================================
16. REQUIRED FIGURES / TABLES
==================================================

Generate:

TABLE A — Main StyleMNIST result
    Method / configuration
    Future Validity
    Cost
    Runtime

TABLE B — Geometry budget ablation
    beta_geo
    Future Validity
    Cost
    Phi_hat_r
    Runtime

FIGURE A — Future Validity vs external shift severity

FIGURE B — qualitative factual / CF refinement grid

Use publication-quality PDF figures and CSV backing data.

==================================================
17. IMPORTANT SCIENTIFIC RULES
==================================================

Do NOT:

- add a current-validity repair/check branch;
- tune on final future shifts;
- directly perturb model weights and call that the primary future-shift model;
- conflate external evaluation severity with internal r;
- conflate latent Wasserstein geometry with user recourse cost;
- choose the best classifier / autoencoder based on final test FV;
- fabricate missing checkpoint metadata;
- retrain existing good classifiers or autoencoders unnecessarily.

If an existing asset is incompatible, document why.

==================================================
18. FINAL REPORT
==================================================

At completion, summarize:

ASSETS
Which classifier(s) 4--9 and which autoencoder were actually used.

TASK
Exact StyleMNIST prediction / recourse task.

GEOMETRY
Pixel or latent W_CF and why.

MAIN RESULT
Future Validity, Cost, Runtime.

BETA EFFECT
Results for beta_geo = 1.00, 1.05, 1.10, 1.20.

SHIFT ROBUSTNESS
Whether GRACE degrades gradually under stronger external style shifts.

COMPUTE
How many servers / GPUs were used and total GPU-hours.

FAILURES
Any failed jobs, infeasible geometry, incompatible checkpoints, or missing
assets.

LIMITATIONS
Finite bank, local solver, StyleMNIST-specific representation choices.

Do not make claims not directly supported by the completed run.