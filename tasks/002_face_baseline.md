Implement an IDENTITY-PRESERVING FFHQ realization of the current GRACE method.

The goal is to solve the following issue:

    population-level distributional robustness does not by itself guarantee
    preservation of each factual individual's identity / non-target content.

For the FFHQ task

    non-smiling -> smiling,

we want GRACE to modify only a smile-relevant latent bottleneck while preserving
the factual-specific identity/content representation.

IMPORTANT:
- Do NOT redesign the GRACE WRO objective.
- Do NOT add a soft identity penalty as the primary solution.
- Do NOT add the current-validity repair/check branch.
- Prefer a HARD frozen content bottleneck.
- Reuse existing LeapFactual face assets wherever possible.

Existing project:

    /mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/

Use all available GPUs safely where useful.

==================================================
0. AUDIT LEAPFACTUAL FIRST
==================================================

Inspect the LeapFactual project for FFHQ / face assets related to:

    encoder
    decoder / generator
    identity/content bottleneck
    smile / attribute bottleneck
    disentangled representation
    latent editing
    factual residual
    immutable code
    mutable code
    ArcFace / face identity embedding
    reconstruction
    counterfactual generation

Do NOT assume the latent factorization.

Find the exact existing architecture and answer:

1. Is the face latent explicitly decomposed into task/attribute and residual
   content components?
2. Is there already a smile-specific bottleneck?
3. Which latent variables are modified by the old face-to-smile pipeline?
4. Which latent variables are preserved from the factual?
5. Is there an existing identity-preservation metric?
6. Is ArcFace or another face embedding model already available?
7. Are there trained checkpoints for all required components?

Write:

    identity_bottleneck_inventory.md

with exact code paths, checkpoint paths, tensor shapes, and semantics.

Prefer the existing LeapFactual decomposition if it is compatible.

==================================================
1. KEEP THE CORE GRACE OBJECTIVE UNCHANGED
==================================================

The data-side uncertainty remains:

    U_r = {P : W_data(P, P_tr_hat) <= r}.

Each shifted training distribution induces:

    f_P = Update(f_0,P).

The ideal robust objective remains:

    Phi_r(Q)
      =
      sup_{P in U_r}
      E_{z~Q} [tau - f_P(z)]_+.

The finite-bank approximation remains:

    Phi_hat_r(Q)
      =
      max_{f in A_r}
      E_{z~Q} [tau - f(z)]_+.

Use:

    max_model mean_CF hinge

NOT:

    mean_CF max_model hinge.

The geometry budget remains:

    B_geo = beta_geo * D

with:

    beta_geo in {1.00,1.05,1.10,1.20}.

Do NOT modify the data-side WRO.

==================================================
2. DEFINE A CONDITIONAL LATENT REALIZATION
==================================================

Preferred representation:

    E(x_i) = (u_i, v_i)

where:

    u_i = smile / target-relevant bottleneck

and

    v_i = factual-specific content / identity / nuisance representation.

For every factual x_i, freeze:

    v_i

throughout GRACE.

Only optimize:

    u_i.

The final refined image must be realized as:

    x_i' = D(u_i', v_i).

Thus:

    v_i' = v_i

is a hard architectural restriction, not a penalty.

Do NOT optimize the full face latent in the primary identity-preserving run.

==================================================
3. WHICH v_i TO PRESERVE
==================================================

Use the content / residual code extracted from the ORIGINAL FACTUAL x_i,
not from an arbitrary target face.

Preferred:

    (u_i^fact, v_i^fact) = E(x_i)

and:

    x_i' = D(u_i', v_i^fact).

If the existing LeapFactual base CF generator already uses exactly this
conditional realization, preserve that implementation.

Do not overwrite individual-specific v_i with a population-level target code.

==================================================
4. BASE CF POPULATION IN THE BOTTLENECK
==================================================

Use the existing base face-to-smile CF generator.

For each fixed factual x_i obtain the base CF:

    z_i = G(x_i).

Encode its task bottleneck:

    u_i^CF.

Define:

    P_CF^u
      =
      (1/n) sum_i delta_{u_i^CF}.

The fixed factual-specific residual remains:

    v_i = v_i^fact.

Do not define the optimization population over the full latent representation.

==================================================
5. TARGET ANCHOR IN TASK SPACE
==================================================

Construct the target anchor from smiling training faces.

For each positive target example y_j:

    E(y_j) = (u_j^+, v_j^+).

Use ONLY the task-relevant component:

    P_1^u
      =
      empirical distribution of {u_j^+}.

Do not transport target identities / residual codes into the factual examples.

This is important:

    P_1^u provides the target-class geometry,

while

    v_i^fact preserves individual content.

==================================================
6. GRACE GEOMETRY IN THE TASK BOTTLENECK
==================================================

Define the CF-side distance only on u:

    d_CF(u,u').

Prefer the existing LeapFactual task-latent metric if one exists.

Otherwise use:

    d_CF(u,u') = ||u-u'||_2.

Then:

    D
      =
      W_CF(P_CF^u, P_1^u).

The GRACE feasible region is:

    G_B^u
      =
      {
        Q^u :
        W_CF(P_CF^u,Q^u)
        +
        W_CF(Q^u,P_1^u)
        <= B_geo
      }.

Use:

    B_geo = beta_geo * D.

This is the only geometry budget.

Do NOT introduce an identity penalty coefficient.

==================================================
7. PRACTICAL ROBUST OBJECTIVE
==================================================

Represent:

    Q^u
      =
      (1/n) sum_i delta_{u_i'}.

For classifier evaluation decode each support with its factual residual:

    x_i' = D(u_i', v_i^fact).

Then evaluate:

    Phi_hat_r(Q^u)
      =
      max_{f in A_r}
      (1/n) sum_i [tau - f(D(u_i',v_i^fact))]_+.

Optimize:

    minimize_{Q^u in G_B^u}
        Phi_hat_r(Q^u).

This is the primary identity-preserving GRACE realization.

==================================================
8. PRIMARY FALLBACK IF LEAPFACTUAL HAS NO EXPLICIT BOTTLENECK
==================================================

Only if no validated task/content decomposition exists, construct a fixed
task-relevant subspace from the pretrained smile classifier.

Do NOT immediately train a new disentanglement model.

For latent h:

    score(h) = f_0(D(h)).

Compute classifier gradients on TRAINING / CALIBRATION latents only:

    g_i = grad_h score(h_i).

Stack the gradients and obtain an SVD / PCA basis.

Define a task-relevant subspace U_task from the leading singular directions.

Choose its dimension using a PREDECLARED gradient-energy criterion, e.g. the
smallest k explaining at least 95% of gradient energy.

Do not select k using final Future Validity.

Define:

    P_task = U_task U_task^T

and:

    P_content = I - P_task.

For factual latent h_i^fact preserve:

    v_i = P_content h_i^fact.

Optimize only:

    u_i' in span(U_task).

Realize:

    h_i' = v_i + u_i'.

Then:

    x_i' = D(h_i').

This fallback is an approximate task/content decomposition and must be reported
as such.

==================================================
9. VERIFY THAT THE BOTTLENECK ACTUALLY MEANS WHAT WE CLAIM
==================================================

Do not call v "classification-irrelevant" without diagnostics.

On training/calibration data evaluate:

A. Target information in u:
    train a lightweight frozen-data probe predicting smile label from u.

B. Target information in v:
    train the same-capacity probe predicting smile label from v.

Desired qualitative behavior:

    Acc(y | u) >> Acc(y | v).

Do not use test outcomes to construct the decomposition.

Also evaluate reconstruction using:

    D(u_i^fact, v_i^fact).

Save:

    bottleneck_diagnostics.csv

containing:

    smile_probe_u
    smile_probe_v
    reconstruction_error
    latent_dimensions.

==================================================
10. IDENTITY PRESERVATION METRIC
==================================================

Reuse an existing LeapFactual identity model if available.

Prefer:

    ArcFace

or the project's validated face-identity embedding.

For each factual / CF pair compute:

    IdentitySim_i
      =
      cosine(
        FaceEmb(x_i),
        FaceEmb(x_i')
      ).

Report:

    mean identity similarity
    std identity similarity
    low-percentile identity similarity

for:

    Base CF
    full-latent GRACE
    bottleneck GRACE.

Do NOT claim architectural identity preservation solely from latent freezing;
also report the measured face-embedding similarity.

==================================================
11. CRITICAL ABLATION
==================================================

Compare three realizations using the SAME:

    factual IDs
    base CFs
    f_0
    attack bank
    r
    beta grid
    external future shifts.

A. Base CF

B. Full-latent GRACE
   - existing current FFHQ GRACE
   - entire latent can move

C. Bottleneck-preserving GRACE
   - optimize only u
   - freeze factual-specific v_i

Report:

    Future Validity
    Phi_hat_r
    Recourse Cost
    Identity Similarity
    LPIPS / perceptual distance if already available
    Runtime

This comparison is essential.

==================================================
12. GEOMETRY BUDGET ABLATION
==================================================

For bottleneck-preserving GRACE evaluate:

    beta_geo in {1.00,1.05,1.10,1.20}.

Report:

    beta_geo
    Future Validity
    Phi_hat_r
    Cost
    Identity Similarity
    Runtime.

Use a fixed r for this ablation according to the existing calibration protocol.

==================================================
13. CALIBRATION
==================================================

Calibrate:

    r_star
    beta_geo_star

using calibration data only.

Do NOT use final FFHQ future shifts.

The bottleneck decomposition itself must also be frozen before final test
evaluation.

Save:

    selected_configuration.json

with:

    r_star
    beta_geo_star
    bottleneck type
    bottleneck dimension
    decomposition checkpoint / basis path.

==================================================
14. QUALITATIVE FIGURE
==================================================

Produce a figure specifically designed to expose identity drift.

Columns:

    Factual
    Base
    Full GRACE
    Bottleneck GRACE
    beta=1.00
    beta=1.05
    beta=1.10
    beta=1.20

It is acceptable to split this into two compact figures if necessary.

Use fixed factual IDs chosen before final evaluation.

Choose a diverse fixed set based on metadata / deterministic IDs, NOT based on
which examples make GRACE look good.

The qualitative panel must make it possible to inspect:

    identity
    smile
    hairstyle
    pose
    age appearance
    background
    other facial attributes.

==================================================
15. DO NOT CHANGE THE CORE PAPER CLAIM
==================================================

GRACE itself remains:

    population-level robust refinement.

The bottleneck is a DOMAIN-SPECIFIC REALIZATION for settings where
individual-content preservation is semantically important.

Do NOT rewrite the central GRACE theory around identity preservation.

Suggested interpretation:

    GRACE controls population-level robustness in the task-relevant
    bottleneck, while a frozen factual-specific content code preserves
    individual semantics.

==================================================
16. STYLEMNIST / GALAXY
==================================================

Do NOT retroactively require this bottleneck for StyleMNIST or Galaxy.

The purpose of this experiment is to test the strong-identity case FFHQ.

Optionally run the bottleneck realization on StyleMNIST / Galaxy only as a
small appendix diagnostic if the same decomposition is already available.

Do not block the FFHQ run on those domains.

==================================================
17. OUTPUT
==================================================

Write to:

    robust_ce_results/ffhq_grace_identity_v1/<run_id>/

Save:

    identity_bottleneck_inventory.md
    resolved_config.json
    selected_configuration.json
    bottleneck_manifest.json
    bottleneck_diagnostics.csv
    identity_metrics.csv
    full_vs_bottleneck.csv
    geometry_ablation.csv
    future_validity.csv
    runtime.csv
    factual_ids.npy
    qualitative_examples/
    figures/
    logs/
    checkpoints/
    report.md

Do not overwrite previous FFHQ / LeapFactual / GRACE runs.

==================================================
18. REQUIRED MAIN RESULTS
==================================================

TABLE 1:

    Method
    Future Validity
    Cost
    Identity Similarity
    Runtime

Rows:

    Base CF
    Full-latent GRACE
    Bottleneck-preserving GRACE

TABLE 2:

    beta_geo
    Future Validity
    Phi_hat_r
    Cost
    Identity Similarity

for bottleneck-preserving GRACE.

FIGURE:

    factual -> base -> full GRACE -> bottleneck GRACE

plus selected beta refinements.

==================================================
19. SUCCESS CRITERION
==================================================

Do NOT define success solely by Future Validity.

The bottleneck realization is useful if it preserves most of the robustness
gain of full-latent GRACE while materially reducing identity drift.

Report the tradeoff explicitly.

For example, the scientifically useful pattern would be:

    full GRACE:
        highest FV but larger identity drift

    bottleneck GRACE:
        similar FV with substantially better identity preservation

but do NOT force this conclusion if the data disagree.

==================================================
20. FINAL REPORT
==================================================

Report:

LEAPFACTUAL REUSE
Which existing bottleneck / encoder / identity components were reused.

DECOMPOSITION
Exact definition of u and v.

DIAGNOSTICS
Smile information in u vs v.

ROBUSTNESS
Base vs full-latent GRACE vs bottleneck GRACE.

IDENTITY
Measured identity preservation.

BETA EFFECT
Robustness / identity behavior as beta_geo increases.

QUALITATIVE RESULT
Whether the same person's identity is visibly preserved.

LIMITATIONS
Whether v is truly classification-independent or only approximately so.

Do not make claims unsupported by the diagnostics.