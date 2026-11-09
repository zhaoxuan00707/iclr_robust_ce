Implement a FAVORABLE-REGION GEODESIC-EXCESS WRO version of LIFT-CF.

This experiment should test the clean formulation:

    minimize_Q
        Phi_hat_r(Q)

    subject to

        Q in P_2(F0)

and

        W_CF(P_CF,Q)
        + W_CF(Q,P1)
        <= B_geo,

where

    F0 = {x : f0(x) >= tau}

and

    W_CF = W_{2,d_F0}

uses the intrinsic ground metric inside the current favorable region.

The key design principle is:

    current validity is built into the decision space itself.

Do NOT add a separate current-validity penalty or augmented-Lagrangian
constraint for f0(x) >= tau.

==================================================
1. FREEZE THE FIXED PROBLEM INSTANCE
==================================================

Use the existing fixed HELOC problem instance.

Freeze:

    f0
    tau
    G
    P_CF
    P1
    preprocessing
    feature representation
    factual/CF identities
    train/calibration/test splits
    official Update_full
    W_data
    U_r
    b_ref
    strengthened adversarial solver
    clean attack bank
    evaluation scenarios

Do not retrain f0.
Do not regenerate P_CF.
Do not change the data-side WRO problem.

==================================================
2. DEFINE THE FAVORABLE REGION
==================================================

Define

    F0 = {x : f0(x) >= tau}.

All admissible refined CF distributions Q must have support in F0.

This is not a soft penalty.

The optimization domain is:

    Q in P_2(F0).

Therefore every retained support point must satisfy

    f0(z) >= tau.

Do not introduce a separate construction threshold such as 0.7 in the
primary experiment.

==================================================
3. DEFINE THE INTRINSIC GROUND METRIC
==================================================

Define the favorable-region intrinsic distance

    d_F0(x,y)
      =
      inf over curves gamma
      integral_0^1 ||gamma'(t)||_2 dt

subject to

      gamma(0)=x
      gamma(1)=y
      gamma(t) in F0 for all t.

Then define

    W_CF
      =
      W_{2,d_F0}.

Use this SAME metric consistently for:

    D = W_CF(P_CF,P1)

    W_CF(P_CF,Q)

    W_CF(Q,P1).

Do not mix Euclidean W2 and intrinsic W2 in the same constraint.

==================================================
4. HANDLE CONNECTIVITY EXPLICITLY
==================================================

Before optimization, verify that the relevant source and target masses lie in
compatible geodesically connected components of F0.

If the graph/discretization implementation is used, distinguish:

A. true favorable-region disconnection;

B. graph sparsity artifact.

Use the previously validated local connectivity repair logic only for
graph-discretization artifacts.

Do NOT:

- connect through unfavorable regions;
- drop disconnected mass;
- silently relax F0;
- invent finite distances across disconnected components.

Report component mass compatibility.

==================================================
5. GEOMETRY BUDGET
==================================================

Compute

    D = W_CF(P_CF,P1).

Define

    B_geo = beta_geo * D,

with diagnostic grid

    beta_geo in {1.00, 1.05, 1.10, 1.20}.

Interpretation:

    beta_geo = 1
        -> shortest favorable-region Wasserstein geometry

    beta_geo > 1
        -> bounded detour inside F0.

No delta, epsilon or kappa should be introduced in this branch.

==================================================
6. DECISION DISTRIBUTION
==================================================

Use a finite-particle approximation:

    Q = sum_j b_j delta_{w_j},

with the same masses b_j as P1.

The support variables are

    {w_j}.

Every optimization step must keep candidate support points inside the
representation of F0.

Preferred implementation:

- parameterize admissible candidates using favorable graph nodes / feasible
  continuous coordinates;
- or reject candidate updates leaving F0 before evaluation.

Do NOT use a soft current-validity penalty.

==================================================
7. ROBUST OBJECTIVE
==================================================

For fixed data-side radius r, let A_r be the clean training-derived full-update
attack bank.

For model a define

    H_a(Q)
      =
      sum_j b_j [tau - f_a(w_j)]_+.

Then

    Phi_hat_r(Q)
      =
      max_{a in A_r} H_a(Q).

Primary reduction must be:

    max_model population_mean_hinge.

Never use:

    population_mean max_model hinge.

For optimization only, a smooth hinge / smooth outer max may be used.

Final evaluation must use true hinge and true max over the full-update bank.

==================================================
8. PRIMARY CONSTRAINED PROBLEM
==================================================

For each beta_geo solve:

    minimize_Q
        Phi_hat_r(Q)

    subject to

        Q in P_2(F0)

and

        W_CF(P_CF,Q)
        + W_CF(Q,P1)
        <= B_geo.

Do NOT add:

    near-target L2 penalties
    current-model penalties
    waypoint penalties
    spline penalties
    extra anchor terms.

The favorable-region geometry and the total Wasserstein budget should define
the decision region.

==================================================
9. NUMERICAL SOLVER
==================================================

Use the simplest solver consistent with the chosen representation.

If support points are optimized continuously:

- use the existing differentiable finite-bank objective;
- use intrinsic/graph OT gradients where available;
- enforce the geometry budget with constrained optimization or primal-dual
  updates;
- keep all support points inside F0 by construction / feasible parameterization.

If support is restricted to favorable graph nodes:

- formulate the problem over admissible support locations / masses;
- exploit exact or discrete OT where practical.

Do not claim global optimality unless the finite subproblem is actually solved
globally.

==================================================
10. SPECIAL CASE beta_geo = 1
==================================================

Verify numerically that

    beta_geo = 1

restricts feasible Q to the shortest favorable-region geodesic set up to
representation / solver tolerance.

Because

    W_CF(P_CF,Q)
    + W_CF(Q,P1)
    >= D,

the beta=1 constraint should force equality.

Report any failure to reproduce the original favorable-region geodesic as a
representation issue.

==================================================
11. NESTED-FEASIBLE-SET CHECK
==================================================

For

    beta_1 <= beta_2

the theoretical feasible sets are nested.

Verify empirically that the best achieved TRAINING-bank objective does not
materially worsen as beta increases, apart from optimization error.

If it does, report local/numerical solver failure rather than a theoretical
contradiction.

==================================================
12. CONSTRUCT A PATH ONLY AFTER Q IS LEARNED
==================================================

The robust decision is Q itself.

A path is only a realization / visualization device.

After learning a feasible Q, construct:

    P_CF
      -> Q
      -> P1

using favorable-region intrinsic geodesic interpolation on each segment.

Because both segment geodesics are computed in d_F0, every continuous segment
should remain inside F0 by definition, subject to graph/discretization
approximation.

Do not use Euclidean interpolation between controls unless F0 convexity is
verified.

==================================================
13. PATH PARAMETERIZATION
==================================================

Let

    d0 = W_CF(P_CF,Q)

    d1 = W_CF(Q,P1)

    L = d0 + d1.

Place Q at

    lambda_Q = d0 / L.

Parameterize the piecewise intrinsic path by normalized Wasserstein arclength.

Do not force lambda_Q = 0.5.

==================================================
14. FRESH ADVERSARIAL DISCOVERY
==================================================

After Q and the path are frozen, run fresh strengthened adversarial searches
against representative path points.

Use only training-side information.

Persist all new:

    perturbations
    shifted training distributions
    full-update models
    source beta_geo
    source lambda
    radius-feasibility values.

Merge compatible attacks into an expanded shared bank.

Do not update Q using these fresh attacks in the primary experiment.

==================================================
15. FINAL OPERATING POINT
==================================================

If the final method still selects an operating point along the frozen path,
evaluate:

    Phi_hat_bank(beta,lambda)

with the expanded bank.

Select:

    lambda_hat
      =
      argmin_lambda
        Phi_hat_bank(beta,lambda).

Because the path lies in F0 under the intrinsic metric, current validity should
hold automatically up to numerical approximation.

Do not use recourse cost or final-test validity to choose lambda.

==================================================
16. REQUIRED CONTROLS
==================================================

Compare:

A. Original Euclidean straight path.

B. Existing favorable-region intrinsic shortest path / beta=1.

C. Favorable-region geodesic-excess WRO with beta > 1.

D. Old heuristic waypoint.

E. Previous Euclidean single-budget geodesic-excess solver.

Use matched attack banks and evaluation scenarios where compatible.

==================================================
17. METRICS
==================================================

Report:

    beta_geo
    selected lambda
    training-bank hinge
    fresh-bank hinge
    mean future validity
    worst future validity
    mean future hinge
    worst future hinge
    recourse cost
    current validity

Geometry diagnostics:

    D
    B_geo
    W_CF(P_CF,Q)
    W_CF(Q,P1)
    total intrinsic detour
    geometry slack
    lambda_Q
    component IDs
    connectivity diagnostics.

==================================================
18. VALIDITY CHECK
==================================================

Explicitly verify:

    support(Q) subset F0

and all evaluated path support points satisfy:

    f0(z) >= tau

up to declared numerical tolerance.

If graph interpolation produces a current-invalid point, treat this as a
discretization/realization bug because the declared intrinsic path should lie
inside F0.

Do not add a soft penalty to hide the issue.

==================================================
19. THEORY-FACING CHECKS
==================================================

Verify numerically:

1. beta=1 recovers the shortest favorable-region geodesic restriction.

2. beta larger produces nested feasible regions.

3. learned Q satisfies exact total intrinsic geometry budget.

4. current validity follows from membership in F0.

5. fresh-bank evaluation remains finite-bank only; do not claim the true
   supremum is solved.

==================================================
20. SCIENTIFIC QUESTIONS
==================================================

Answer:

Q1.
Does defining the CF decision space directly on P_2(F0) remove the
current-validity failure without extra penalties?

Q2.
Does beta_geo > 1 still reduce robust risk?

Q3.
Does the improvement survive fresh adversarial discovery?

Q4.
How does recourse cost change relative to beta=1 and the old waypoint?

Q5.
Does the favorable-region intrinsic geometry introduce practical connectivity
problems?

Q6.
Is the gain due to added detour freedom rather than merely switching from
Euclidean to intrinsic geometry?

==================================================
21. INTERPRETATION RULES
==================================================

Do NOT claim:

    current validity is guaranteed for arbitrary Euclidean interpolation.

The validity argument applies to paths defined in the intrinsic favorable-region
geometry.

Do NOT claim:

    beta larger guarantees future validity.

The structural guarantee is only:

    larger beta gives a nested larger favorable-region decision set.

Do NOT claim exact global WRO optimality.

==================================================
22. OUTPUT
==================================================

Write to a new directory:

    robust_ce_results/favorable_region_geodesic_wro_v1/<run_id>/

Save:

    resolved_config.json
    instance_manifest.json
    component_diagnostics.csv
    graph_connectivity.csv
    attack_bank_manifest.csv
    beta_geo_grid.csv
    optimization_trace.csv
    learned_distributions.*
    geometry_feasibility.csv
    path_points.*
    current_validity_checks.csv
    fresh_attack_manifest.csv
    cross_eval_matrix.csv
    lambda_curves.csv
    final_metrics.csv
    report.md

Do not overwrite previous experiments.

==================================================
23. FINAL RESPONSE
==================================================

Keep concise.

Use:

IMPLEMENTED
Confirm optimization directly over the favorable-region geometry.

CONNECTIVITY
Report whether F0 supports the required transport.

SPECIAL CASE
Did beta=1 recover the intrinsic shortest-geodesic solution?

BETA EFFECT
Robust risk, future validity and recourse cost across beta.

CURRENT VALIDITY
Did all learned and deployed outputs remain in F0?

FRESH-ATTACK CHECK
Did gains survive the enlarged bank?

OLD METHOD COMPARISON
Straight / previous geodesic-excess / old waypoint / new intrinsic formulation.

LIMITATIONS
Finite bank/support, graph approximation, no unrestricted-global guarantee.

CORE CONCLUSION
Under 150 words.