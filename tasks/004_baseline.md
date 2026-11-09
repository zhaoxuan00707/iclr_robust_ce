Implement a CONTINUOUS SINGLE-BUDGET WRO solver with MINIMAL VALIDITY RESTORATION.

The purpose of this experiment is NOT to redesign the geometry again.

We want to preserve the strong performance of the previous continuous
single-budget / waypoint-style realization, while fixing only the final
current-validity issue.

Core principle:

    optimize robust supports freely in continuous feature space;

    then minimally repair only those final support points that are invalid
    under the current model f0.

Do NOT restrict Q to graph nodes.
Do NOT optimize directly over P_2(F0).
Do NOT add a current-validity penalty to the training objective.

==================================================
1. REUSE THE PREVIOUS STRONG CONTINUOUS SETUP
==================================================

Start from the previous continuous single-budget geodesic-excess WRO codebase
that achieved the strong HELOC robustness improvements.

Freeze:

    f0
    tau
    G
    P_CF
    P1
    preprocessing
    factual / CF identities
    W_data
    U_r
    official Update_full
    clean full-update attack bank
    adversarial-search configuration
    evaluation scenarios

Do NOT use the new direct graph-supported Q optimizer as the main solver.

The support variables must remain continuous:

    Q_raw = sum_j b_j delta_{w_j_raw},

with

    w_j_raw in continuous feature space.

==================================================
2. ROBUST OBJECTIVE
==================================================

Keep the same practical finite-bank robust objective:

    H_a(Q)
      =
      sum_j b_j [tau - f_a(w_j)]_+,

and

    Phi_hat_r(Q)
      =
      max_{a in A_r} H_a(Q).

Use

    max_model mean_CF hinge

as the primary reduction.

Never use

    mean_CF max_model hinge.

==================================================
3. GEOMETRY BUDGET
==================================================

Keep the single-budget formulation:

    D = W_CF(P_CF,P1),

    B_geo = beta_geo * D.

Use diagnostic values:

    beta_geo in {1.00, 1.05, 1.10, 1.20}.

The raw optimized Q must satisfy:

    W_CF(P_CF,Q_raw)
    + W_CF(Q_raw,P1)
    <= B_geo

within solver tolerance.

Use the SAME geometry implementation as the previous strong
single-budget experiment unless explicitly required otherwise.

Do NOT replace the continuous support space with graph support.

==================================================
4. CONTINUOUS ROBUST OPTIMIZATION
==================================================

For each beta_geo optimize:

    Q_raw
      =
      argmin_Q Phi_hat_r(Q)

subject to

    W_CF(P_CF,Q)
    + W_CF(Q,P1)
    <= B_geo.

Use the previous continuous optimization machinery:

    differentiable support coordinates
    primal-dual / augmented-Lagrangian geometry constraint
    feature bounds
    multiple initializations if already used.

Do NOT add:

    current-validity penalty
    0.7 construction margin
    near-target penalty
    graph-support restriction.

The goal of this stage is to reproduce the previous strong robust solution.

==================================================
5. IDENTIFY CURRENT-INVALID SUPPORTS
==================================================

After Q_raw is frozen, compute for each support:

    s_j = f0(w_j_raw).

Define invalid set:

    I = {j : s_j < tau}.

Report:

    invalid support count
    invalid mass
    minimum f0 score
    mean violation depth

where violation depth is

    [tau - f0(w_j_raw)]_+.

Do NOT modify valid supports.

==================================================
6. MINIMAL TARGET-DIRECTED VALIDITY RESTORATION
==================================================

For every invalid support w_j_raw, use its corresponding favorable target
anchor y_j from P1.

Define:

    w_j(alpha)
      =
      (1-alpha) w_j_raw
      + alpha y_j,

    alpha in [0,1].

Find the SMALLEST alpha_j such that

    f0(w_j(alpha_j)) >= tau.

Use bisection or another reliable one-dimensional root search.

Set:

    w_j_repair
      =
      w_j(alpha_j).

For already-valid supports:

    w_j_repair
      =
      w_j_raw.

This creates:

    Q_repair
      =
      sum_j b_j delta_{w_j_repair}.

Important:

    use the minimum alpha satisfying current validity.

Do not move an invalid point farther than necessary.

==================================================
7. TARGET ANCHOR VALIDITY CHECK
==================================================

Before repair, verify that each repair anchor y_j satisfies:

    f0(y_j) >= tau.

If a corresponding y_j is not valid:

    do NOT silently repair toward it.

Instead:

    use the nearest compatible valid target anchor under the existing
    P1 matching / coupling,

or mark the repair as infeasible.

Report every such case.

==================================================
8. FEATURE-SEMANTICS CHECK
==================================================

For continuous features, linear interpolation may be used directly.

For discrete / categorical / immutable features:

    preserve the existing legal feature-handling rules.

Do NOT create illegal fractional categorical states.

If the previous pipeline has a projection / decoding layer for legal tabular
states, reuse it.

After legal-state projection, recheck:

    f0(w_j_repair) >= tau.

==================================================
9. RECOMPUTE GEOMETRY AFTER REPAIR
==================================================

After repair compute exactly:

    d0_repair
      =
      W_CF(P_CF,Q_repair),

    d1_repair
      =
      W_CF(Q_repair,P1),

    L_repair
      =
      d0_repair + d1_repair.

Check:

    L_repair <= B_geo.

This is mandatory.

The raw solution being feasible is NOT sufficient.

==================================================
10. IF REPAIR VIOLATES THE GEOMETRY BUDGET
==================================================

Do NOT silently accept an over-budget repaired solution.

First report the violation.

Then attempt a constrained minimal repair:

For each invalid support jointly solve / search for the smallest total repair
that satisfies both:

    f0(w_j) >= tau

for all supports,

and

    W_CF(P_CF,Q)
    + W_CF(Q,P1)
    <= B_geo.

Preferred simple strategy:

    use the target-directed repair path,
    but jointly reduce / allocate repair movement while maintaining all
    current-validity constraints.

If no feasible repaired Q exists for that beta:

    mark that beta as INFEASIBLE.

Do not weaken tau.
Do not enlarge B_geo automatically.

==================================================
11. OPTIONAL SECOND REPAIR BASELINE
==================================================

As a diagnostic only, also implement a nearest-valid projection baseline:

    w_j_proj
      =
      argmin_z d_feature(z,w_j_raw)

    subject to

      f0(z) >= tau
      and legal feature constraints.

This is optional.

Do NOT make it the default method unless it clearly outperforms the simpler
target-directed repair.

The main method remains target-directed minimal restoration.

==================================================
12. EVALUATE RAW VS REPAIRED
==================================================

For every beta report side by side:

    Q_raw
    Q_repair.

Metrics:

    training-bank hinge
    current validity
    mean future validity
    worst future validity
    mean future hinge
    worst future hinge
    recourse cost
    geometry length
    geometry slack.

Also report deltas:

    Delta bank hinge
    Delta mean future validity
    Delta worst future validity
    Delta recourse cost
    Delta geometry length.

Primary scientific question:

    does a SMALL final repair preserve most of the robustness gain?

==================================================
13. PATH / OPERATING POINT
==================================================

If the previous strong method uses a path

    P_CF -> Q -> P1,

construct it using the same realization as the previous single-budget method.

Use Q_repair, not Q_raw, for final deployment evaluation.

If lambda selection is required:

    freeze Q_repair and the path first;

    evaluate all candidate lambdas with the shared attack bank;

    choose lambda only from current-valid final distributions.

Do NOT require all intermediate path points to be current-valid unless the
method explicitly deploys them.

Only the final deployed distribution must be a valid CF distribution.

==================================================
14. FRESH ADVERSARIAL DISCOVERY
==================================================

After Q_repair and final path are frozen, run fresh strengthened
training-side adversarial searches.

Persist:

    perturbations
    shifted training distributions
    full-update models
    source beta
    source lambda
    raw/repaired source label
    data-side radius feasibility.

Merge compatible models into the expanded shared attack bank.

Do NOT feed these fresh attacks back into Q learning in the primary run.

==================================================
15. FINAL SHARED-BANK EVALUATION
==================================================

Evaluate Q_repair / repaired path under the expanded bank:

    Phi_hat_fresh(Q)
      =
      max_{a in A_r_fresh}
        E_Q [tau - f_a(z)]_+.

If lambda selection is used, reselect lambda using ONLY the expanded
training-side bank.

Do not use final future-test scenarios for selection.

==================================================
16. REQUIRED COMPARISONS
==================================================

Compare:

A. previous strong continuous single-budget raw solution

B. same solution + minimal validity restoration

C. direct graph-supported favorable-region method

D. old heuristic waypoint

E. straight / beta=1 baseline

Use matched attack banks where compatible.

==================================================
17. SUCCESS CRITERIA
==================================================

The experiment is scientifically successful if:

1. repaired output has 100% current validity;

2. repaired output remains within B_geo;

3. robustness degradation from raw to repaired is small;

4. most of the previous continuous-method future-validity gain is preserved;

5. repair movement is small.

Do NOT define success merely as beating the graph-supported method.

==================================================
18. IMPORTANT FAILURE CASE
==================================================

If repair causes a large collapse in robustness, report this directly.

Specifically, if:

    raw robust performance is strong,

but

    minimal validity repair destroys it,

then conclude that the previous gain relied materially on current-invalid
regions.

That is an important scientific result and must not be hidden.

==================================================
19. OUTPUT
==================================================

Write to:

    robust_ce_results/minimal_validity_repair_wro_v1/<run_id>/

Save:

    resolved_config.json
    instance_manifest.json
    raw_optimization_trace.csv
    raw_supports.*
    raw_geometry.csv
    raw_validity.csv

    repair_manifest.csv
    repair_alpha.csv
    repaired_supports.*
    repaired_geometry.csv
    repaired_validity.csv

    raw_vs_repaired_metrics.csv
    fresh_attack_manifest.csv
    cross_eval_matrix.csv
    lambda_curves.csv
    final_metrics.csv
    report.md

Do not overwrite previous experiments.

==================================================
20. REPORT
==================================================

Use this structure:

IMPLEMENTED
Confirm continuous WRO was preserved and only final invalid supports were
minimally repaired.

RAW SOLUTION
Robustness, geometry, current-invalid mass.

REPAIR SIZE
Number/mass repaired, alpha statistics, support displacement.

FEASIBILITY
100% current validity? Geometry budget still satisfied?

ROBUSTNESS RETENTION
Raw vs repaired bank hinge and future validity.

FRESH ATTACK CHECK
Did repaired robustness survive the expanded bank?

COMPARISON
Straight vs graph-supported vs old waypoint vs raw continuous vs repaired
continuous.

CORE CONCLUSION
State whether minimal validity restoration preserves the earlier strong
continuous-WRO gains.