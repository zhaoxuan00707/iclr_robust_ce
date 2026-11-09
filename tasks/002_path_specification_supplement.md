# Task 002: explicit replacement specification for missing sections 1–9

Status: newly declared experimental design, 2026-09-16. This is not recovered
original text and not an implemented or evaluated method. The user authorized
resolving the preparation blockers. The original truncated `002_baseline.md`
remains unchanged. This supplement supplies missing definitions; its sections
10–20 remain requirements. No production or manuscript changes are implied.

An optional question requested a preference for aggregation across lambda. The
predeclared default below is **mean of population bank risks over the shared
21-point grid**. Any subsequent user correction must precede optimization and
create an explicit config revision. Do not treat absent feedback as approval of
scientific results.

## 1. Fixed instance and first-run scope

Use the fixed HELOC seed-1 instance at:
`/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/heloc`.
Freeze f0, generator-derived CFs, factual IDs, target samples/masses,
preprocessing, train/cal/test membership, official full Update, data uncertainty,
b_ref and data radius. No regeneration. Fixed `r/b_ref=3`, absolute data budget
`8.188831830024718`, tau `.5`. All scientific source changes belong in the new
repository's `experiments/src/lift_ce/`.

Promote the completed run's 5,387 compatible training-derived models from
`evaluation_bank.json` to this new experiment's construction bank. Reverify
instance, full-Update seed/protocol and actual perturbation budgets before use.
Historical post-construction attacks may be construction information in a new
version, but never feed this version's fresh attacks back into training.
Historical future-model metrics are not construction information.

## 2. Metric and one absolute geometry budget

Use unregularized discrete Euclidean W2 consistently. Preserve exact-plan
network simplex and equal-uniform assignment implementations. No graph,
Sinkhorn, new anchor, or robustness/geometry weighted sum.

`D_train = W2(P_CF_train, P1) = 4.868478257271079`.
`beta_geo = [1, 1.05, 1.10, 1.20]`; `B_geo = beta_geo * D_train`.
Keep B_geo absolute and unchanged on held-out batches. Do not choose beta using
held-out outcomes or enlarge it to avoid rejection.

## 3. Path representation

Primary K=2 **intermediate control distributions**, each with the 3,336 target
masses: C_k = sum_j b_j delta_w[k,j]. Endpoints C_0=P_CF_batch and C_3=P1.
Optional K=3/4 is deferred until primary feasibility is established; it is not a
silently substituted fallback. K counts intermediate distributions, not all
knots or individual particles.

Compute optimal OT between every adjacent pair. Preserve split mass exactly.
Interpolate each leg using its own plan and Euclidean displacement population;
never collapse split destinations into a barycentric output. For factual cost,
propagate factual identity with the Markov gluing of adjacent plans through
shared control marginals. Report/support-check gluing mass conservation; scoring
can use the equivalent marginal leg population without expanding full chains.

This finite equal-target-mass control family need not realize every interior
shortest geodesic. Keep a separate exact straight OT control as a representation
reference and report any mismatch, especially for beta=1.

## 4. Total length and lambda parameterization

Let d_k=W2(C_k,C_{k+1}); L=sum_k d_k. Enforce L<=B_geo.
Knots occur at cumulative d_k/L, not at fixed lambda positions. Within each
nonzero leg use constant-speed exact-plan displacement interpolation. Zero
length legs are skipped without changing endpoint masses; if L=0 the path is
constant. Preserve endpoints and all knots exactly.

Use the common lambda grid 0,.05,...,1. Freeze controls, couplings, and arclength
knots before attack discovery. Grid-point current validity is a discrete check,
not a continuous favorable-region or intrinsic-geodesic certificate.

## 5. True path objective

At each lambda, H_lambda=max_a sum_i mass_i [tau-f_a(z_i(lambda))]_+.
The primary training objective is R_path=(1/21) sum_lambda H_lambda.
Model identity may differ between lambda columns; within a column one model
acts on the whole population. This is a new path-design criterion, not the
original single-Q WRO problem or a claim that one future model maximizes all
columns simultaneously. Including endpoints contributes fixed terms.

Do not substitute max_lambda H_lambda, min_lambda H_lambda, or pointwise
per-CF model maxima. Final deployment selection remains the original argmin of
true fresh-bank risk over lambda, separately for each fixed beta.

## 6. Feasibility domain, with explicit scope

Preserve the historical all-mutable continuous HELOC feature representation.
Apply the observed training feature box to **optimizable control particles**,
as in the previous control solver. Do not change source/target CFs or silently
expand a box to contain them. On complete paths, report training-box violation
mass as a diagnostic, separately from declared hard constraints.

This scope follows the original generation policy and prior control solver;
it does not claim physical actionability or observed-range compliance of all
outputs. A different experiment requiring all outputs inside the training box
is infeasible with these unchanged source endpoints and must be marked so.

Require f0>=.5 for all controls and all positive-mass support points on the
21-point deployment grid, including endpoints, independently of box reporting.
Use reference-CPU true scores for final retention. No probability safety margin
below .5, mass dropping, output projection, or invalid-lambda filtering.
Threshold .7 remains diagnostic only. Report any invalid mass and minimum score.

## 7. Numerical optimization and initialization

Start from the prior numerical schedule: Adam 120 steps, lr .03, seed 20260915,
random sd .02; smooth hinge sharpness 50 and outer model-max sharpness 100.
Apply mean across lambda after the smooth population/model reduction. Exact OT
plans may be held fixed for envelope/path gradient steps; recompute exact plans,
lengths, traversal and true risk before retaining any candidate. This remains
local, nonsmooth numerical optimization, not differentiation through a globally
smooth OT optimum.

Use augmented Lagrangian handling of g_geo=L-B_geo and current-validity
inequalities. A smooth search residual may be used, but no constant .7 hinge
penalty enters the primary objective. Box-project only control variables.
Initial rho=1, dual update every 5 steps, stall check every 20; increase rho only
on documented stalled violation. Keep separate duals/traces for geometry and
current validity. Final feasibility uses exact quantities, not AL loss.

Four deterministic starts: straight-path-derived controls, previous single
control, old training-only waypoint, and small random perturbations near P1.
For straight initialization use target-conditional means of exact straight-plan
positions at progress 1/3 and 2/3 as numerical starts only; this is not an exact
split geodesic representation. Initialize the two controls by copying the
previous control/waypoint for those respective starts. Any projected or invalid
initialization is only a start, never an accepted candidate by assumption.

Choose lowest true R_path among feasible iterates/starts. Carry smaller-budget
feasible incumbents into larger budgets. If none is feasible, return an explicit
failure; never return the smallest-violation iterate as a feasible solution.
Geometry tolerance: 1e-6*max(1,D_train). Zero current-invalid support is required.

## 8. Held-out batch rule and leakage boundary

Freeze the training-defined method: K, metric, training absolute B_geo, bank,
optimizer/smoothing settings, initialization recipe and seed, source/target
rules, current-validity/domain requirements, selection/tie rule. Learned
training control coordinates are warm starts, not an out-of-sample map.

For each held-out batch solve the same constrained path problem for its local
control coordinates using only that batch's original CFs, fixed P1/f0 and the
frozen construction bank. No future scenario outcomes, labels beyond inherited
inputs, new attack search, coefficient tuning or parameter refitting is allowed.
Per-batch particle optimization is explicitly an inference-time optimization
rule; it is not a fixed transport map or a reuse of training coupling.

Before solving, check W2(P_CF_batch,P1)<=B_geo and endpoint validity. Afterward
verify all exact legs, total length and grid validity. Freeze each batch path
before scoring the expanded evaluation bank. If batch inference fails, reject
that beta/batch without changing B_geo or domain. Calibration picks lambda;
apply the same chosen lambda on the frozen test path. Do not optimize test
controls on fresh-bank evaluation outcomes. No fresh attacks originate from
cal/test data. Report batch-specific speed constants and path costs.

Beta=1 implies equality/shortest-geodesic restriction on training only. Holding
B_geo fixed means calibration/test can have positive detour slack; do not claim
beta=1 is a shortest-geodesic restriction there. Endpoint necessary checks are
in `results/004_path_preflight.json`; they do not establish path feasibility.

## 9. Preflight, controls, and execution gates

Keep original straight, old waypoint, and previous single-control artifacts as
controls, with their mismatched length/domain failures disclosed. Compare on
one shared expanded bank after matched construction-side fresh discovery.
Original task sections 10–20 specify outputs, selection, metrics and rejection.
No radius selection is invented: report fixed r=3.

Before a large replay: test the implementation on synthetic distributions,
then a small construction-only subset, estimate full-support memory and runtime,
then run full-support construction feasibility. Run held-out inference with
frozen method parameters before claiming deployment feasibility. Stop expensive
fresh discovery for an infeasible path; report exact missing cells and failures.
Use identical resources across feasible methods and budgets. Previous inspected
future scenarios remain exploratory, not untouched confirmation.

New raw outputs must use a unique directory under
`/home/xu.zhao/robust_ce_results/iclr_runs/path_level_geodesic_wro_v1/`.
This adapts the task's example output directory to the repository launcher rule.
Use the local launcher after a new versioned driver is implemented; the existing
single-budget driver does not implement this task. Do not launch an old driver
and label it the path-level method.
