We need a careful HELOC-only comparison between two ways of realizing the
same learned GRACE robust control population Q as factual-level
counterfactuals.

IMPORTANT:
- Do NOT retrain or redesign the GRACE population optimizer unless absolutely
  necessary.
- Reuse the newest saved HELOC artifacts, especially the seed-1 continuous
  single-budget WRO runs.
- Hold fixed the learned robust control population Q for each beta_geo.
- The purpose is to isolate the factual-level realization / assignment step.
- Do not silently change preprocessing, base CFs, future-model banks,
  calibration settings, or evaluation future shifts.
- Audit the repository first and identify the exact scripts/artifacts used by
  the current HELOC pipeline before modifying anything.

============================================================
BACKGROUND
============================================================

The current HELOC implementation first learns a free population-level control
distribution

    Q = sum_j b_j delta_{w_j}

by minimizing the frozen finite-bank robust objective

    Phi_hat_r(Q)
      = max_{a in A_r} sum_j b_j [0.5 - f_a(w_j)]_+

subject to

    W2(P_CF, Q) + W2(Q, P1)
      <= beta_geo * W2(P_CF, P1).

The particle indices of Q do NOT inherently correspond to factual identities.

For the latest HELOC seed-1 setup:
- M = 3336 equally weighted Q particles.
- 22 continuous features.
- frozen construction bank: 5927 updated classifiers.
- data uncertainty radius approximately
      r = 3 * b_ref ≈ 8.1888.
- full model updates start from frozen f0, reset optimizer state, and run
  AdamW for 8 epochs.
- beta_geo in {1.00, 1.05, 1.10, 1.20}.
- Q optimization uses the newest continuous solver and exact discrete OT
  feasibility checks.
- Use the saved raw Q supports rather than rerunning them if compatible
  artifacts already exist.

The current/old realization constructs two OT plans

    P_CF -> Q
    Q    -> P1

joins them through shared Q control particles, obtains weighted transport atoms

    Base CF z_i -> control w_j -> favorable reference p_k,

and realizes factual-level CFs along the resulting piecewise-linear transport
path. Preserve the exact existing implementation of this OLD method. Do not
approximate or rewrite it from memory.

We want to compare this against a simpler NEW realization:

    P_CF -> Q

only.

For the NEW method, solve an optimal assignment / optimal transport plan from
the Base CF population P_CF to the already learned Q. Since the HELOC run uses
the same number of equally weighted Base CFs and Q particles, first determine
whether the discrete W2 solution can be represented as a one-to-one minimum
cost assignment. If yes, use the exact optimal assignment

    sigma* = argmin_sigma sum_i ||z_i - w_{sigma(i)}||^2

with the same feature representation and ground metric used by the GRACE
geometry.

Then assign factual x_i, whose Base CF is z_i, the robust support

    CF_new_i = w_{sigma*(i)}.

This must NOT change Q: after assignment, the empirical distribution of all
CF_new_i must be exactly the learned Q up to ordering.

If the implementation's OT solver returns a fractional transport plan instead
of a permutation despite equal uniform masses, either:
1. obtain an exact minimum-cost one-to-one assignment using the corresponding
   cost matrix (e.g. linear_sum_assignment), or
2. clearly explain why that is inconsistent with the current discrete setup.

Do not silently collapse fractional mass with barycentric averaging.

============================================================
CRITICAL CURRENT-VALIDITY ISSUE
============================================================

The raw Q optimizer did NOT use f0 as a constraint. Therefore direct Q supports
may fail current validity:

    f0(w_j) >= 0.5.

This must be measured explicitly.

Do NOT hide this issue and do NOT modify Q before first evaluating the pure
direct-assignment method.

Evaluate the following variants:

A. OLD
   Existing two-leg transport realization:
       P_CF -> Q -> P1
   using the exact current code, lambda/path selection, weights, and any
   existing current-validity handling that is genuinely part of the evaluated
   old pipeline.

B. NEW-DIRECT
   One-to-one optimal assignment
       P_CF -> Q
   and directly return assigned Q supports.

C. NEW-DIRECT-VALID (only if NEW-DIRECT has nontrivial current invalidity)
   Use the smallest possible validity correction that preserves the new
   interpretation.

   For each pair z_i -> w_sigma(i):
   - z_i is the current-valid Base CF.
   - If w_sigma(i) is current-valid, keep it unchanged.
   - Otherwise move only along the segment from w_sigma(i) back toward z_i
     and find the closest point to w_sigma(i) satisfying f0(.) >= 0.5
     (e.g. binary search / backtracking).
   - Do not use P1.
   - Do not re-optimize Q.
   - Report how much this correction changes the learned Q distribution.

This C variant is a diagnostic, not a replacement for B.

============================================================
FAIR COMPARISON
============================================================

Run the comparison on exactly the same:
- HELOC factual instances;
- Base CFs;
- P1;
- learned Q for each beta_geo;
- preprocessing;
- f0;
- construction bank;
- held-out future-model evaluation set;
- random seeds / realizations.

Do not select a different beta for each realization method when making the
primary comparison.

Primary comparison:
for every beta_geo in

    {1.00, 1.05, 1.10, 1.20}

compare OLD vs NEW-DIRECT, and NEW-DIRECT-VALID if needed.

Also report the currently selected/calibrated beta as a highlighted row.

============================================================
METRICS
============================================================

For each beta and method report:

1. Current Validity
   fraction of deployed factual-level CFs satisfying

       f0(CF_i) >= 0.5.

2. Future Validity
   on exactly the same held-out future models used by the existing paper.

   Report:
   - overall mean Future Validity;
   - std / variability consistent with current experiments;
   - if already available, per shift type and severity.

3. Recourse Cost
   exactly the paper's factual-to-final-CF metric:

       mean_i ||x_i - CF_i||_2

   using the same normalized/raw representation convention as the current
   paper.

4. Base-to-realized displacement

       mean_i ||z_i - CF_i||_2.

   This is important for understanding how much the realization step modifies
   the original recourse.

5. Population robust risk on the final realized CF population

       max_{a in A_train}
       mean_i [0.5 - f_a(CF_i)]_+.

   This is distinct from Phi_hat_r(Q) if OLD or validity correction outputs a
   population different from Q.

6. Fresh-bank / held-out robust risk if the repository already supports it.

7. Distance from final deployed empirical population Q_deploy to learned Q:

       W2(Q_deploy, Q).

   For NEW-DIRECT this should be numerically zero (up to tolerance), which is
   an important sanity check.

8. Geometry diagnostics:

       W2(P_CF, Q)
       W2(Q, P1)
       W2(P_CF, Q) + W2(Q, P1)
       beta_geo * D

   These should be identical across realization methods because Q itself is
   held fixed.

9. Assignment cost for NEW:

       W2(P_CF, Q)

   and verify that the chosen one-to-one assignment attains the same discrete
   OT cost within numerical tolerance.

10. Runtime of the realization stage only:
    OLD realization versus NEW assignment.
    Do not include Q training time in this comparison.

============================================================
WEIGHTED OLD OUTPUTS
============================================================

The OLD method may produce multiple weighted realization atoms for one factual.

Do not make an unfair comparison.

First inspect how the current evaluation computes:
- Future Validity;
- Recourse Cost;
- current validity

for these weighted outputs.

Preserve that as "OLD-native".

Additionally, if meaningful, create an apples-to-apples deterministic OLD-1CF
variant with exactly one CF per factual, using the current code's intended
selection rule. Do not invent a new aggregation rule without clearly labeling
it.

The primary table should clearly state whether each row is weighted or
one-CF-per-factual.

============================================================
SANITY CHECKS
============================================================

Verify:

1. NEW-DIRECT output marginal equals Q exactly:
       empirical(CF_new) == Q
   up to permutation/numerical tolerance.

2. No factual identities are lost:
   each assigned CF maps back to the factual via its Base CF z_i.

3. Assignment is globally optimal, not nearest-neighbor independently chosen
   with duplicate Q supports.

4. No test future models influence the assignment.

5. P1 is NOT used in NEW deployment; it was already used during the learning
   of Q through the two-anchor geometry.

6. Do not accidentally recompute or alter Q when changing realization.

7. The OLD method must reproduce its previously saved HELOC numbers before
   using the comparison.

============================================================
OUTPUTS
============================================================

Create a results directory, e.g.

    results/heloc_realization_comparison/

Save:

1. CSV with one row per
       beta_geo x realization_method x seed/evaluation realization

2. Summary CSV with means/std.

3. A compact table:

    beta | method | Current Validity | Future Validity | Cost |
           Base->CF | train-bank risk | W2(Q_deploy,Q) | runtime

4. Plots:
   a. Future Validity vs Recourse Cost
   b. beta_geo vs Future Validity
   c. beta_geo vs Recourse Cost
   d. beta_geo vs Current Validity
   e. train-bank risk of learned Q vs risk of deployed population
   f. distribution of per-factual recourse costs for OLD vs NEW

5. A short markdown report explaining:
   - exact implementation paths used;
   - whether the old results reproduce;
   - whether direct Q assignment preserves robustness better;
   - whether it damages current validity;
   - whether recourse cost improves or worsens;
   - whether NEW-DIRECT requires the minimal validity correction;
   - whether the simplified deployment is empirically preferable.

============================================================
DECISION CRITERIA
============================================================

Do NOT simply pick the method with the highest Future Validity.

Interpret the result as follows:

NEW-DIRECT is attractive if:
- its Current Validity is effectively 100%, or needs negligible correction;
- its Future Validity is at least comparable to OLD;
- its deployed population retains the low robust risk of Q;
- its Recourse Cost is comparable or lower;
- and realization is materially simpler.

NEW-DIRECT-VALID is attractive if:
- correction is small;
- it restores ~100% Current Validity;
- Future Validity remains comparable or better than OLD;
- and W2(Q_deploy,Q) remains small.

OLD remains preferable if:
- direct assignment causes substantial current invalidity;
- validity correction significantly destroys Q's robust risk;
- or OLD achieves meaningfully better Future Validity / Recourse Cost.

The key scientific question is:

    Does directly assigning the learned robust population Q back to factuals
    via optimal P_CF -> Q transport preserve its population robustness while
    producing cheaper and simpler individual recourse than the existing
    P_CF -> Q -> P1 path realization?

Do not claim a winner until the metrics above are computed.

At the end, print:
- the exact commands needed to reproduce the comparison;
- the artifact paths;
- a concise numerical conclusion;
- any inconsistencies between the paper's current Method description and the
  actual implementation discovered during the audit.