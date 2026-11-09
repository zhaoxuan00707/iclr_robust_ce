

(\widetilde R_{\rm path}) is a smooth search surrogate;

(\nu_{\rm geo}\ge0) is a dual variable;

(\rho_{\rm geo}) is an optimization parameter, not a modeling tradeoff.

Use Adam or the existing differentiable optimizer for the path-control supports.

Final candidate selection must use:

true hinge;

exact / official (W_{\rm CF});

official current-validity checks.

10. Warm starts

Use several deterministic / reproducible initializations:

A. Straight-geodesic initialization

Initialize controls on the original Wasserstein path.

B. Previous single-budget control initialization

Use the previously learned (Q_W) where compatible.

C. Old waypoint initialization

Use the previous training-only waypoint as one intermediate control where compatible.

D. Small random perturbations

Use a small number of random initializations for numerical robustness.

Choose the final retained path using the lowest true training-bank path objective among exactly feasible candidates.

Do not use validation or test outcomes to choose an initialization.

11. Fresh adversarial discovery after the path is frozen

After learning and freezing each candidate path, run fresh strengthened adversarial searches against representative (\lambda) values.

Use the existing strengthened attack procedure.

Persist:

shifted training distributions;

perturbations;

full-update models;

source (\beta_{\rm geo});

source path;

source (\lambda);

budget-feasibility measurements.

Merge compatible attacks into an expanded bank

[
\mathcal A_r^{\rm fresh}.
]

Do not feed the fresh attacks back into path training in this first experiment.

An iterative min-max version can be a later experiment.

12. Deployment-point selection

After the path is frozen and the shared fresh bank is built, evaluate

\max_{a\in\mathcal A_r^{\rm fresh}}
\mathbb E_{Q_\lambda}
[\tau-f_a(z)]_+.
]

Then select

[
\widehat\lambda_r
\in
\arg\min_{\lambda\in\Lambda}
\widehat\Phi_r^{\rm bank}(\lambda).
]

Use the same (\lambda) grid for all methods.

Do not use:

recourse cost;

final-test future validity;

preferred interior solutions;

post-hoc plateau rules

to choose (\lambda).

13. Held-out batch transfer

This is a required part of the task.

The previous method failed because a training-feasible control did not yield budget-preserving held-out refinement.

Implement a batch-level constrained refinement rule for held-out CF populations.

For a held-out batch, keep fixed:

(f_0);

the learned path construction parameters;

(W_{\rm CF});

(B_{\rm geo});

target-anchor definition;

all training-derived attack models.

Do not reuse training support coordinates blindly.

Instead solve the declared constrained refinement on the held-out batch, using the same frozen method specification.

Verify for every held-out batch:

[
L_{\rm path}^{\rm heldout}
\le
B_{\rm geo},
]

and

[
Q_\lambda^{\rm heldout}(F_0)=1.
]

If the current representation cannot support this without retraining method parameters, document the limitation precisely.

Do not claim deployment feasibility until this passes.

14. Required controls

Compare:

Original straight path / (\beta_{\rm geo}=1).

Previous single-control geodesic-excess solver.

Old heuristic waypoint.

New path-level constrained solver.

Optional:

Path-level constrained solver with (K=3) or (K=4).

Use matched attack banks and evaluation scenarios whenever compatible.

15. Metrics

For each method and geometry budget report:

Robustness

training-bank hinge;

fresh-bank hinge;

mean future hinge;

worst future hinge;

mean future validity;

worst future validity.

Feasibility

exact path length;

geometry-budget slack;

current-valid mass over the full deployment grid;

minimum current-model score;

held-out budget violation;

held-out current-validity violation.

Cost

factual-to-final-CF recourse cost;

path length;

endpoint distances.

Do not substitute path length for user recourse cost.

16. Main scientific questions

Answer:

Q1

Does path-level geometry enforcement remove the previous deployment-budget violation?

Q2

Does path-level current-validity enforcement remove invalid intermediate outputs?

Q3

Does the robustness gain from (\beta_{\rm geo}>1) survive fresh attack discovery?

Q4

Does the robustness / future-validity trend survive on held-out batches while respecting the same budget?

Q5

How much additional benefit comes from (K>2) controls?

Q6

Does the new path-level solver outperform the old heuristic waypoint at comparable recourse cost?

17. Theory-facing diagnostics

Verify numerically:

[
\beta_{\rm geo}=1
]

reduces to the shortest-geodesic restriction up to representation / solver tolerance.

For

[
B_1\le B_2,
]

check that the best feasible training objective does not worsen beyond numerical tolerance.

If it does, report local optimization failure rather than a theoretical contradiction.

Measure the approximation errors separately:

finite attack-bank error proxy;

geometry-feasibility residual;

numerical optimization residual;

held-out transfer residual.

Do not claim an unrestricted global WRO guarantee.

18. Failure criteria

Mark the experiment as not deployment-feasible if any selected output has:

[
Q_\lambda(F_0)<1-\text{tolerance},
]

or

[
L_{\rm path}>B_{\rm geo}+\text{tolerance}.
]

Do not hide or average away constraint violations.

If held-out batches systematically violate the budget, treat that as a method-level realization issue requiring further work.

19. Output directory

Write everything to a new directory such as

robust_ce_results/path_level_geodesic_wro_v1/<run_id>/

Save:

resolved_config.json
instance_manifest.json
attack_bank_manifest.csv
geometry_budget_grid.csv
path_control_points.*
path_couplings.*
optimization_trace.csv
dual_trace.csv
path_feasibility.csv
heldout_feasibility.csv
fresh_attack_manifest.csv
cross_eval_matrix.csv
lambda_curves.csv
final_metrics.csv
report.md

Do not overwrite previous experiments.

20. Final response format

Keep the final response concise.

Use:

IMPLEMENTED

What changed from the previous control-level solver.

TRAINING FEASIBILITY

Path length and current-validity compliance.

HELD-OUT FEASIBILITY

Whether budget and validity survive batch transfer.

ROBUSTNESS

Straight vs previous control-level vs old waypoint vs new path-level solver.

FRESH-ATTACK CHECK

Whether gains survive expanded attacks.

COST

Matched recourse-cost comparison.

LIMITATIONS

Finite bank, finite support, numerical OT, no unrestricted-global guarantee.

CORE CONCLUSION

Under 150 words.

Do not claim success unless the selected held-out path is both geometry-budget feasible and current-valid.