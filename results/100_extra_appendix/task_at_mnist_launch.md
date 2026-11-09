Codex Task: Three priority experiments for GRACE
Goal
Run three targeted experiments that strengthen the current ICLR submission without changing the mathematical definition of GRACE.
The three experiments are:
1. MNIST update-horizon mismatch (T_design=3, varying T_eval).
2. MNIST geometry-ablation with the beta_geo constraint removed.
3. Single-run replications/variants of Table 1: current MLP setup, logistic-regression classifier, and an alternative base CF generator.
Reuse the existing repository code, splits, checkpoints, shift generators, calibration logic, metrics, and output conventions wherever possible. Do not tune on the final test future models.
Experiment 1 — MNIST update-horizon mismatch
Question
Does a GRACE population optimized assuming a 3-step future update remain robust when the realized future fine-tuning horizon is longer than 3 steps?
This is a deployment-misspecification test. It is not intended to show that T has no effect.
Fixed settings
Use the current MNIST 4→9 pipeline from the paper.
Fix:
- T_design = 3
- r = 0.05
- beta_geo = 1.02
- the current MNIST classifier architecture
- the current base CF generator / LeapFactual pipeline
- the current support-population size
- the current classifier-update learning rate and optimizer
- the current GRACE optimization settings
- the same train / calibration / test splits
- the same held-out future-shift construction
Do not recalibrate r or beta_geo for this experiment.
Evaluation horizons
Evaluate the same fixed GRACE CF population at:
T_eval in {3, 5, 10, 20, 40}
For each held-out future-shift realization:
- start from the same deployed model f0;
- run one update trajectory up to step 40;
- save checkpoints at steps 3, 5, 10, 20, and 40;
- evaluate the exact same fixed CFs at every checkpoint.
Do not independently retrain a separate future model for each T_eval if checkpoints from one trajectory can be reused.
Required metrics
At each T_eval, report:
- GRACE Future Validity (FV)
- Base-CF FV
- Proximity of the fixed GRACE CFs (constant across T_eval, but report once)
- future-model training loss at the checkpoint
- parameter displacement ||theta_t - theta_0||_2
- optional: future-model accuracy / prediction drift if already easy to compute
Preserve the same aggregation structure used by the current paper.
Output
Create:
- results/exp1_mnist_T_mismatch/raw.csv
- results/exp1_mnist_T_mismatch/summary.csv
- results/exp1_mnist_T_mismatch/fv_vs_Teval.pdf
- results/exp1_mnist_T_mismatch/train_loss_vs_Teval.pdf
- results/exp1_mnist_T_mismatch/parameter_displacement_vs_Teval.pdf
- results/exp1_mnist_T_mismatch/summary.md
The summary should explicitly state whether performance degrades gradually or sharply as T_eval moves beyond T_design=3.
Experiment 2 — MNIST ablation without the geometry constraint
Question
What happens if GRACE is allowed to optimize the population objective without the two-anchor geometry restriction?
This experiment should isolate the practical role of the beta_geo constraint.
Compared configurations
Use MNIST 4→9 and keep everything identical except the geometry constraint.
Constrained GRACE
Use:
- T = 3
- r = 0.05
- beta_geo = 1.02
Unconstrained GRACE
Use:
- the same initialization from the same Base CF population
- the same T = 3
- the same r = 0.05
- the same nominal loss and update-aware sensitivity penalty
- the same number of support-optimization iterations
- the same optimizer / step-size settings where meaningful
But remove the two-anchor constraint
W_CF(P_CF, Q) + W_CF(Q, P_1) <= B_geo.
The unconstrained version should optimize the GRACE surrogate directly over support locations, subject only to the existing valid latent / feature-domain constraints required by the implementation.
Do not replace the removed geometry constraint with another regularizer.
Required metrics
Compare at least:
- Future Validity (FV)
- Proximity
- nominal population loss F_Q(theta_T^nom)
- update-aware sensitivity ||s_Q||_*
- surrogate objective tilde Phi_r
- W_CF(P_CF, Q)
- W_CF(Q, P_1)
- total detour W_CF(P_CF,Q)+W_CF(Q,P_1)
Also save several decoded qualitative examples for both settings using the same factuals.
The point of this ablation is to see whether unconstrained optimization attains lower robust loss / higher FV by moving much farther away from the Base CFs or favorable-class geometry.
Output
Create:
- results/exp2_mnist_no_geometry/metrics.csv
- results/exp2_mnist_no_geometry/examples/
- results/exp2_mnist_no_geometry/summary.md
The summary must avoid calling the unconstrained model "better" solely because FV increases; explicitly discuss the robustness–proximity / geometry trade-off.
Experiment 3 — Single-run Table 1 replications and robustness checks
General rule
Rerun Table 1 with one fixed seed only for each variant below. This is a diagnostic / robustness check, not a replacement for the multi-run main table.
Use the four tabular datasets from Table 1:
- HELOC
- Diabetes
- German Credit
- FICO
Use the same held-out shift families and the same FV / Proximity definitions as the current paper.
Use one explicitly recorded seed for all variants wherever possible.
Experiment 3A — Current Table 1 setup, one run
Rerun the current Table 1 pipeline once with the existing main classifier architecture and current Wachter Base CF generator.
Include:
- T-Rex
- ROAR
- PROPLACE
- ElliCE
- Wachter Base
- GRACE
Purpose:
- verify that the current implementation reproduces the qualitative ranking and approximate scale of the reported main results in a single deterministic run;
- provide a clean reference for 3B and 3C.
Save:
- results/exp3_table1_single_run/current_mlp.csv
Experiment 3B — Logistic-regression classifier
Rerun the same Table 1 experiment using logistic regression as the deployed / updated classifier, following the repository's existing logistic-regression setup if one already exists.
Keep everything else as close as possible to 3A:
- same datasets and splits
- same future-shift construction
- same evaluation metrics
- same baseline implementations
- same calibration protocol
Each method may use its normal method-specific hyperparameters, calibrated on the calibration split exactly as in the paper.
Save:
- results/exp3_table1_single_run/logistic_regression.csv
Experiment 3C — Alternative Base CF generator
Use a second Base CF generator for GRACE to test whether the population-level wrapper is tied specifically to Wachter CFs.
Recommended generator: DiCE
Use DiCE as the preferred alternative Base CF generator, provided the repository can support it cleanly.
Why DiCE:
- it is a widely recognized tabular counterfactual baseline;
- it is methodologically distinct from the Wachter objective;
- it supports continuous and categorical tabular features;
- it is a credible test of the claim that GRACE is a post-hoc wrapper rather than a Wachter-specific method.
For compatibility with GRACE's one-CF-per-factual population:
- generate multiple valid DiCE candidates per factual if this is the natural DiCE interface;
- select the valid candidate with minimum factual-to-CF proximity using the same normalized feature representation / ground cost used by the experiment;
- if the repository already has a standard single-CF DiCE configuration, use that instead and document it.
If DiCE is already implemented in the repository, reuse that implementation. Do not reimplement it from scratch unless necessary.
For this variant:
- keep the main MLP classifier from 3A;
- replace Wachter Base CFs with DiCE Base CFs for GRACE;
- rerun GRACE calibration normally on the calibration split;
- report the DiCE Base row and the resulting GRACE-on-DiCE row.
The robust baselines (T-Rex, ROAR, PROPLACE, ElliCE) should retain their own original construction procedures; they do not need to be forced to use DiCE.
Save:
- results/exp3_table1_single_run/dice_base_generator.csv
Combined output for Experiment 3
Create:
- results/exp3_table1_single_run/combined_summary.csv
- results/exp3_table1_single_run/summary.md
The summary should answer:
1. Does the one-run current setup reproduce the overall Table 1 pattern?
2. Does GRACE remain competitive when the classifier is logistic regression?
3. Does GRACE remain effective when initialized from DiCE rather than Wachter?
Do not overinterpret single-seed differences.
Implementation constraints for all experiments
- Reuse existing code paths instead of duplicating training logic.
- Do not change the mathematical definition of GRACE except for the explicit no-geometry ablation in Experiment 2.
- Do not tune any hyperparameter on final-test future models.
- Cache future-model checkpoints where possible so methods are evaluated on identical realized models.
- Record all random seeds.
- Preserve the current data preprocessing and normalization.
- Keep Base CFs and refined CFs fixed during future-model evaluation.
- Log all relevant configuration values into the output directory.
- If any requested configuration is unsupported by the current codebase, document the exact obstacle before making a substitute choice.
First step before coding
Inspect the repository and write a short implementation plan identifying:
1. the MNIST T=3 update implementation and the code that creates future-model checkpoints;
2. where r and beta_geo enter MNIST GRACE optimization;
3. the exact augmented-Lagrangian / geometry-constraint code that must be disabled for Experiment 2;
4. the script that currently produces Table 1;
5. the existing logistic-regression experiment path;
6. whether DiCE is already implemented or available as a dependency;
7. the exact seed(s) currently used by the main experiments.
Do this inspection before editing code or launching long jobs.