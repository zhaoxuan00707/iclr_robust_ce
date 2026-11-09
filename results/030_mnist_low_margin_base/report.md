# MNIST controlled lower-margin Base experiment

The strong classifier, decoder, factuals, references and update data are fixed. Four Base conditions and four radii were evaluated with800 support updates each, on eight GPUs. Results are retrospective single-checkpoint diagnostics, without validity correction or a certified robustness guarantee.

## Main result and visual interpretation

The manipulation worked: near-boundary Base test FV is **77.39%**, down from **84.24%** for the unchanged original Base, while current validity remains100%. Its initial sensitivity rises from1.0636 to3.6303. Nominal-only refinement reaches97.92%; calibrated r=.25 reaches100.00%. This splits recovery into20.53 percentage points from nominal optimization and only2.08 further points from the sensitivity penalty. Do not attribute the full Base-to-final gain to robustness regularization.

Near-boundary factual pixel cost is5.3584 for Base,5.4186 for nominal-only, and5.4718 for selected GRACE. At r=.25, direct CF-minus-Base mean image L2 grows from.283568 for original to.384345 for near-boundary (about35.5% larger). Direct r.25-minus-r0 movement grows from.137360 to.172804 (about25.8% larger). The refined.52 control has similar or larger movement than.5001; visual displacement is not monotone in the requested margin under different population anchors/budgets and finite nonconvex solves.

Visual inspection of the fixed matched panel shows localized stroke/intensity adjustments, still subtle rather than dramatic shape changes. Shared-scale difference maps reveal the changes without implying perceptual improvement. Lower robustness of Base is established for these scenarios; better-looking digits or a general causal guarantee are not.63/64 test cases reach each refined target, with one unchanged positive-start fallback included in every full-population score. The original Base's84.24% FV here differs from task015's85.55% because the future random scenarios changed.

## Did Base quality change?

The operational intervention is lower classifier margin, not an imposed visual corruption or a guarantee of lower future validity. Construction results (original is an unchanged control, not a boundary-search success):

```
condition split  successes  total  probability_min  probability_mean  probability_max  adequate_manipulation
 original   cal         64     64         0.521360          0.603490         0.743970                   True
 original  test         64     64         0.524521          0.621971         0.975871                   True
   p05001   cal         64     64         0.500100          0.500101         0.500102                   True
   p05001  test         63     64         0.500100          0.507535         0.975871                   True
    p0505   cal         64     64         0.505000          0.505001         0.505003                   True
    p0505  test         63     64         0.505000          0.512358         0.975871                   True
     p052   cal         64     64         0.520000          0.520001         0.520002                   True
     p052  test         63     64         0.520000          0.527124         0.975871                   True
```

Unmodified fallback cases remain in the main64-case population. An arm is an inadequate manipulation if fewer than48 cases reach its requested margin on either split. successful_subset_current.csv and the per-model successful_subset_FV columns provide explicitly conditional diagnostics, not replacement headline results. CPU-recomputed score distributions are in current_probabilities.csv; saved GPU construction scores determine target-tolerance success against the float32-represented target (for .505 this is .5049999952).

## Robustness recovery and costs

All headline FV values average the same36 test scenarios. Radius is selected independently per condition using36 calibration scenarios; test outcomes do not change it.

```
condition  Base_FV  nominal_FV  r025_FV  selected_r  selected_FV  Base_cost  nominal_cost  selected_cost  Base_sensitivity  nominal_sensitivity  r025_sensitivity  nominal_gain_pp  robust_increment_r025_pp
 original 0.842448    0.980903      1.0        0.25          1.0   5.403529      5.445229       5.487764          1.063635             0.010229          0.000666        13.845486                  1.909722
     p052 0.793403    0.980035      1.0        0.25          1.0   5.366915      5.423845       5.480843          3.024572             0.010380          0.000547        18.663194                  1.996528
    p0505 0.780816    0.979167      1.0        0.25          1.0   5.360509      5.419914       5.474301          3.481146             0.010380          0.000572        19.835069                  2.083333
   p05001 0.773872    0.979167      1.0        0.25          1.0   5.358408      5.418637       5.471775          3.630260             0.010388          0.000578        20.529514                  2.083333
```

Separate nominal refinement (Base→r0) from incremental robust refinement (r0→positive r). Raw Phi_r changes objective across radii; compare common_r025_surrogate for a shared reference radius. Feasibility is a population geometry constraint, not an individual semantic-quality guarantee.

Calibration grid:
```
condition    r   cal_FV  pixel_cost  nominal  sensitivity  surrogate  feasible  historical_stop_step  final_objective_change
 original 0.05 0.994358    5.643586 0.000040     0.002584   0.000169      True                   308            1.688168e-07
 original 0.10 0.997396    5.650884 0.000027     0.001585   0.000186      True                   313            1.798035e-07
 original 0.25 0.999566    5.664141 0.000016     0.000776   0.000210      True                   312            1.932058e-07
 original 0.00 0.957465    5.623430 0.000146     0.012691   0.000146      True                   281            1.434528e-07
     p052 0.05 0.994358    5.624130 0.000040     0.002606   0.000170      True                   308            1.776789e-07
     p052 0.10 0.997396    5.631857 0.000027     0.001579   0.000185      True                   310            1.771114e-07
     p052 0.25 0.999566    5.654486 0.000016     0.000776   0.000210      True                   261            1.484150e-07
     p052 0.00 0.956597    5.603742 0.000148     0.012861   0.000148      True                   294            1.472945e-07
    p0505 0.05 0.994358    5.619444 0.000040     0.002612   0.000171      True                   305            2.056186e-07
    p0505 0.10 0.997396    5.627046 0.000027     0.001589   0.000186      True                   310            1.768785e-07
    p0505 0.25 0.999566    5.648169 0.000017     0.000815   0.000220      True                   242            1.629815e-07
    p0505 0.00 0.956597    5.599058 0.000148     0.012889   0.000148      True                   298            1.482840e-07
   p05001 0.05 0.994358    5.617922 0.000040     0.002615   0.000171      True                   305            1.546432e-07
   p05001 0.10 0.997396    5.625466 0.000028     0.001591   0.000187      True                   315            1.794979e-07
   p05001 0.25 0.999566    5.645805 0.000017     0.000831   0.000225      True                   244            1.658191e-07
   p05001 0.00 0.956597    5.597508 0.000149     0.012904   0.000149      True                   299            1.494045e-07
```

Final test diagnostics:
```
condition         method  selected  feasible  current_validity  FV_mean  pixel_cost  sensitivity  historical_stop_step  final_objective_change
 original r0.05_beta1.02     False      True               1.0 0.998698    5.464159     0.002197                 266.0            1.357112e-07
 original  r0.1_beta1.02     False      True               1.0 0.998698    5.471607     0.001361                 277.0            1.487788e-07
 original r0.25_beta1.02      True      True               1.0 1.000000    5.487764     0.000666                 275.0            1.580629e-07
 original    r0_beta1.02     False      True               1.0 0.980903    5.445229     0.010229                 246.0            1.162625e-07
 original           Base     False      True               1.0 0.842448    5.403529     1.063635                   NaN                     NaN
     p052 r0.05_beta1.02     False      True               1.0 0.998698    5.443090     0.002207                 272.0            1.368171e-07
     p052  r0.1_beta1.02     False      True               1.0 0.998698    5.451139     0.001340                 260.0            1.370790e-07
     p052 r0.25_beta1.02      True      True               1.0 1.000000    5.480843     0.000547                 225.0            9.451469e-08
     p052    r0_beta1.02     False      True               1.0 0.980035    5.423845     0.010380                 259.0            1.206936e-07
     p052           Base     False      True               1.0 0.793403    5.366915     3.024572                   NaN                     NaN
    p0505 r0.05_beta1.02     False      True               1.0 0.998698    5.439178     0.002214                 273.0            1.351291e-07
    p0505  r0.1_beta1.02     False      True               1.0 0.998698    5.447025     0.001352                 274.0            1.531589e-07
    p0505 r0.25_beta1.02      True      True               1.0 1.000000    5.474301     0.000572                 212.0            4.372851e-08
    p0505    r0_beta1.02     False      True               1.0 0.979167    5.419914     0.010380                 260.0            1.203298e-07
    p0505           Base     False      True               1.0 0.780816    5.360509     3.481146                   NaN                     NaN
   p05001 r0.05_beta1.02     False      True               1.0 0.998698    5.437905     0.002216                 271.0            1.393200e-07
   p05001  r0.1_beta1.02     False      True               1.0 0.998698    5.445649     0.001357                 269.0            1.456210e-07
   p05001 r0.25_beta1.02      True      True               1.0 1.000000    5.471775     0.000578                 233.0            1.078442e-07
   p05001    r0_beta1.02     False      True               1.0 0.979167    5.418637     0.010388                 261.0            1.213193e-07
   p05001           Base     False      True               1.0 0.773872    5.358408     3.630260                   NaN                     NaN
```

## Method and provenance

Task tasks/016_mnist_low_margin_base.md; config experiments/configs/mnist_low_margin_base.json; core experiments/src/lift_ce/experiments/mnist_low_margin_base.py. Raw run `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/20260924_v1`. Parent strong-arm population `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_weaker_classifier/20260924_v1/strong/population.pt` is reused byte-for-byte at the root; original arm Base/code tensors are unchanged. Other arms follow the same latent line from Enc(factual) to original Base code, first sampled crossing on4096 intervals then24 bisections at p=.52/.505/.5001. Bracket validity and1e-4 probability tolerance are checked. Already-positive starts, missing brackets or numerical failures keep the original valid Base; no IDs are discarded. The search does not assume monotonicity or prove a globally earliest/minimal-distance crossing. Precise.52 controls for coarse generator overshoot.

Classifier checkpoint `/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt`, latent64-dimensional decoder,64 factuals and64 equal-mass Q supports per split,64 P1 references and256 update images are fixed. Update is the complete T=3 full-batch SGD lr.01 with BCE, frozen BN/dropout disabled. Smooth probability hinge t=.05 appears in nominal loss and full outer sensitivity derivative; empirical RMS dual is sqrt(256)*norm(dF/dh), with training perturbations in pixel space. Exact OT geometry and one-to-one assignment operate in latent space. D and budget1.02*D vary with Base and are reported.

The ALM equations retain primal20/rho1, but this predeclared experiment runs exactly800 updates, with no early stopping or previous-iterate substitution. Historical tolerance-crossing steps are logged only. Final objective/constraint gradients and changes are diagnostics, not KKT/global-optimality certificates; converged=false explicitly avoids declaring convergence from a fixed budget. Calibration feasible 16/16; test feasible 16/16. Failed final iterates are retained and excluded from deployment; no repair.

One common set of72 newly generated future checkpoints serves every arm: cal seeds71000+rep/test81000+rep, three rotation/translation/noise families with nominal strengths15/2/.08, severities .5/1/1.5/2, three realizations. Model updates use the same T=3 operator. Duplicate distribution audit: {'cal_unique': 28, 'test_unique': 28, 'cross_split_identical': 4, 'scenarios_per_split': 36}. These are not72 independent dataset replicates. Old inspected assets remain retrospective, and short-update FV does not establish arbitrary-retraining robustness or independent semantic digit quality. Historical85.55% Base FV used different random scenarios and is not an expected value here.

## Checks and artifacts

15 tests pass, including nonmonotonic crossing, valid upper endpoint, positive-start/unbracketed fallback and discontinuous-path rejection. Actual CNN directional-gradient checks are saved per condition. Independent reporting checks32 fixed800-update traces, exact geometries/assignments, decoded images/current validity/cost, initial/final smooth-objective arithmetic, input and model hashes, original/fallback preservation, direct image differences, frozen choices and GPUs0–7. Each worker caps its allocator at6.5%; no other jobs are stopped.

Figures: comparison.pdf joins population sensitivity/FV/cost and measured image differences; qualitative.pdf matches four conditions for first8 fixed factuals; all64_qualitative.pdf includes every case and condition; per-condition first8 panels also saved. Signed difference maps share one scale (difference_scale.json), separate from normal grayscale images. Metrics are population-level, not mislabeled as per-image sensitivity. No example filtering for attractive outcomes. Large raw outputs remain external. No manuscript claims changed.

```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/064_run_mnist_low_margin_base.py --output /home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/NEW_RUN_ID
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mnist_low_margin_mpl /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/065_report_mnist_low_margin_base.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/NEW_RUN_ID
```
Use a new raw directory and separate checkout for report reproduction; existing report is protected.

## Two-heatmap visual supplement

User-requested layout: Factual | Base | r0 | r.25 | (r.25 minus r0) | (r0 minus Base).
The last two columns isolate additional robust refinement and nominal refinement.
Both use signed pixel differences and the same symmetric scale across all64 cases
and four conditions; red means increased intensity, blue decreased. No experiment
was rerun. New PDFs: figures/mnist_low_margin_base/qualitative_two_heatmaps.pdf
(first8 cases, four matched conditions per page) and near_boundary_two_heatmaps.pdf
(first8 near-boundary cases on one page). Scale/provenance: two_heatmaps.json.
Script: experiments/066_plot_mnist_two_heatmaps.py. Original figures preserved.

## Same-Q gradient alignment supplement

Direct latent-space gradients were evaluated for nominal loss and sensitivity at
Base (cal/test) and saved r0/r.25 test Q, for all4 conditions. See
gradient_alignment/report.md and summary.csv. Test cosines range.98550–.99439;
near-boundary Base cosine.994392, with weighted r.25 sensitivity-gradient norm
12.8768 times nominal. The directions are strongly aligned but the penalty is
not negligible; do not infer equivalence or full-trajectory alignment.16 gradient
decomposition and input-hash checks pass; no experiment output was modified.
