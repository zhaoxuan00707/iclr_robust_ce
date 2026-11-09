# Identity-feasible GENKI GRACE search

The feasible block search improved saved-image average future validity from 45% to 50% (Base: 31.67%) and current-model validity from 4/8 to 5/8. All eight outputs pass the unchanged preservation thresholds. Cases 5 and 7 changed; the other six PNGs, including fixed cases 3 and 8, are byte-identical to previous GRACE. Minimum ArcFace stays 0.800367; mean ArcFace changes from 0.919918 to 0.914881. Saved-image bank hinge falls from 0.144902 to 0.142230. This demonstrates a modest improvement within our identity limits, not consistently robust CFs or unchanged human identity.

Raw run: `/home/xu.zhao/robust_ce_results/genki_preservation_search/20260917_feasible_v1`. Parent: `/home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1`.

The two previously limiting faces (zero-based indices 2 and 7; cases 3 and 8 in the PDF) remain at their original feasible GRACE edits. They remain in every population objective and evaluation. Six other faces receive gradient steps on max-model mean-population hinge with other faces fixed at the incumbent. Every proposed step passes ArcFace>=0.8, FaceNet>=0.8, LPIPS<=0.15, outside RMS<=0.04 and beta=3 exact geometry before acceptance; it must also improve bank risk. Steps start at 0.05, grow to at most 0.2 on acceptance, halve on rejection, stop below 0.00005, and run at most 24 attempts. Feasible per-face candidates are combined by greedy coordinate selection on the same global bank objective under exact geometry. This is a feasible block search, not a jointly differentiated identity penalty or global optimum. Final faces are independently rendered and screened on eight GPUs.

Radius 2, beta 3, original Base, four-channel editor, reference distribution, bank, and preservation thresholds remain fixed. The previous beta 1.2 and beta 3 results were identical. No future outcomes select candidates. Primary evaluation uses the exact saved PNGs uniformly on CPU float32, caching by image hash. Native LPIPS/outside diagnostics retain their original definitions; the mask is an ellipse, not background segmentation.

```
         method  current_validity  future_validity  ArcFace_mean  ArcFace_min  LPIPS_mean  outside_RMS_mean  bank_hinge
           Base             0.250         0.316667      0.922661     0.800097    0.014471          0.005676    0.148015
       Previous             0.500         0.450000      0.919918     0.800367    0.014756          0.005834    0.144902
Feasible_search             0.625         0.500000      0.914881     0.800367    0.015409          0.006188    0.142230
```

Search diagnostics:

```
 factual_index  fixed  accepted_updates  attempts  final_conditional_population_risk
             0  False                 0         0                           0.144694
             1  False                 0         0                           0.144694
             2   True                 0         0                           0.144686
             3  False                 0         0                           0.144694
             4  False                 8         8                           0.143157
             5  False                 0         0                           0.144694
             6  False                23        24                           0.143484
             7   True                 0         0                           0.144709
```

Geometry:

```
 beta_geo         D    B_geo  total_length  bank_hinge
        3 10.592897 31.77869     11.606833    0.141955
```

Images unchanged from previous GRACE, out of eight:

```
method
Base               0
Feasible_search    6
Previous           8
```

This is exploratory reuse of eight generated FFHQ faces and 30 previously observed future heads (18 distinct states), not new independent test evidence or inversion of real GENKI images. Higher average future validity does not establish robustness to all future models. All cases, including failures and the fixed faces, are shown in the PDF. Quantized Base images can lose nominal validity close to the 0.5 boundary.

Reproduction: existing wm_gpu Python, `experiments/046_run_genki_preservation.py --output /home/xu.zhao/robust_ce_results/genki_preservation_search/20260917_feasible_v1`, worker module `genki_preservation_search` through `experiments/run.py`, seed 20260918. Configuration, input hashes, checkpoint paths, selection and search summaries accompany this report. Validation: seven focused tests, 19 completed jobs, unchanged frozen inputs and images, eight-GPU independent final checks, exact geometry and every preservation gate. Existing raw outputs were preserved. Figure: `figures/genki_preservation_search/qualitative.pdf`, four pages covering all eight faces.

Four eligible faces (cases 1, 2, 4, 6) stopped immediately with zero gradient of the current worst-model hinge. Case 5 accepted eight updates before its gradient became zero; case 7 accepted 23 of 24 attempts and remained unsuccessful at the smile threshold. This illustrates the hinge objective and finite search limitations separately from identity rejection.


Additional Base-versus-ours samples: `figures/genki_preservation_search/base_vs_ours_largest_differences.pdf`. Cases 7, 6 and 4 are the top three by full-image RGB RMS difference between saved Base and Feasible_search images, irrespective of success. Identical lower-face crop coordinates are shown without contrast amplification. Ranking and source hashes are saved alongside this report. These largest differences remain visually subtle.
