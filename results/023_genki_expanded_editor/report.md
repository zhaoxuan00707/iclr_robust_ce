# Expanded-channel and margin experiment

The matched experiment separates two improvements: with the original 0.5 objective, expanding 4 to 16 channels increased average future validity from 50% to 60%; at 16 channels, increasing the optimization target to 0.7 increased it to 75%. Original Base is 31.67% and previous feasible GRACE is 50%. The 16-channel/0.7 arm has six faces valid under every one of the 30 tested future heads; cases 3 and 8 fail under all 30. This is evidence on the previously seen evaluation set, not a robustness guarantee.

| Mutable channels | Target 0.5 future validity | Target 0.7 future validity |
|---|---:|---:|
| 4 | 50.00% | 58.33% |
| 8 | 51.67% | 62.50% |
| 16 | 60.00% | 75.00% |

All selected faces pass the original identity and preservation limits. For the 16-channel/0.7 arm, mean ArcFace is 0.873764 (previous 0.914881), minimum ArcFace 0.800100 and minimum FaceNet 0.800999. Thus passing thresholds does not mean no identity change. Mean full-image RGB RMS distance from Base is 0.015990, versus 0.003060 for previous GRACE, approximately 5.2 times larger.

Visual review: edits are now noticeably larger, including wider mouths and more exposed teeth. However, case 7 also acquires pink cheeks/nose, and a wider open mouth is not automatically a smile. Classifier improvements may partly reflect non-expression appearance changes; human expression labels have not been collected. The largest-difference figure deliberately retains this imperfection. Stronger margins helped the measured objective, but clean semantic editing remains unresolved.


Raw run: `/home/xu.zhao/robust_ce_results/genki_expanded_editor/20260917_capacity_margin_v1`. Fixed bank/original Base parent: `/home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1`. Starting feasible CF parent: `/home/xu.zhao/robust_ce_results/genki_preservation_search/20260917_feasible_v1`.

Matched six-arm design: 4, 8, 16 mutable channels crossed with target 0.5 or 0.7 in max-model mean-population hinge. Classification is always evaluated at 0.5; the 0.7 target is an objective ablation, not the original objective. Every arm starts from the same last feasible four-channel result, uses 48 attempts per face (step 0.05, cap 0.3, growth 1.3, halve on rejection), and permits all eight faces to move. The new four-channel control accounts for the longer search and release of previously fixed faces. Proposals must improve conditional bank risk and pass identity/background and exact geometry gates before acceptance. Greedy coordinate combination uses the global bank objective; no future outcomes select candidates. This is a finite block search, not a global solution or jointly differentiated identity constraint.

The original four channels are retained. Twelve additional channels were selected from only the first 32 existing generated training faces using mean GENKI smile gradient divided by estimated outside-region Jacobian RMS, eligible layers 3–11, sign agreement >=0.75, three random probes per face. Nested 8/16 channel sets are frozen before test preparation. Channel IDs and ranking are recorded. No new model/bank training occurred.

All six arms share exact W2 geometry in the union of 16 standardized style coordinates, radius 2 and beta 3. Original Base/reference images are fixed; inactive coordinates stay factual. This extends the previous 4-D geometry, so direct capacity/margin claims use the new matched controls. Fourier input and every unselected style remain factual. Preservation limits are unchanged: ArcFace>=0.8, FaceNet>=0.8, LPIPS<=0.15, outside RMS<=0.04. The outside mask is an ellipse, not exact background segmentation. Metric thresholds are not proof of unchanged human identity.

Primary saved-image results:

```
  method  current_validity  future_validity  mean_nominal_probability  nominal_target07_count  ArcFace_mean  ArcFace_min  FaceNet_min  LPIPS_mean  outside_RMS_mean  bank_hinge05  bank_hinge07  RGB_RMS_vs_Base
    Base             0.250         0.316667                  0.357676                       0      0.922661     0.800097     0.837818    0.014471          0.005676      0.148015      0.348015         0.000000
Previous             0.625         0.500000                  0.371485                       0      0.914881     0.800367     0.847559    0.015409          0.006188      0.142230      0.334226         0.003060
   c4_t5             0.625         0.500000                  0.379368                       0      0.900229     0.800348     0.847952    0.017635          0.007221      0.134500      0.326497         0.005230
   c8_t5             0.625         0.516667                  0.408587                       0      0.901219     0.800255     0.847727    0.021625          0.008591      0.105621      0.297618         0.006617
  c16_t5             0.750         0.600000                  0.419146                       0      0.901629     0.800566     0.847827    0.018654          0.007606      0.095710      0.287078         0.006243
   c4_t7             0.625         0.583333                  0.460514                       3      0.874216     0.800348     0.803222    0.023685          0.010394      0.133263      0.247339         0.013739
   c8_t7             0.625         0.625000                  0.512099                       4      0.883049     0.800255     0.809512    0.027107          0.011311      0.105712      0.199217         0.014012
  c16_t7             0.750         0.750000                  0.554014                       5      0.873764     0.800100     0.800999    0.024859          0.010658      0.095629      0.153498         0.015990
```

Geometry:

```
 channels  target  beta_geo         D     B_geo  total_length  bank_hinge
        4     0.5         3 11.678896 35.036687     16.561359    0.134103
        8     0.5         3 11.678896 35.036687     17.359986    0.105167
       16     0.5         3 11.678896 35.036687     15.364471    0.095145
        4     0.7         3 11.678896 35.036687     19.711171    0.247063
        8     0.7         3 11.678896 35.036687     18.642966    0.198643
       16     0.7         3 11.678896 35.036687     17.438047    0.153031
```

All predicted probabilities and future metrics are recomputed uniformly on saved PNGs using CPU float32, with image-hash caching. Native LPIPS/outside diagnostics retain their original definitions. Bank hinge at both 0.5 and 0.7 is reported on the same saved images; optimization traces retain native cached risks. Larger pixel differences do not establish stronger smiles, and classifier confidence is not a substitute for human expression labels.

This is exploratory reuse of eight generated FFHQ faces and 30 previously inspected future heads (18 distinct states). It does not invert real GENKI photos or provide fresh independent validation. All failures and arms are shown. `figures/genki_expanded_editor/qualitative.pdf` has eight pages: four per target. `largest_differences.pdf` shows the top three image differences in the predeclared 16-channel/0.7 arm, selected by RGB RMS, not success. These selections are illustrations, not representative statistical evidence.

Reproduction: wm_gpu Python, `experiments/049_run_genki_expanded.py --output /home/xu.zhao/robust_ce_results/genki_expanded_editor/20260917_capacity_margin_v1`, module `genki_expanded_editor` through `experiments/run.py`; seed 20260918. Configuration and input/model hashes accompany this report. Validation: seven relevant unit tests, 114 completed jobs, frozen controls/inputs/images, exact common geometry, all accepted-step and final preservation gates, eight GPU IDs for each final arm screen. Existing raw outputs remain unchanged.
