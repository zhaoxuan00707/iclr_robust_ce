# Larger-beta GENKI GRACE extension

Increasing beta to 1.5, 2 and 3 returned exactly the same eight PNGs as the previous beta=1.2 result. Future validity remained 45%, versus 31.67% for the unchanged Base. Of 106 candidates screened, only the previous winner passed preservation for all eight faces. The other 105 candidates each failed ArcFace for cases 3 and 8 (zero-based indices 2 and 7); FaceNet, LPIPS and outside-region limits had no failures. Thus ArcFace was the observed rejection reason in this finite search; merely expanding the geometry budget did not change the result.

Raw run: `/home/xu.zhao/robust_ce_results/genki_grace_beta_extension/20260917_larger_beta_v1`. Parent: `/home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1`.

User requested larger beta instead of stronger nominal Base. New beta values: 1.5, 2, 3; radius remains 2. The original frozen beta=1.2 winner and Base are shown unchanged. New candidates are pooled with prior candidates and screened against unchanged per-face ArcFace>=0.8, FaceNet>=0.8, LPIPS<=0.15 and outside-region RMS<=0.04. Selection uses cached finite-bank population hinge, then pixel cost; future outcomes do not select candidates or a winning beta.

```
           method  current_validity  future_validity  ArcFace_mean  ArcFace_min  LPIPS_mean  outside_RMS_mean  bank_hinge
             Base              0.25         0.316667      0.922661     0.800097    0.014471          0.005676    0.148015
Previous_beta1.20              0.50         0.450000      0.919918     0.800367    0.014756          0.005834    0.144902
   GRACE_beta1.50              0.50         0.450000      0.919918     0.800367    0.014756          0.005834    0.144902
   GRACE_beta2.00              0.50         0.450000      0.919918     0.800367    0.014756          0.005834    0.144902
   GRACE_beta3.00              0.50         0.450000      0.919918     0.800367    0.014756          0.005834    0.144902
```

Geometry (the 1.2 row is the new pooled solution; the figure's previous 1.2 column remains the original frozen solution):

```
 beta_geo  candidate         D     B_geo  total_length     slack  Phi_hat_r_cached  pixel_cost                                                coordinate_sha256  same_coordinates_as_previous_beta1_2
      1.2          0 10.592897 12.711476     10.910279  1.801197          0.144694    6.512011 2c54628d597f285ce4441c81edc4c9de0ad669236e887ce458d9a901d343bc6a                                  True
      1.5          0 10.592897 15.889345     10.910279  4.979066          0.144694    6.512011 2c54628d597f285ce4441c81edc4c9de0ad669236e887ce458d9a901d343bc6a                                  True
      2.0          0 10.592897 21.185794     10.910279 10.275515          0.144694    6.512011 2c54628d597f285ce4441c81edc4c9de0ad669236e887ce458d9a901d343bc6a                                  True
      3.0          0 10.592897 31.778690     10.910279 20.868411          0.144694    6.512011 2c54628d597f285ce4441c81edc4c9de0ad669236e887ce458d9a901d343bc6a                                  True
```

Candidate counts: {"D": 10.592896759059967, "unique_evaluated_candidates": 171, "screened_candidates": 106, "previous_incumbent_risk": 0.14469365403056145, "previous_incumbent_candidate": 0}. Candidates passing preservation for all eight faces: 1. Failed candidate/face checks (categories can overlap): {"ArcFace": 210, "FaceNet": 0, "LPIPS": 0, "outside_RMS": 0}.

Images identical to the original beta=1.2, out of eight per method:

```
method
Base                 0
GRACE_beta1.50       8
GRACE_beta2.00       8
GRACE_beta3.00       8
Previous_beta1.20    8
```

Increasing beta expands the admissible geometric path length; it does not require using the extra budget. These results concern the unchanged finite 12-step search and fixed four-channel editor, not a global optimum or proof that further changes are impossible. Preservation can restrict candidate acceptance independently of beta. All failures remain visible. Primary scores are recomputed uniformly on the exact saved PNGs using CPU float32, with image-hash caching. Cached optimization risk is separately reported in geometry.csv. Identity and native-image LPIPS/outside diagnostics retain the original measurement definitions; the outside region is a fixed ellipse mask, not segmentation or exact background preservation.

This is exploratory reuse of eight generated FFHQ test faces and the previously observed 30 future heads (18 distinct states), not new independent test evidence or real GENKI photo inversion. Original frozen Base can lose nominal validity after image quantization near its 0.5 boundary.

Reproduction: `experiments/044_run_genki_beta_extension.py --output /home/xu.zhao/robust_ce_results/genki_grace_beta_extension/20260917_larger_beta_v1` via the existing wm_gpu Python environment; worker launcher `experiments/run.py run genki_grace_beta_extension`; config copied here as resolved_config.json; seed 20260918. Frozen checkpoint paths and hashes are in input_manifest.json. Existing outputs were preserved. Validation: seven focused tests, 15 completed new jobs, eight GPUs for independent case screening, frozen input/image hashes, all selected geometry and preservation gates. Figure: `figures/genki_grace_beta_extension/qualitative.pdf` (four pages, all eight cases).
