# HELOC same-Q gradient alignment audit

The nominal and sensitivity support gradients are positively aligned in HELOC, but less nearly parallel than in MNIST. HELOC test cosines range0.876746–0.895057 (angles26.48–28.75 degrees), compared with approximately.986–.994 in the MNIST task016 test audit. This is local evidence at saved points, not a whole-trajectory or general dataset claim.

At each Q, compute cos(grad_Q F_Q(theta_T_nom),grad_Q ||s_Q||_*), flattening all128x22 normalized feature coordinates. Both terms use the same saved population, current MLP, full6000 training rows and complete10-step SGD lr.01 update, smooth probability hinge t=.05 and empirical-RMS sqrt(6000) dual scaling. No geometry/ALM term or radius multiplier enters the cosine. Positive scaling by r would not change the cosine.

```
split        point   cosine  angle_degrees  nominal_gradient_norm  sensitivity_gradient_norm  weighted_ratio_r025_bref  weighted_ratio_selected_r
  cal         Base 0.917364      23.456250               0.014476                   0.018208                  0.858314                  10.299769
  cal    r/b_ref=0 0.893971      26.623422               0.002306                   0.003972                  1.175592                  14.107099
  cal r/b_ref=0.25 0.904683      25.219332               0.001681                   0.002362                  0.958825                  11.505894
  cal    r/b_ref=3 0.889628      27.173512               0.000663                   0.000791                  0.814288                   9.771455
 test         Base 0.876746      28.747737               0.014417                   0.020507                  0.970683                  11.648201
 test    r/b_ref=0 0.881296      28.200933               0.002267                   0.004087                  1.230020                  14.760243
 test r/b_ref=0.25 0.895057      26.484260               0.002038                   0.002965                  0.992506                  11.910074
 test    r/b_ref=3 0.877405      28.669089               0.001008                   0.001145                  0.775264                   9.303174
```

All saved optimized Q use beta1.1, the selected geometry budget. Probe Base, nominal-only Q, r/b_ref=.25 Q and selected r/b_ref=3 Q on both calibration/test populations. HELOC uses b_ref=2.729610610, so ratio.25 means absolute r=.682402653 and selected ratio3 means r=8.188831830. These must not be confused with absolute MNIST r=.25. Nominal-only HELOC beta1.1 solves were feasible but iteration-limited; these are saved iterates, not established optima.

At test Base, cosine=.876746 and the weighted sensitivity gradient norm is.970683 times nominal at r/b_ref=.25, or11.648201 times nominal at selected r/b_ref=3. A cosine around.88 has an orthogonal component of about48% of the sensitivity gradient norm, so it should not be described as identical or merely an exact rescaling of the nominal term. At high selected r the sensitivity contribution dominates in magnitude, even though measured FV is already almost saturated under nominal refinement. Gradient alignment does not isolate geometry's contribution or prove sensitivity is redundant; the constrained update can point elsewhere.

All8 full-gradient decompositions grad(Phi_r)=grad(F)+r*grad(sensitivity) pass (maximum absolute coordinate residual7.45e-09). Independent float64 dot/norm recomputation agrees to1e-12; saved Q hashes unchanged. All8 GPUs were used, with4% allocator caps. No model retraining or CF optimization rerun. No saved experiment outputs modified. Gradients, norms, hashes and weighted ratios are saved alongside summary.csv.

Reproduction (existing summary protected):
```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/068_heloc_gradient_alignment.py --run /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/20260924_v1
```
