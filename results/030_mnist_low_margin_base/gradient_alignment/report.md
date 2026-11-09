# Nominal/sensitivity gradient alignment at identical Q

Directly compute cos(grad_Q F_Q(theta_T_nom), grad_Q ||s_Q||_*) from the existing smooth update-aware objective. Both gradients are evaluated at the identical saved latent population Q and flattened across all64x64 coordinates. Sensitivity includes differentiation through the full3-step update and the data-gradient norm. No geometric penalty or r multiplier is included in the cosine. This is not pixel-change cosine and not a cosine between gradients at different Q.

Test-population results:
```
point          Base        r0      r025
condition                              
original   0.985590  0.987986  0.986648
p052       0.993971  0.988183  0.985844
p0505      0.994383  0.988358  0.985502
p05001     0.994392  0.988371  0.985868
```

Near-boundary Base: ||grad F||=.0350302, ||grad sensitivity||=1.804276; cosine=.994392. At r=.25, the weighted sensitivity gradient has12.8768 times the nominal gradient norm. Across the16 audited points, alignment is high but not exact. This supports locally similar nominal and sensitivity descent directions and is consistent with the visually similar difference maps. It does not prove redundancy/equivalent objectives, global alignment along every iterate, or that the sensitivity term is numerically negligible. Constraints may change the actual update direction. The low-margin Base case gives a weighted directional amplification rather than an entirely new direction; different finite-iteration trajectories can still lead to different robustness.

Both final r0 and final r.25 are separate probe locations; at each one, both objective terms are reevaluated together. Saved per-term gradients are64x64 tensors. Gradient decomposition grad(Phi_.25)=grad(F)+.25*grad(sensitivity) passed torch.testing.assert_close at all16 points; max coordinate residual 1.12e-08. Independent float64 dot/norm recomputation agrees to1e-12 and all Q input hashes are unchanged. CSV includes norms, dot products, weighted norm ratios, point/checkpoint/config hashes and GPU IDs. All8 GPUs used in two waves; no optimization/model training was rerun.

Source: experiments/067_mnist_gradient_alignment.py. Command:
```
/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/067_mnist_gradient_alignment.py --run /home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/20260924_v1
```
Existing summary is protected against overwrite.
