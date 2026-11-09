# MNIST update-horizon mismatch

Fixed task016 near-boundary p05001 population, seed0, saved CNN/64D decoder,64 factuals, T_design3, r.05, beta1.02. No recalibration. One trajectory per each of36 old held-out scenarios; checkpoints3/5/10/20/40, all eight GPUs. Scenario-derived seeds81000/81001/81002 are inherited realizations, not independent experiment-seed runs. Original inputs unchanged. Raw `/home/xu.zhao/robust_ce_results/iclr_runs/appendix_mnist/20260925_v1`.

|   T_eval | method   |       FV |   FV_scenario_std |   train_loss |   parameter_displacement |   proximity |   scenarios |
|---------:|:---------|---------:|------------------:|-------------:|-------------------------:|------------:|------------:|
|        3 | Base     | 0.773872 |        0.412664   |   0.0517253  |                0.0359921 |     5.35841 |          36 |
|        3 | GRACE    | 0.998698 |        0.00575616 |   0.0517253  |                0.0359921 |     5.4379  |          36 |
|        5 | Base     | 0.769531 |        0.411833   |   0.0409728  |                0.0432808 |     5.35841 |          36 |
|        5 | GRACE    | 0.998264 |        0.00725971 |   0.0409728  |                0.0432808 |     5.4379  |          36 |
|       10 | Base     | 0.757378 |        0.411157   |   0.0279478  |                0.0573309 |     5.35841 |          36 |
|       10 | GRACE    | 0.997396 |        0.00952265 |   0.0279478  |                0.0573309 |     5.4379  |          36 |
|       20 | Base     | 0.755642 |        0.385984   |   0.0174847  |                0.0772796 |     5.35841 |          36 |
|       20 | GRACE    | 0.997396 |        0.00875956 |   0.0174847  |                0.0772796 |     5.4379  |          36 |
|       40 | Base     | 0.746094 |        0.357348   |   0.00992696 |                0.104301  |     5.35841 |          36 |
|       40 | GRACE    | 0.996094 |        0.00943064 |   0.00992696 |                0.104301  |     5.4379  |          36 |

Proximity is mean factual-to-CF pixel L2 and remains fixed. Future training loss is post-update BCE on the scenario training images. Parameter displacement includes all trainable CNN parameters, not buffers. Scenario std is descriptive across shifts, not a multi-seed standard error. Checkpoints atT3 replay prior states exactly;180 checkpoints independently reconstructed using independent explicit autograd/SGD arithmetic and scores verified.

GRACE FV at the five horizons: T3: 99.87%, T5: 99.83%, T10: 99.74%, T20: 99.74%, T40: 99.61%. Largest adjacent change: 0.13 percentage points. Do not infer arbitrary-retraining robustness from this short-update mismatch diagnostic. Saved CNN retained per user instruction despite PDF MLP wording.

## Interpretation
Degradation is gradual and small here: GRACE FV99.8698% atT3 to99.6094% atT40 (0.2604 percentage-point decrease), with no sharp collapse. Base falls77.3872%→74.6094%. GRACE proximity remains5.4379. This is one saved CNN and36 scenarios, including repeated deterministic shifts, not a universal horizon-mismatch guarantee. An initial independent audit using torch.optim.SGD differed numerically because its fused-alpha arithmetic differs from unroll; the verified audit uses explicit out-of-place SGD arithmetic and passes all180 checkpoints.
