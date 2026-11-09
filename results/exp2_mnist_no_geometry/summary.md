# MNIST without the geometry constraint

Same saved CNN, near-boundary Base latent codes,64 supports, T3 eta.01, smooth.05, r.05, primal20 and800 updates; no added regularizer. Reused completed constrained output. Only augmented geometry terms and dual update disabled; geometry remains a diagnostic and one-to-one assignment is retained. Latent domain is R64 with identity projection. Seed0; new unconstrained solve on GPU0 after the eight-GPU horizon stage.

| method        |       FV |   proximity |     nominal |   sensitivity |   surrogate |   W2_Base_Q |   W2_Q_P1 |   total_detour |   original_budget | within_original_budget   |   current_validity |   updates |
|:--------------|---------:|------------:|------------:|--------------:|------------:|------------:|----------:|---------------:|------------------:|:-------------------------|-------------------:|----------:|
| Constrained   | 0.998698 |      5.4379 | 3.52927e-05 |    0.00221563 | 0.000146074 |     1.17895 |   58.3253 |        59.5043 |           59.8108 | True                     |                  1 |       800 |
| Unconstrained | 0.998698 |      5.4379 | 3.52927e-05 |    0.00221563 | 0.000146074 |     1.17895 |   58.3253 |        59.5043 |           59.8108 | True                     |                  1 |       800 |

FV uses the same36 T3 scenarios. Proximity is factual-to-decoded-CF pixel L2; geometry is latent Euclidean W2. These are distinct units. Both outputs independently decoded and their smooth objective recomputed. Final supports/assignment preserved; no validity repair, early stopping or convergence claim. Four qualitative factuals match the existing four-style illustration; no output-based reselection.

Raw: `/home/xu.zhao/robust_ce_results/iclr_runs/appendix_mnist/20260925_v1`. Original constrained: `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/20260924_v1/p05001/test/r0.05_beta1.02`.

## Interpretation
The constrained and unconstrained Q, assigned latents and decoded images are byte-identical. The geometry constraint is inactive on this particular r=.05 trajectory: final detour59.50426 is below budget59.81083. Removing it yields no change in FV, cost, or surrogate. This experiment does not demonstrate a robustness–proximity trade-off or establish that geometry is unnecessary in general.

Saved constrained trace confirms all800 dual values are zero; maximum logged geometry violation is -0.30672836.
