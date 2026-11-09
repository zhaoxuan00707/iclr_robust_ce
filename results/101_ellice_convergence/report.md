# ElliCE calibration-only convergence screen

Raw: `/home/xu.zhao/robust_ce_results/iclr_runs/ellice_convergence/20260925_v1`. Frozen German/logistic seed0 model and all calibration factuals. Epsilon and ellipsoid regularization unchanged across optimizer comparisons; no test scoring. Success means positive independently recomputed worst-case ellipsoid logit, not merely a nonempty returned frame.

|   epsilon | config       |   n |   current_valid |   ellipsoid_robust |   mean_robust_logit |    cost |   seconds |   warning_count |
|----------:|:-------------|----:|----------------:|-------------------:|--------------------:|--------:|----------:|----------------:|
|     0.005 | default      |  62 |               1 |                  1 |            0.95787  | 12.3918 |   2.60941 |               0 |
|     0.005 | longer       |  62 |               1 |                  1 |            0.95787  | 12.3918 |   1.49769 |               0 |
|     0.005 | smaller_step |  62 |               1 |                  1 |            0.742311 |  9.3847 |   4.59702 |               0 |
|     0.01  | default      |  62 |               1 |                  1 |            0.474183 | 10.8054 |   1.63344 |               0 |
|     0.01  | longer       |  62 |               1 |                  1 |            0.474183 | 10.8054 |   1.66307 |               0 |
|     0.01  | smaller_step |  62 |               1 |                  1 |            0.401944 |  8.5946 |   4.59247 |               0 |
|     0.05  | default      |  62 |               1 |                  0 |           -0.307481 | 10.5316 |  27.9999  |              62 |
|     0.05  | longer       |  62 |               1 |                  0 |           -0.257751 | 10.4572 | 140.215   |              62 |
|     0.05  | smaller_step |  62 |               1 |                  0 |           -0.264918 | 10.4689 | 175.792   |              62 |
