# PROPLACE German Credit MLP follow-up

Frozen radius .005; only previously failed/skipped cases retried. All successful original CFs unchanged. Original runs preserved. Statistics use two training seeds, sample SD. Relaxed results do not assert hard-category feasibility. Native robustness independently checked per factual in native_checks.csv.

|   seed | realization   |       FV |   proximity |   current_validity |   failures |   native_robust_fraction |
|-------:|:--------------|---------:|------------:|-------------------:|-----------:|-------------------------:|
|      1 | relaxed       | 100      |     6.86748 |           100      |          0 |                        1 |
|      1 | hard          |  98.0556 |     7.348   |            97.9167 |          0 |                      nan |
|      2 | relaxed       | 100      |     7.39027 |           100      |          0 |                        1 |
|      2 | hard          | 100      |     7.81964 |           100      |          0 |                      nan |

| realization   |   FV_mean |   FV_std |   proximity_mean |   proximity_std |
|:--------------|----------:|---------:|-----------------:|----------------:|
| hard          |   99.0278 |  1.37493 |          7.58382 |        0.333495 |
| relaxed       |  100      |  0       |          7.12888 |        0.36967  |
