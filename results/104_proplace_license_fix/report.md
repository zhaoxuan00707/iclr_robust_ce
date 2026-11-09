# PROPLACE German Credit MLP follow-up

Frozen radius .005; only previously failed/skipped cases retried. All successful original CFs unchanged. Original runs preserved. Statistics use two training seeds, sample SD. Relaxed results do not assert hard-category feasibility. Native robustness independently checked per factual in native_checks.csv.

|   seed | realization   |       FV |   proximity |   current_validity |   failures |   native_robust_fraction |
|-------:|:--------------|---------:|------------:|-------------------:|-----------:|-------------------------:|
|      1 | relaxed       | 100      |     6.86748 |           100      |          0 |                 1        |
|      1 | hard          |  98.0556 |     7.348   |            97.9167 |          0 |               nan        |
|      2 | relaxed       |  97.8723 |     7.26351 |            97.8723 |          1 |                 0.978723 |
|      2 | hard          |  97.8723 |     7.65063 |            97.8723 |          1 |               nan        |

| realization   |   FV_mean |   FV_std |   proximity_mean |   proximity_std |
|:--------------|----------:|---------:|-----------------:|----------------:|
| hard          |   97.9639 | 0.129551 |          7.49932 |        0.21399  |
| relaxed       |   98.9362 | 1.50448  |          7.06549 |        0.280032 |
