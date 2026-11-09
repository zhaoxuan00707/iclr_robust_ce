# Task100 ten-CF runtime

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1`. One timing per dataset/method; seed0.

All methods use the first ten frozen unfavorable test factuals. GRACE uses a fresh M=n=10 solve and independently generated ten-CF Wachter Base. Refinement and Base-inclusive total are separated. CUDA synchronization brackets refinement; shared fitting/calibration/loading are excluded. Baselines here execute on CPU with one thread; GRACE uses its assigned GPU. Dataset workers overlap on separate GPUs and share host resources, so these are observed wall times, not isolated latency or statistically significant speedups. Failures and unsupported configurations are retained; ten attempts do not imply ten valid CFs.

| dataset   | method        | device   |    seconds |   total_seconds |         FV |   valid_cf_count |   native_failures | status                              |
|:----------|:--------------|:---------|-----------:|----------------:|-----------:|-----------------:|------------------:|:------------------------------------|
| diabetes  | trex          | cpu      |  10.1544   |       10.1544   | 1          |               10 |                 0 | executed                            |
| diabetes  | roar          | cpu      |   1.66142  |        1.66142  | 0.7        |                7 |                 0 | executed                            |
| diabetes  | proplace      | cpu      |  25.3712   |       25.3712   | 0.00666667 |                0 |                10 | infrastructure_failure              |
| diabetes  | ellice        | cpu      |   0.206585 |        0.206585 | 1          |               10 |                 0 | executed                            |
| diabetes  | grace_wachter | cuda     |  22.7797   |       23.1945   | 1          |               10 |                 0 | executed                            |
| fico      | trex          | cpu      |  88.422    |       88.422    | 1          |               10 |                 0 | executed                            |
| fico      | roar          | cpu      | nan        |      nan        | 0.0133333  |                0 |                10 | unsupported_categorical_constraints |
| fico      | proplace      | cpu      | nan        |      nan        | 0.0133333  |                0 |                10 | unsupported_categorical_constraints |
| fico      | ellice        | cpu      |   0.32997  |        0.32997  | 1          |               10 |                 0 | executed                            |
| fico      | grace_wachter | cuda     |  18.9776   |       19.701    | 1          |               10 |                 0 | executed                            |
| german    | trex          | cpu      |  42.0379   |       42.0379   | 1          |               10 |                 0 | executed                            |
| german    | roar          | cpu      | nan        |      nan        | 0.00444444 |                0 |                10 | unsupported_categorical_constraints |
| german    | proplace      | cpu      | nan        |      nan        | 0.00444444 |                0 |                10 | unsupported_categorical_constraints |
| german    | ellice        | cpu      |   0.391244 |        0.391244 | 1          |               10 |                 0 | executed                            |
| german    | grace_wachter | cuda     |  58.5691   |       60.516    | 1          |               10 |                 0 | executed                            |
| heloc     | trex          | cpu      |  90.297    |       90.297    | 1          |               10 |                 0 | executed                            |
| heloc     | roar          | cpu      |   2.17302  |        2.17302  | 1          |               10 |                 0 | executed                            |
| heloc     | proplace      | cpu      | 300.01     |      300.01     | 0.0422222  |                0 |                10 | executed                            |
| heloc     | ellice        | cpu      |   0.191366 |        0.191366 | 1          |               10 |                 0 | executed                            |
| heloc     | grace_wachter | cuda     |  81.6388   |       82.0504   | 1          |               10 |                 0 | executed                            |
