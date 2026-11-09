# Two-seed full-baseline replication

Raw source: `/home/xu.zhao/robust_ce_results/iclr_runs/tabular_replication/20261109_full_v1`. Completed workers: 16/16. Saved-output checks: 160.

Mean and sample SD use training/split seeds as units (two seeds); future-model scores are nested and include repeated scenarios. FV/CV fractions in CSV. Mixed relaxed and hard results are distinct. Failed generations remain in denominators. Seed0 is preserved and not pooled because these adapters differ.

| Dataset | Classifier | Method | Realization | Seeds | FV (%) | Proximity | Native failures |
|---|---|---|---|---:|---:|---:|---:|
| diabetes | logistic | dice | native | 2 | 98.38 ± 0.53 | 2.2462 ± 0.3317 | 0 |
| diabetes | logistic | ellice | native | 2 | 100.00 ± 0.00 | 2.0479 ± 0.2637 | 0 |
| diabetes | logistic | grace_dice | native | 2 | 100.00 ± 0.00 | 2.3394 ± 0.3144 | 0 |
| diabetes | logistic | grace_wachter | native | 2 | 100.00 ± 0.00 | 1.3083 ± 0.1578 | 0 |
| diabetes | logistic | proplace | native | 2 | 100.00 ± 0.00 | 1.7209 ± 0.1401 | 0 |
| diabetes | logistic | roar | native | 2 | 80.19 ± 9.24 | 1.8483 ± 0.1246 | 14 |
| diabetes | logistic | trex | native | 2 | 100.00 ± 0.00 | 2.1579 ± 0.3667 | 0 |
| diabetes | logistic | wachter | native | 2 | 89.72 ± 4.78 | 0.9218 ± 0.1466 | 0 |
| diabetes | mlp | dice | native | 2 | 95.87 ± 1.09 | 2.3711 ± 0.4183 | 0 |
| diabetes | mlp | ellice | native | 2 | 100.00 ± 0.00 | 2.3126 ± 0.3319 | 0 |
| diabetes | mlp | grace_dice | native | 2 | 100.00 ± 0.00 | 2.5767 ± 0.3255 | 0 |
| diabetes | mlp | grace_wachter | native | 2 | 100.00 ± 0.00 | 1.5981 ± 0.0958 | 0 |
| diabetes | mlp | proplace | native | 2 | 100.00 ± 0.00 | 2.0644 ± 0.2679 | 0 |
| diabetes | mlp | roar | native | 2 | 81.04 ± 6.55 | 1.9539 ± 0.1920 | 15 |
| diabetes | mlp | trex | native | 2 | 100.00 ± 0.00 | 2.2084 ± 0.2519 | 0 |
| diabetes | mlp | wachter | native | 2 | 80.61 ± 4.09 | 0.9739 ± 0.0973 | 0 |
| fico | logistic | dice | native | 2 | 96.62 ± 2.49 | 5.6020 ± 1.4921 | 0 |
| fico | logistic | ellice | native | 2 | 100.00 ± 0.00 | 6.8386 ± 1.0990 | 0 |
| fico | logistic | grace_dice | hard | 2 | 99.49 ± 0.58 | 5.6850 ± 1.4725 | 0 |
| fico | logistic | grace_dice | relaxed | 2 | 99.51 ± 0.59 | 5.6854 ± 1.4728 | 0 |
| fico | logistic | grace_wachter | hard | 2 | 99.25 ± 0.71 | 2.2900 ± 0.2094 | 0 |
| fico | logistic | grace_wachter | relaxed | 2 | 99.33 ± 0.68 | 2.2922 ± 0.2111 | 0 |
| fico | logistic | proplace | hard | 2 | 98.85 ± 0.18 | 4.0512 ± 0.3325 | 1 |
| fico | logistic | proplace | relaxed | 2 | 98.96 ± 0.15 | 4.0321 ± 0.3831 | 1 |
| fico | logistic | roar | hard | 2 | 45.63 ± 4.24 | 3.7092 ± 0.2436 | 153 |
| fico | logistic | roar | relaxed | 2 | 40.17 ± 1.62 | 4.0159 ± 0.2599 | 153 |
| fico | logistic | trex | native | 2 | 99.79 ± 0.27 | 4.8350 ± 0.4339 | 0 |
| fico | logistic | wachter | native | 2 | 85.36 ± 1.31 | 2.0172 ± 0.2171 | 0 |
| fico | mlp | dice | native | 2 | 96.74 ± 2.57 | 5.2957 ± 0.5069 | 0 |
| fico | mlp | ellice | native | 2 | 100.00 ± 0.00 | 5.3169 ± 0.6880 | 0 |
| fico | mlp | grace_dice | hard | 2 | 99.97 ± 0.05 | 5.3642 ± 0.4951 | 0 |
| fico | mlp | grace_dice | relaxed | 2 | 99.97 ± 0.05 | 5.3652 ± 0.4957 | 0 |
| fico | mlp | grace_wachter | hard | 2 | 99.88 ± 0.15 | 1.4831 ± 0.0077 | 0 |
| fico | mlp | grace_wachter | relaxed | 2 | 99.90 ± 0.15 | 1.4908 ± 0.0058 | 0 |
| fico | mlp | proplace | hard | 2 | 100.00 ± 0.00 | 6.1928 ± 1.4845 | 0 |
| fico | mlp | proplace | relaxed | 2 | 100.00 ± 0.00 | 6.1442 ± 1.4998 | 0 |
| fico | mlp | roar | hard | 2 | 78.96 ± 12.69 | 3.4648 ± 0.2240 | 59 |
| fico | mlp | roar | relaxed | 2 | 79.37 ± 7.48 | 3.7088 ± 0.1619 | 59 |
| fico | mlp | trex | native | 2 | 94.53 ± 5.52 | 5.2100 ± 0.7816 | 14 |
| fico | mlp | wachter | native | 2 | 87.78 ± 7.07 | 1.1022 ± 0.0037 | 0 |
| german | logistic | dice | native | 2 | 87.23 ± 2.34 | 3.4234 ± 0.0375 | 0 |
| german | logistic | ellice | native | 2 | 100.00 ± 0.00 | 9.5701 ± 0.2976 | 0 |
| german | logistic | grace_dice | hard | 2 | 95.90 ± 0.77 | 3.5145 ± 0.0248 | 0 |
| german | logistic | grace_dice | relaxed | 2 | 99.43 ± 0.27 | 3.5469 ± 0.0144 | 0 |
| german | logistic | grace_wachter | hard | 2 | 88.86 ± 0.99 | 2.1973 ± 0.0339 | 0 |
| german | logistic | grace_wachter | relaxed | 2 | 99.34 ± 0.01 | 2.3472 ± 0.0179 | 0 |
| german | logistic | proplace | hard | 2 | 95.80 ± 0.62 | 6.8994 ± 0.2046 | 2 |
| german | logistic | proplace | relaxed | 2 | 98.35 ± 1.76 | 6.4609 ± 0.4607 | 2 |
| german | logistic | roar | hard | 2 | 86.35 ± 9.21 | 4.9324 ± 0.5884 | 0 |
| german | logistic | roar | relaxed | 2 | 100.00 ± 0.00 | 5.5302 ± 0.1049 | 0 |
| german | logistic | trex | native | 2 | 99.74 ± 0.32 | 7.1285 ± 0.0386 | 0 |
| german | logistic | wachter | native | 2 | 58.60 ± 0.14 | 1.8847 ± 0.0672 | 0 |
| german | mlp | dice | native | 2 | 85.88 ± 3.73 | 3.8945 ± 0.3038 | 0 |
| german | mlp | ellice | native | 2 | 99.95 ± 0.07 | 9.5007 ± 0.3936 | 0 |
| german | mlp | grace_dice | hard | 2 | 96.70 ± 2.27 | 4.0197 ± 0.2546 | 0 |
| german | mlp | grace_dice | relaxed | 2 | 98.84 ± 1.64 | 4.0600 ± 0.2402 | 0 |
| german | mlp | grace_wachter | hard | 2 | 93.62 ± 1.00 | 2.2320 ± 0.1525 | 0 |
| german | mlp | grace_wachter | relaxed | 2 | 98.89 ± 1.57 | 2.3776 ± 0.1365 | 0 |
| german | mlp | proplace | hard | 2 | 74.06 ± 19.63 | 5.4780 ± 1.7088 | 26 |
| german | mlp | proplace | relaxed | 2 | 74.06 ± 19.63 | 5.1643 ± 1.6678 | 26 |
| german | mlp | roar | hard | 2 | 85.20 ± 12.98 | 4.5904 ± 0.7732 | 6 |
| german | mlp | roar | relaxed | 2 | 93.31 ± 3.70 | 5.3499 ± 0.1332 | 6 |
| german | mlp | trex | native | 2 | 100.00 ± 0.00 | 7.4185 ± 0.2731 | 0 |
| german | mlp | wachter | native | 2 | 50.44 ± 4.75 | 1.7557 ± 0.1740 | 0 |
| heloc | logistic | dice | native | 2 | 97.89 ± 0.28 | 4.9032 ± 0.1012 | 0 |
| heloc | logistic | ellice | native | 2 | 100.00 ± 0.00 | 3.6819 ± 0.0931 | 0 |
| heloc | logistic | grace_dice | native | 2 | 99.92 ± 0.11 | 4.9845 ± 0.0865 | 0 |
| heloc | logistic | grace_wachter | native | 2 | 99.92 ± 0.09 | 1.9475 ± 0.0562 | 0 |
| heloc | logistic | proplace | native | 2 | 99.81 ± 0.10 | 3.1461 ± 0.1286 | 0 |
| heloc | logistic | roar | native | 2 | 45.51 ± 1.09 | 2.9126 ± 0.0789 | 140 |
| heloc | logistic | trex | native | 2 | 100.00 ± 0.00 | 3.9264 ± 0.1195 | 0 |
| heloc | logistic | wachter | native | 2 | 91.81 ± 1.60 | 1.6934 ± 0.0557 | 0 |
| heloc | mlp | dice | native | 2 | 95.25 ± 0.55 | 5.7385 ± 0.1191 | 0 |
| heloc | mlp | ellice | native | 2 | 100.00 ± 0.00 | 2.7929 ± 0.5286 | 0 |
| heloc | mlp | grace_dice | native | 2 | 99.96 ± 0.01 | 5.7924 ± 0.0978 | 0 |
| heloc | mlp | grace_wachter | native | 2 | 99.88 ± 0.12 | 1.4004 ± 0.2406 | 0 |
| heloc | mlp | proplace | native | 2 | 100.00 ± 0.00 | 4.7292 ± 0.4432 | 0 |
| heloc | mlp | roar | native | 2 | 75.97 ± 8.18 | 2.8920 ± 0.1701 | 62 |
| heloc | mlp | trex | native | 2 | 99.61 ± 0.55 | 3.8871 ± 0.2440 | 1 |
| heloc | mlp | wachter | native | 2 | 73.12 ± 8.91 | 1.0960 ± 0.1659 | 0 |
