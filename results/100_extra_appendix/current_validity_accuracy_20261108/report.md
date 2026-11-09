# Current validity and classifier accuracy

Linear classifier means logistic regression. Current CF validity is evaluated on the fixed test CF cohort using the original deployed classifier; classifier accuracy uses labeled factual data.

Tabular test/calibration accuracy was recomputed from frozen checkpoints and saved complete data splits, with no training or raw writes. All eight calibration accuracies match saved metadata within1e-6. Absolute paths and checkpoint hashes are in provenance.json.

| worker            |   cal_correct |   cal_n |   cal_accuracy |   test_correct |   test_n |   test_accuracy | dataset   | classifier   |
|:------------------|--------------:|--------:|---------------:|---------------:|---------:|----------------:|:----------|:-------------|
| diabetes_logistic |           118 |     154 |       0.766234 |            110 |      154 |        0.714286 | diabetes  | logistic     |
| diabetes_mlp      |           126 |     154 |       0.818182 |            111 |      154 |        0.720779 | diabetes  | mlp          |
| fico_logistic     |          1500 |    2092 |       0.717017 |           1506 |     2092 |        0.719885 | fico      | logistic     |
| fico_mlp          |          1491 |    2092 |       0.712715 |           1479 |     2092 |        0.706979 | fico      | mlp          |
| german_logistic   |           150 |     200 |       0.75     |            143 |      200 |        0.715    | german    | logistic     |
| german_mlp        |           148 |     200 |       0.74     |            147 |      200 |        0.735    | german    | mlp          |
| heloc_logistic    |          1415 |    2000 |       0.7075   |           1449 |     2000 |        0.7245   | heloc     | logistic     |
| heloc_mlp         |          1421 |    2000 |       0.7105   |           1486 |     2000 |        0.743    | heloc     | mlp          |

Image current validity:

| task                                      | method         |   current_validity | source                                                         |
|:------------------------------------------|:---------------|-------------------:|:---------------------------------------------------------------|
| Galaxy: Merging to Round                  | Base           |              0.75  | results/031_galaxy_update_aware/merging_round_test_summary.csv |
| Galaxy: Merging to Round                  | r0_beta1.02    |              1     | results/031_galaxy_update_aware/merging_round_test_summary.csv |
| Galaxy: Round to Cigar                    | Base           |              0.5   | results/031_galaxy_update_aware/round_cigar_test_summary.csv   |
| Galaxy: Round to Cigar                    | r0.25_beta1.02 |              1     | results/031_galaxy_update_aware/round_cigar_test_summary.csv   |
| FFHQ                                      | Base           |              0.875 | results/032_ffhq_update_aware/test_summary.csv                 |
| FFHQ                                      | r0_beta1.02    |              1     | results/032_ffhq_update_aware/test_summary.csv                 |
| MNIST 4 to 9 (task016 near-boundary Base) | Base           |              1     | results/030_mnist_low_margin_base/p05001/summary.csv           |
| MNIST 4 to 9 (task016 near-boundary Base) | r0.25_beta1.02 |              1     | results/030_mnist_low_margin_base/p05001/summary.csv           |

Image classifier accuracy: MNIST strong CNN99.6986% on official digit4/9 test examples; weak diagnostic CNN88.6991%, separate model. Galaxy VGG67.1875% (43/64) on a historical calibration-pool audit, not full held-out test accuracy. Its historical model selection used the original test folder. FFHQ original CLIP has no ground-truth smile accuracy evaluation; current CF validity must not be called accuracy.

All tabular CF rows, including unsupported and failed outcomes, are retained in current_validity.csv. The mixed GRACE hard FICO/logistic row is100% currently valid but geometry-infeasible. German/logistic ElliCE100% current validity is not native robust feasibility.
