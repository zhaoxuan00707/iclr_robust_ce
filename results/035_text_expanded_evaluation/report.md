# Expanded frozen-settings ParaDetox evaluation

64 new factuals, eight populations of eight. Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/text_expanded_evaluation/20260925_v1`.
Same trained models, nominal128 examples, favorable references and selected r0/beta1.5 as task020; diagnostic r.25/beta1.5 fixed in advance. No retuning or new training.

| method   |       FV |   current_validity |    cost |   cosine |   word_edit |   nominal |   sensitivity |
|:---------|---------:|-------------------:|--------:|---------:|------------:|----------:|--------------:|
| Base     | 78.90625 |           76.56250 | 0.43031 |  0.89290 |     0.25830 |   0.04703 |       1.83177 |
| Selected | 92.44792 |           90.62500 | 0.67757 |  0.70584 |     0.52914 |   0.01460 |       0.62943 |
| r0.25    | 92.44792 |           90.62500 | 0.67757 |  0.70584 |     0.52914 |   0.01460 |       0.62943 |

FV/current validity are percentages. Cost is normalized MiniLM embedding L2, cosine is an automatic similarity proxy, word edit is normalized word Levenshtein. Nominal/sensitivity are averaged over eight population objectives, not one64-support objective.
Selected changes 44/64 strings versus Base; r.25 changes 0/64 versus selected r0. The report includes all64 cases without filtering failed Base generation or bad rewrites. Base failed for 15/64 cases. No semantic gate or validity repair was added.

## Scope and checks
All texts in all task019/020 partitions excluded from new selection. Seed2101; first64 classifier-negative5–25-word factuals in shuffled remaining corpus, eight consecutive blocks. Test human rewrites withheld until reporting.8 beam+64 sampled candidates, exact same Base policy. T30 full-batch eta.5, smooth temperature.05, ten coordinate sweeps, geometry tolerance1e-8. All eight GPUs used, one block each. Five existing focused tests passed.

Independent actual-string re-encoding, Base construction, population geometry, exact assignment, full30-step autograd sensitivity, monotonic traces and claimed single-coordinate stopping checked for all16 solves. All24 existing future heads reconstructed from original shifted training examples and matched. Original input hashes unchanged; new output freeze hashes verified.

The24 future classifiers are reused from task020, already examined previously. This is an extension to unseen factual sentences under a fixed previously examined shift bank, NOT an independent new-shift evaluation. No bootstrap confidence intervals or human semantic labels are claimed. Fixed embedding/classifier proxies can reward unrelated or ungrammatical rewrites. Examine the complete qualitative PDF/HTML and exact CSV, not only favorable examples.

Artifacts: `summary.csv`, `population_summary.csv`, `individual.csv`, `qualitative_examples.csv`, `independent_checks.csv`, `future_checks.csv`; PDF/HTML in `figures/text_expanded_evaluation/`.
