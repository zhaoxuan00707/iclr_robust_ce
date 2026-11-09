# Task100 execution progress (partial)

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1`.
Seed0, eight independent dataset/classifier GPU workers. Full run is active.
Controller094; postprocessor097 waits for all workers, then runs hard-category
stress checks096 and independent reporter095, and updates AGENTS/state/decisions.

Diabetes/logistic finished;53 independent checks pass, including all45 held-out
model reconstructions, CF metrics, geometry, and calibration selection.
Wachter Base FV85.1064%, cost0.927620; selected GRACE FV100%, cost1.405188.
This is an early single-worker diagnostic result, not the final four-dataset report.
The other workers and ten-CF timings must finish before Task100 is complete.

Preflight v1 and v2 remain preserved. The source is frozen for the running jobs.
No historical experiment outputs were replaced. Categorical treatment follows
PDF73 A.5; unsupported mixed ROAR/PROPLACE and native solver/license failures
remain explicitly labeled. Final summaries will replace only the prior not-run
status CSVs in results/exp3_table1_single_run and results/exp4_runtime_10cf.
