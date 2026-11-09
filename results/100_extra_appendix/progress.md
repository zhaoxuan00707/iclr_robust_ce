# Task100 — complete and verified

All four requested experiments are saved. All eight tabular workers exited
successfully, and the final automatic postprocessor completed. No Task100
experiment or reporting process remains running.

- Experiment1: `results/exp1_mnist_T_mismatch/summary.md` —180 independently
  reconstructed MNIST checkpoints; gradual FV decrease from99.87% atT3 to99.61% atT40.
- Experiment2: `results/exp2_mnist_no_geometry/summary.md` —constrained and
  unconstrained outputs are byte-identical; the constraint was inactive here.
- Experiment3: `results/exp3_table1_single_run/summary.md` —62 reported rows,
  726 independent checks passing across all eight dataset/classifier workers.
- Experiment4: `results/exp4_runtime_10cf/summary.md` —24 timing rows including
  Base, with native failures and unsupported methods retained.

Final completeness/preservation check:
`recovery_20260926/final_completeness_check.json`. All required outputs exist;
all414 previously hashed completed-worker files are unchanged. Three focused
mixed-domain tests passed during recovery. The recovery CSVs remain an archived
partial snapshot; use the final experiment3/4 tables for complete results.

Raw tabular run:
`/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1`.

These are single-seed reconstructed diagnostics, not verified reproduction of
historical Table1. Mixed-domain GRACE runtime measures relaxed simplex outputs,
not ten legal hard-category CFs. Hard realization is scored separately and can
break geometry. CPU/GPU differences and shared-host overlap limit runtime
comparisons. See `recovery_20260926/reporting_notes.md` for interpretation details.
