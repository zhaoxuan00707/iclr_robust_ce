Task100 is now complete. See `../progress.md` and `final_completeness_check.json`.
The partial snapshot below is retained for provenance.

# Task100 recovery snapshot — partial

The original eight-worker run survived the interrupted session. Seven workers
are complete, with 625 independent checks passing. Three mixed-domain unit
tests also pass. These CSVs are a verified partial snapshot, not final results.
HELOC/MLP remains active; PROPLACE per-factual timeouts are preserved as failures.
No experiment was restarted, no completed raw artifact was overwritten, and
no core implementation was changed.

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1`.
Original controller PID452394; remaining worker PID452399; existing final
postprocessor PID456931 (`experiments/097_finish_appendix_tabular.py`).

Final reports are to be saved at `results/exp3_table1_single_run/summary.md`
and `results/exp4_runtime_10cf/summary.md` after all eight workers and independent
checks finish. MNIST experiments1/2 remain complete and unchanged.

The tabular run is a single-seed diagnostic reconstruction, not a verified
reproduction of historical Table1. Keep relaxed and hard categorical results,
geometry failures, native solver failures and ElliCE native robustness distinct.
The separate task101 follow-up must not replace original task100 measurements.
