# Two-seed benchmark — launched 2026-11-09

Status at launch: running, no final results yet.

- Seeds 1 and 2; four datasets/representations; MLP and logistic regression.
- All eight requested methods; 16 dataset/classifier/seed jobs.
- Four GPUs allocated through genv: 1, 2, 3, 5; dynamic job queue.
- Controller PID at launch: 2474917; persistent tmux task103_full_20261109.
- Raw root: `/home/xu.zhao/robust_ce_results/iclr_runs/tabular_replication/20261109_full_v1`.
- Controller log: `controller_20261109_full_v1.log` in this directory.
- Reproduction: `bash experiments/run_task103_four_gpus.sh` (requires a fresh output root).
- Automatic reporter109 writes `20261109_full_v1/` here after completion.

Validation: nine adapter tests and three mixed-objective regression tests pass.
The original PROPLACE error and subsequent full-MLP license-size failure are
resolved in an opt-in equivalent MILP adversary with reconstructed weight cuts.
Diabetes/logistic and [64,32] MLP end-to-end smoke tests succeed for PROPLACE.
Earlier failed smoke outputs remain preserved. ROAR invalid constructions are
recorded as failures; no post-hoc validity repair. Future individual baseline
timeouts or failures remain possible and must not be reported as successes.

See tasks/103_tabular_two_seed_replication.md for exact protocol changes.
Do not pool adapted seeds1/2 with historical seed0 without qualifying differences.
Do not modify core code while these workers run.

## Expanded to eight GPUs, 2026-11-09

User authorized all eight GPUs. New supervisor110 in tmux
`task103_eight_20261109` adopted the four live worker PIDs unchanged and started
four untouched queued jobs on GPUs0,4,6,7. At handover: 1 completed, 8 active,
7 queued. No experiment restarted, no outputs overwritten, no core/config edits.
Old scheduler PID2474917 is stopped to prevent duplicate dispatch; the new
supervisor retires it only after all adopted workers finish. New supervisor
invokes reporter109 after all jobs finish. Status is in the raw root at
`eight_gpu_status.json`; handover provenance is `eight_gpu_expansion.json`.
New controller log: `results/103_tabular_replication/eight_gpu_controller.log`.
The existing shell retains a stale CUDA_VISIBLE_DEVICES value; each worker is
explicitly assigned its physical GPU. Registry attachment covers indices0–7.

## Handover correction and recovery
The first scheduler handover was faulty: tmux automatically continued its stopped
pane leader, so the old scheduler attempted duplicate launches. Output guards
rejected those attempts before rewriting experiment artifacts, but some text
logs were truncated. The first replacement scheduler then exited on an existing
output and four of its workers were interrupted. Do not treat stale exit/status
files as the final scientific outcome.

Supervisor111 now owns scheduling. Its dedicated task103 tmux server and obsolete
schedulers are suspended together, preventing automatic continuation; active
experiment processes continue independently. The server is resumed after their
completion. See raw `handover_v2_status.json` for current workers. Recovery112
queues the four interrupted workers into distinct `_recovery1` directories,
reusing saved prepared classifiers, factuals, Base CFs, future banks and completed
calibration solves. Original raw directories remain unchanged. It uses spare GPUs
once supervisor111's queue drains, and performs the final combined reporting.
Current reports/status must disclose interrupted originals and recovered outputs.

## Completed and checked
All16 dataset/classifier/seed jobs completed, including all four recovery jobs.
No experiment workers remain running. The automatic reporter failed due to
colliding scenario/experiment `realization` and `seed` field names; reporter109
now stores scenario identifiers as future_realization/future_seed. Rerun exited0.
Final tables, raw per-seed metrics, per-future-model scores, classifier accuracy,
paired proximity changes and saved-output checks are in `20261109_full_v1/`.
Completion does not imply every baseline construction or GRACE solve succeeded;
retain their recorded failures and geometry statuses in scientific reporting.
