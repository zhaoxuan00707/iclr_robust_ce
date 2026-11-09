# Task 004: in progress

Implemented continuous geometry-only raw optimization and invalid-only first-crossing repair. 82 tests pass. Main run: `/home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2/`. Initial full-size smoke preserved in sibling `20260916_minimal_v1/`; no old experiment overwritten.

The first smoke passed on all5927 models and3336 support points. The final preflight uses batch-consistent threshold checks and avoids redundant OT restoration when the raw iterate is already exactly feasible. The revised full-size smoke passed in58.0s. Twelve raw optimization jobs are running on four GPUs; matched repair, fresh attacks and final evaluation remain pending.
