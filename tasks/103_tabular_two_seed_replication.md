# Full tabular replication with explicit baseline adapters

User authorized two additional seeds and autonomous baseline repairs. Seeds 1,2
are fresh data split/classifier training seeds, unlike task102's population-only
sampling seeds. Four representations: HELOC continuous, Diabetes, German mixed,
FICO mixed. HELOC/FICO share the same underlying corpus. Both MLP/logistic;
Wachter, GRACE-Wachter, DiCE, GRACE-DiCE, T-Rex, ROAR, PROPLACE, ElliCE.

Preserve seed0 task100 and preflight artifacts. Use original 128 maximum factuals,
training, calibration grids, update operator, GRACE geometry/optimizer/matching,
45 calibration and 45 held-out future scenarios. Scenarios include deterministic
repeats; do not describe them as 45 independent distributions. New seed-specific
scenario offsets. Calibrate before constructing test banks. Record all failures.

Opt-in fixes (`replication_fixes`): favorable-label T-Rex candidates; signed fixed
inputs in native interval checker; cache invariant PROPLACE candidate screening
once per fitted model/radius, with 1800s preparation cap, then original 30s per
factual cap; fresh PROPLACE state each instance. Standardized simplex constraints
for mixed PROPLACE, projected ROAR, separately evaluated hard realization.
ElliCE requires native robustness rather than merely validity. Baseline selection
maximizes failure-aware calibration FV then minimizes cost/radius. These explicit
protocol changes mean seed0 cannot simply be pooled with new adapted results.
No GRACE validity repair or geometry changes. Native baseline robustness models
remain method-specific; this is not an assertion of equivalent guarantees.

Tests: test_mixed_baseline_adapters.py and test_replication_adapters.py, 7 pass.
Raw outputs: /home/xu.zhao/robust_ce_results/iclr_runs/tabular_replication/.
Configs: experiments/configs/tabular_replication_seed{1,2}.json.
Controller: experiments/108_run_tabular_replication.py; reporter109.

## Validated PROPLACE solver fix, 2026-11-09
The output interval in the vendor adversary failed for signed logistic inputs
and did not connect output weight variables to its objective. A corrected
explicit formulation passes exact linear/small-MLP checks but exceeds the
restricted quadratic-model license on the full MLP. The production adapter
therefore eliminates independent box-bounded weights: for fixed previous
activation h each row reaches exactly [w.h+b-d(||h||_1+1),
w.h+b+d(||h||_1+1)]. Hidden h>=0 makes bounds affine; the first input is fixed
and signed. The resulting MILP projects the same feasible adversarial network,
retaining the original M=1000. Reconstruct a bounded row perturbation for every
optimized activation and verify its forward replay before adding any cut.
Tests compare min/max to explicit weight optimization on linear and one/two
hidden-layer tiny networks. Full [64,32] Diabetes MLP and logistic smoke tests
both now succeed. Nine adapter tests and the mixed-objective regression tests
pass. This does not guarantee that all factual optimizations converge within
the unchanged 30-second per-instance budget.

Launch: `bash experiments/run_task103_four_gpus.sh` (new root only).
Persistent tmux session: task103_full_20261109. GPUs 1,2,3,5, dynamically scheduled.
Raw: /home/xu.zhao/robust_ce_results/iclr_runs/tabular_replication/20261109_full_v1.
Controller automatically invokes reporter109 after all workers finish.
