# German MLP PROPLACE license repair

User requested resolving task103 PROPLACE solver failures. Seeds1/2 selected
radius.005; successful original outputs must remain unchanged. Original master
uses semi-continuous absolute-value auxiliaries and can exceed restricted Gurobi
size after multiple adversarial cuts. Opt-in fallback uses equivalent continuous
nonnegative auxiliaries and HiGHS only on license-size error10010. A subsequent
exact reformulation tightens ReLU bounds over the same candidate convex hull,
removes provably stable activation binaries, and preserves original big-M1000
restrictions. No change to uncertainty set, candidate set, radius, or network.

Runner114 retries only20/6 failed/skipped cases into external proplace_license_fix
roots. Seed1_v1 repairs20/20; seed2_v1 repairs5/6, v2 tighter formulation retains
one30s timeout. Reporter115 independently checks all95 native robustness outcomes
and saved45 future models;94/95 robust. Same-budget results are in results104.

Single unresolved seed2 factual41 is retried at300s in seed2_v3 as an explicitly
extended-budget diagnostic, not a replacement for the30s benchmark. Do not hide
the different computational budget. Independent reporting remains required for
v3. Solver unit tests3 pass; previous9 adapter tests pass. First extra test had
a fixture-only Sigmoid.bias lookup error, corrected to select the final Linear.
Raw historical/task103 artifacts remain untouched; derived results are separate.
