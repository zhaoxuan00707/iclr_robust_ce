# Task020 — Repair discrete-text stagnation on fresh held-out pairs

## Diagnosis and scope
User requests a fix for identical text outputs. Task019 is preserved and completed.
Read-only diagnostic082 found zero single-support lexical changes feasible at
beta1.02 for all16 prior cases. Minimum beta for any lexical change ranged about
1.032–1.209; nominal-improving changes can require1.326. Thus the image-like small
budget was not appropriate for a sparse sentence codebook. Old future-update
probability shifts were small (mean test absolute change about.00246).
This follow-up changes representation coverage, budget grid and prescribed update;
its results are not a matched causal ablation against the old cohort/protocol.

## Frozen reuse and fresh data
Reuse original MiniLM, fitted logistic head and fitting-only FLAN-T5 checkpoint
from task019; do not retrain or overwrite them. Preserve Base construction:
nearest valid among first4 deterministic beam outputs, highest-score fallback.
No deliberate lowering of Base margin or new validity-repair step.
Reconstruct ParaDetox's greedy shared-text exclusion exactly as task019; remove
EVERY previously used pair/text across all7 old partitions. Shuffle remaining
pairs seed2001. New128 nominal-update pairs,256 cal-future pairs,256 test-future
pairs,256 cal-factual pairs and256 test-factual pairs. First8 classifier-negative
5–25-word sources per factual split. Never filter cases by Base/FV success.
Old fitting/validation assets may be reused; new evaluation pools are disjoint
from the old experiment and from one another. Test human neutral counterparts
remain withheld from generation, optimization and selection.

## Candidate coverage repair
Generate8 deterministic beams plus64 sampled rewrites per factual:32 at temp.8,
32 at1.1, top-p.95, max64 generated tokens. Use saved generator, prefix detoxify:,
seeds200100/201100+case. Union the split's generated texts, Bases, factuals and8
original favorable fitting references. Keep original Base strings first, then
reference/factual strings, then alternatives. Deduplicate alternatives by lowercase
word sequence (regex Unicode word tokens), so case/apostrophe/punctuation-only
variants cannot be counted as successful refinement. Drop empty alternatives only.
No semantic-validity gate; record poor grammar/meaning preservation honestly.
Classifier always scores actual saved strings, not canonical strings or invented
latent points. This filter is candidate-construction only, not validity correction.

## Objective, solver and grid
Same task019 smooth hinge .05, empirical-RMS sensitivity and exact Euclidean W2
geometry in normalized MiniLM384 representations. Fixed complete head update now
30 full-batch SGD steps, lr.5,128 examples64/class. Differentiate ALL30 steps.
This is a fully prescribed longer update, not truncated differentiation.
Sensitivity remains continuous sentence-feature uncertainty, not token robustness.
Q has8 equal-mass realizable sentence supports, global lookup and one-to-one
Base assignment. Discrete solver: exact single-coordinate enumeration up to10
sweeps with strict improvement1e-10 and geometry tolerance1e-8.
For efficiency, compute an exact Gram-factor representation of candidate adjoints;
verify its population norms against direct full-feature gradients. Preserve full
adjoints for independent auditing. No differentiation through decoding.

r={0,.25,1,4}, beta={1.02,1.10,1.25,1.50}:16 settings/split,32 solves total.
Eight GPU workers generate proposals and compute update objectives; CPU exact-OT
coordinate enumeration may use compact gradient factors. Stop only after no
improving feasible single replacement or the fixed10-sweep cap; no global claim.
Record lexical-change availability per beta and changed-word counts versus Base
and matched-beta r0. A wider budget may permit unwanted semantic changes.

## Future shifts, freeze and evaluation
Same sampling families length/error/topic; severities now{1,2,4,8},2 repeats,
24 cal/24 test heads from NEW disjoint pools. Labels64/class remain fixed.
Use seeds211000+job for calibration and221000+job for test. All use30-step updates.
These are stress distributions, not guaranteed members of every r-ball or inner
maximizers. Compute classwise OT displacement certificates.
Select only by cal meanFV, then semantic cost,r,beta. Freeze before test optimization;
freeze test strings before constructing/evaluating new test future models. Never
change settings in response to final test outcomes.

## Completion
Raw runs under /home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_followup/<fresh>;
use experiments/run.py and eight workers. Lightweight results034 and text-followup
figures. Save full grid, provenance, current/FV, nominal/sensitivity/cost, word edit,
cosine, unique outputs and changed-vs-Base/r0 diagnostics. All-case PDF with both
word-difference panels, not a cherry-picked improvement example. Independently
check re-encoding, all30 update steps, geometry, assignments, outputs, data split
exclusion, selection/freeze and future models. Record whether bottlenecks were
removed separately from whether robustness actually improved. If identical outputs
or saturated FV persist, report that rather than manufacture differences.
Update AGENTS.md, decisions, state.yaml. Original outputs unchanged.
