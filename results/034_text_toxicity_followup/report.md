# Text toxicity follow-up: candidate coverage and geometry repair

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_followup/20260924_v2`. Task020; controller083; independent reporter084.
Parent task019 results are unchanged. New cohort/protocol: do not interpret this
as a matched causal ablation of any single change.

## Results

Calibration selected r=0, beta=1.5,
FV=86.46%. All settings were frozen before final test scoring.

| method        |        FV |   nominal |   sensitivity |   semantic_cost |   word_edit |   cosine_similarity |
|:--------------|----------:|----------:|--------------:|----------------:|------------:|--------------------:|
| Base          | 85.416667 |  0.042454 |      1.081432 |        0.391647 |    0.216048 |            0.903074 |
| r0_beta1.5    | 99.479167 |  0.000118 |      0.023938 |        0.630717 |    0.554157 |            0.723456 |
| r1_beta1.5    | 99.479167 |  0.000118 |      0.023938 |        0.630717 |    0.554157 |            0.723456 |
| r0.25_beta1.5 | 99.479167 |  0.000118 |      0.023938 |        0.630717 |    0.554157 |            0.723456 |
| r4_beta1.5    | 99.479167 |  0.000118 |      0.023938 |        0.630717 |    0.554157 |            0.723456 |

FV is percent. Selected versus Base changes 7/8 word sequences;
r.25 versus matched-beta r0 changes 0/8. Selected FV gain over Base:
14.06 percentage points; selected
gain over nominal r0: 0.00 points.
These are classifier-relative outcomes, not human-rated semantic detoxification.
Cost is mean factual-to-output normalized MiniLM embedding L2; word-edit cost
and cosine similarity are shown separately. Preserve failures and all8 examples.

## Why the previous experiment stagnated

Read-only parent diagnostic082 found zero feasible single-support word-changing
replacements at beta1.02 for all16 old cases. Changes of capitalization/apostrophes
could fit, but the codebook's next word-changing candidates were too far away.
This is a finite-neighborhood finding, not a proof that all simultaneous support
moves are impossible. Previous test Base probability changes averaged only.00246;
Base already scored100% FV. Increasing r alone cannot create missing feasible
sentences or increase a saturated validity metric.

## Follow-up design

Original fitted logistic head, MiniLM and fitting-only FLAN-T5-small are reused.
No intentional degradation of Base margins. Same first-four-beam nearest-valid
Base construction with unchanged failed-output fallback. Fresh ParaDetox pairs
exclude every text in every prior partition; nominal/cal/test/future pools are
also mutually disjoint. Seed2001;8 factuals per split, no Base-success filtering.
Human neutral counterparts remain excluded from generation and optimization.

8 beam plus64 sampled rewrites per case (32 each at temperature.8 and1.1).
Alternatives duplicating lowercase word sequences are excluded so capitalization
and punctuation variants do not masquerade as improvement; Base/reference/factual
anchor strings are preserved. No current-validity or semantic gate is added.

The prescribed update is30 full-batch SGD steps at lr.5 on128 sentence embeddings,
64/class. All30 steps are differentiated. Smooth hinge temperature.05, nominal
population mean plus r*sqrt(128)*norm(mean feature adjoint), exact two-anchor W2
constraint and one-to-one deployment remain. Q supports are actual sentence codes;
no arbitrary-vector decoder or straight-through token derivative is used.
The sensitivity neighborhood is continuous feature space, not discrete token edits.
An exact gradient-Gram factor accelerates CPU enumeration; direct full-adjoint
values are independently verified. Eight GPUs handle generation/update scoring;
CPU NumPy/SciPy performs the discrete coordinate search.

Grid r={0,.25,1,4}, beta={1.02,1.1,1.25,1.5}:32 solves. Up to10 coordinate sweeps,
strict improvement1e-10, exact geometry tolerance1e-8. 32/32 saved runs reached
the no-improving-single-coordinate condition; no global optimum is claimed.
Future sampling families length/error/topic, severities1/2/4/8, two repeats;
24 independent cal and24 test models. Seeds211000+/221000+. All new heads use the
same complete30-step operator. These are stress scenarios, not certified inner
maximizers or guaranteed members of each radius ball. Calibration selection uses
FV, then embedding cost, r, beta. Test heads are created after outputs are frozen.

## Feasible lexical coverage

Counts of word-changing one-support replacements available from Base:

|                |   min |   median |   max |
|:---------------|------:|---------:|------:|
| ('cal', 1.02)  |     0 |      0   |     0 |
| ('cal', 1.1)   |     1 |      4.5 |    10 |
| ('cal', 1.25)  |    12 |     18   |    26 |
| ('cal', 1.5)   |   189 |    190   |   190 |
| ('test', 1.02) |     0 |      0   |     1 |
| ('test', 1.1)  |     1 |      2   |    19 |
| ('test', 1.25) |     3 |     22.5 |    43 |
| ('test', 1.5)  |   225 |    226   |   226 |

These counts distinguish removal of the search bottleneck from demonstrated FV
improvement. Wider geometry can also admit semantic drift or unrelated sentences.
The pool is discrete/non-geodesic; continuous geometric theorems do not automatically
apply. The parent interpolation/snapping failure remains documented in results033.

## Verification and interpretation

32 outputs independently checked: actual-string re-encoding, full30-step
nominal/sensitivity recomputation, geometry, assignment, costs, all per-example
future probabilities and FV. All48 future heads reconstruct under autograd SGD and
match saved weights/displacements. Old-partition exclusion, selection/freeze order,
input hashes and raw inventory hashes checked. Four inherited objective/geometry
tests plus one new gradient-factor cancellation test passed. Finite-difference
checks are in gradient_audit.json. Experiment time1.4 minutes.

Source corpus labels and classifier acceptance can disagree with human judgments;
fluency and meaning can deteriorate. No independent toxicity model or human study
was added. Compare all examples rather than treating changed text as evidence of
improvement. The test cohort remains only8 cases. No manuscript claims changed.

## Files

All-case PDF/HTML: `figures/text_toxicity_followup/qualitative_all8.pdf` and
`qualitative_all8.html`. Full grid, exact strings, radius equality, parent
stagnation diagnostics and independent checks are in results034. Original raw
runs/checkpoints preserved; no completed prior experiment was rerun.

## Qualitative completion review

The wider grid fixes the restricted one-step search neighborhood, but it does NOT
fix semantic preservation or demonstrate a positive-radius FV advantage. Selected
r0/beta1.5 changes7/8 word sequences and reaches99.48% test FV versus85.42% Base,
while all positive radii at beta1.5 return the same outputs as r0. Cost rises from
.391647 to.630717 and distinct outputs decrease from8 to7.

In particular, cases1 and4 both become "I will watch the movies"; case4 originally
expressed a completely different proposition. This is an explicit example of
population geometry admitting semantically unrelated replacement and collapse.
Case2 becomes "How does they swim?", a grammatical regression. Case5 still contains
"troglodyte" and introduces awkward wording despite higher classifier acceptance.
These issues prevent calling the new outputs successful meaning-preserving CFs.

At beta1.25, r.25 and r0 differ lexically on case6; this is a diagnostic, not the
calibration-selected setting. At beta1.1, their only difference is a period on
case4: although generated punctuation-only alternatives were removed, the original
factual and Base anchor strings were deliberately retained, allowing this move.
Do not claim all punctuation-only transitions were eliminated.

Both example rendering and the eight-page PDF count were checked. Next scientific
repair requires explicit per-factual semantic correspondence/meaning safeguards
and better candidate-quality checks; adding these changes the text realization
and must be declared in a new task, not retrofitted to this held-out run. Different
radii need not produce different optima even with a correct implementation.
