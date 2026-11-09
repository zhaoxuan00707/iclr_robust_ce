# Task019 — Toxic to non-toxic text: realizable sentence-codebook GRACE pilot

## Request and scientific scope
Design and execute a text/LLM detoxification experiment, explicitly accounting for
text's different latent space. Continue eight-GPU use. Follow PDF62 Section3 plus
smooth hinge .05 for the population objective, full prescribed update, exact W2
budget and equal-mass one-to-one deployment. This is a **discrete text adaptation**,
not an unchanged implementation of Algorithm1's continuous support-gradient solver.
No manuscript claims or historical results are modified.

## Why a different realization is required
Token IDs are categorical; autoregressive argmax/beam decoding is discontinuous.
A sentence embedding has no guaranteed inverse, and linear interpolation need not
be a realizable sentence. Do NOT optimize arbitrary hidden states and report their
classifier scores as decoded-text results; do NOT use straight-through gradients.
Use frozen normalized MiniLM384 sentence embeddings as explicit classifier inputs
and Euclidean chord distances. Keep every support on a finite codebook of actual
sentences proposed by a language model. Decoding is exact lookup of saved text.
Global codebook lookup and Base-to-Q assignment are not factual-conditioned.

This finite codebook is not geodesic: the ambient two-anchor W2 inequality is
computable but continuous/geodesic existence arguments do not transfer to it.
Search uses exact discrete coordinate replacement, without a global-optimality
claim. The sensitivity is with respect to continuous sentence representations,
not token edits, word embeddings, or language-model weights. Its continuous
neighborhood is a relaxation; do not claim certified discrete-text robustness.
Explicitly audit interpolated embeddings followed by nearest-text snapping.

## Assets, data splits and generator
Public ParaDetox toxic/neutral pairs, with source URL/revision/hash recorded.
MiniLM: sentence-transformers/all-MiniLM-L6-v2, mean pooling plus L2 normalization,
max256 tokens. Generator: google/flan-t5-small (77M class); this is a small language
model feasibility pilot, not a multi-billion-parameter LLM experiment.
Download assets externally under /home/xu.zhao/robust_ce_data/text019_assets_v1.
Remove exact shared-text pairs greedily (case/whitespace normalized) before seeded
shuffle1901, so neither side of retained pairs overlaps across partitions.
2000 fitting pairs,256 classifier-validation pairs,128 nominal-update pairs,
256 calibration-future pairs,256 test-future pairs,256 calibration-factual pairs,
256 test-factual pairs. Fail clearly if too few pairs remain; never merge splits.
Human neutral counterparts of cal/test factuals are withheld from generation and
optimization; use them only as a separate descriptive reference in final reports.

Train logistic regression (C10, LBFGS max1000, seed1901) on frozen fitting embeddings.
Fine-tune FLAN-T5-small on fitting pairs only: prefix 'detoxify: ',300 AdamW steps,
lr1e-4, batch16, maximum96 source/target tokens, seed1901; save final checkpoint.
No future/cal/test outputs tune this generator. Report validation and held-out
classifier accuracy; inherited pretraining overlap cannot be excluded.
Choose first8 current-toxic sources per factual split (5–25 words, shuffled order).
Do not filter them by Base success. Generate8 deterministic beam rewrites plus
16 seeded temperature.8/top-p.95 rewrites each. Seed190100+split offset+case.
Deduplicate exact strings; remove empty strings only. Base: minimum Euclidean
semantic cost currently non-toxic among first4 beam outputs; if none valid, retain
highest-scoring beam output and record failure. No robust-model information in Base.
Keep all candidates regardless of current validity for GRACE. Population codebook
is the union of all own-split rewrites, Bases, factuals and8 current-favorable
neutral fitting references. No test sentences in calibration codebook.

## Update, uncertainty and optimization
Use128 nominal-update sentences,64 toxic/64 neutral, from its separate partition.
Fixed complete update: three full-batch SGD steps, lr.5, on all logistic head weights
and bias; labels fixed, no weight decay. Differentiate all three steps.
F(Q)=mean .05*softplus((.5-p_non_toxic(Update(f0,X),Q))/.05).
S(Q)=sqrt(128)*||dF/dX||_F. Minimize F+r*S under
W2(Base,Q)+W2(Q,P1)<=beta*W2(Base,P1). N=M=8 CF supports, eight favorable references.
Precompute each candidate's nominal loss and full training-feature gradient;
population sensitivity is the norm of the **mean gradient**, not the mean norm.
Verify cached population values against direct unrolled autograd and directional
finite differences; include gradient cancellation in tests.

Grid r={0,.05,.25,1}, beta={1.02,1.10},16 solves across cal/test, eight GPU workers.
Initialize Base. Up to10 full coordinate sweeps, visit supports0..7; enumerate the
entire split codebook, accept lowest-surrogate exactly feasible replacement with
strict improvement>1e-10. Stop after a full unchanged sweep. Record this finite
coordinate condition, not global convergence. Exact Hungarian OT, tolerance1e-8.
No validity repair, semantic gate, or post-selection replacement. Equal-mass
supports may repeat; one-to-one assignment is between support indices.

## Future models and selection
24 calibration and24 test future models use the same three-SGD-step update, on128
sentences (64/class) sampled with replacement from their disjoint future pools.
Three within-class sampling shifts: sentence length, current-classifier error,
and similarity to a fixed fitting-only topic anchor. Standardize each statistic
within class; sample proportional to exp(severity*clipped_z), z clipped[-3,3].
Severities {.5,1,1.5,2}, two seeds each; cal191000+, test201000+.
Compute label-preserving paired classwise OT displacement to nominal update data.
These are held-out stress distributions, not inner maximizers or guaranteed
members of every radius ball. Construct/evaluate test models only after selection
and all test outputs are frozen. Selection: cal FV descending, semantic cost,
r, beta ascending. Freeze selection before test optimization. Shared factuals and
future models across methods; count failed Bases and selected outputs.

## Deliverables and validation
Raw root /home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_update_aware/<fresh>.
Task/config/code before execution, through experiments/run.py; record revisions,
commands, seeds, hashes, paths and runtimes. No large assets or corpora in Git.
Results results/033_text_toxicity_update_aware; figures figures/text_toxicity_update_aware.
Report FV/current validity, mean embedding L2/cosine preservation, normalized word
edit distance, exact unchanged-output rate, nominal loss, sensitivity, surrogate,
geometry, collapse/unique-output count, and semantic mismatch risks. Score actual
saved strings through encoder/head, not arbitrary latent points. A separate human
reference is not a method baseline selected using unseen targets. No human-rated
non-toxicity or fluency claim from classifier success alone.
Qualitative PDF: all8 factual/Base/r0/r.25/selected texts; highlight word edits for
r.25 versus r0 and r0 versus Base (text analogue of the two image heatmaps). Add
ambient interpolation/snapping diagnostics to document the representation gap.
Independent checks: re-encode outputs, recompute OT/assignment/cost, full update
predictions/FV, frozen selection, input hashes, and surrogate cache equivalence.
Update AGENTS.md, notes/decisions.md, state.yaml only with verified outcomes.
