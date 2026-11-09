# Text toxicity: discrete sentence-codebook pilot

Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_update_aware/20260924_v1`. Task019; controller076; independent reporter081.
Frozen source and exact configs: `experiments/snapshots/text019_v1/`.

## Result

Calibration selected r=0, beta=1.02. Test Base and
all eight refined settings have100% FV on8 texts ×24 shared future models.
There is **no demonstrated future-validity improvement** in this pilot.
At selected beta1.02, all four radii return exactly the same eight sentences;
only1/8 differs from Base, mainly capitalization, apostrophes and punctuation.
Sensitivity decreases, but this does not establish improved semantic detoxification.

| method         |         FV |   nominal |   sensitivity |   semantic_cost |   word_edit |   cosine_similarity |
|:---------------|-----------:|----------:|--------------:|----------------:|------------:|--------------------:|
| Base           | 100.000000 |  0.000319 |      0.013229 |        0.472127 |    0.311924 |            0.860138 |
| r0_beta1.02    | 100.000000 |  0.000228 |      0.009241 |        0.472418 |    0.344819 |            0.860089 |
| r0.05_beta1.02 | 100.000000 |  0.000228 |      0.009241 |        0.472418 |    0.344819 |            0.860089 |
| r0.25_beta1.02 | 100.000000 |  0.000228 |      0.009241 |        0.472418 |    0.344819 |            0.860089 |
| r1_beta1.02    | 100.000000 |  0.000228 |      0.009241 |        0.472418 |    0.344819 |            0.860089 |

FV is percent; cost is mean Euclidean distance in normalized MiniLM384 embedding
space; word_edit is normalized word-level Levenshtein distance from the factual.
Do not compare this cost or sensitivity numerically with image experiments.
All methods have8 distinct test outputs. Calibration Base and refined FV are87.5%;
one failed calibration Base remains in the population. No validity repair.

## Method

ParaDetox toxic/neutral pairs, exact shared-text exclusion before seeded splitting.
2000 fitting pairs,256 validation pairs,128 nominal-update pairs,256 pairs each
for calibration future models, test future models, calibration factuals and test
factuals. Seed1901. Pair partitions were independently checked disjoint. The
MiniLM encoder is frozen. A logistic head is fitted with C10/LBFGS; validation
accuracy is90.82%, held-out test-pair accuracy
89.84%. These accuracies are distinct from
future CF validity. Pretrained encoder/LM corpus overlap cannot be excluded.

FLAN-T5-small is fine-tuned only on fitting pairs for300 AdamW steps, batch16,
lr1e-4. It is a small language model pilot, not a multi-billion-parameter LLM.
Each factual gets8 beam and16 sampled rewrites. Base is the nearest currently
valid first-four beam rewrite, or highest score if none is valid. Pool sizes are
132 calibration/112 test sentences, including factuals and fitting references.
Human neutral test counterparts are withheld from generation/selection; they are
saved only as descriptive references in qualitative_examples.csv.

Each future update is all3 full-batch SGD steps, lr.5, on128 frozen sentence
embeddings with64 examples per class. All head weights and bias update; the
encoder and generator remain fixed. Mean smooth hinge (.05 temperature) plus
r times sqrt(128)||dF/dX|| is computed through all3 steps. The gradient is averaged
across population supports before taking its norm. Data uncertainty is in the
continuous sentence-representation space, not categorical token IDs.

Q has8 equal-mass supports, restricted to actual sentence codes. Search enumerates
single-support replacements for up to10 sweeps under exact Euclidean W2
Base→Q plus Q→reference <= beta times Base→reference. Final deployment uses an
exact one-to-one assignment. All16 runs stop with no improving feasible single
replacement; this is not a global optimum or continuous Algorithm1 convergence.
Finite-codebook restriction is an explicit numerical adaptation.

Future models use disjoint calibration/test sentence pools and within-class
length/error/topic-biased sampling;4 severities ×2 seeds ×3 families per split.
Exact model-update, displacement and random-seed records are in the raw run.
These scenarios are not certified inner maximizers or guaranteed within each
r-ball. Test model checkpoints were created after output freezing. Calibration
selection was independently checked and predates all test output saves.

## Latent-space finding

Sentence-code lookup makes deployed text exact, but arbitrary embedding vectors
lack an inverse. Both alpha=.5 interpolations obey the ambient geometry before
snapping and violate beta1.02 after nearest-sentence snapping. Test geometry rises
from1.270212 to1.728938 (budget1.295616); mean round-trip embedding error0.634123.
The finite sentence set is not geodesic. This directly illustrates why an image-
style continuous latent solver followed by text decoding cannot be assumed valid.
These diagnostics concern this candidate pool, not every possible text decoder.

## Verification and limitations

All192 pause-inventory hashes, source-snapshot hashes, input hashes and test-freeze
hashes pass. All16 outputs were independently re-encoded, checked for exact
assignment/geometry/cost, full-update nominal loss and training-feature sensitivity,
per-example future probabilities and FV. All48 future heads were reconstructed
using autograd SGD gradients and matched their saved weights/displacements.
Four focused unit tests passed before execution. Both original end-to-end
finite-difference checks had relative error below3e-9. Experiment compute took
1.7 minutes. Eight GPUs handled parallel generation/model jobs; finite
coordinate enumeration itself uses CPU NumPy/SciPy, with GPU objective checks.

Classifier acceptance is not human-rated non-toxicity, fluency or meaning
preservation. Rewrites can remove meaning: e.g. case7 removes the name/token
'dick' from a sentence, and case5 drops its trailing target. Some accusations
and hostile content remain accepted. No independent toxicity model or human
annotation was performed. This small saturated pilot is evidence of a working
discrete realization, not evidence that GRACE improves text robustness.

## Artifacts

- `figures/text_toxicity_update_aware/qualitative_all8.pdf`: eight pages, every case,
  factual/Base/r0/r.25/selected, and two word-difference panels.
- `figures/text_toxicity_update_aware/qualitative_all8.html`: readable browser view.
- `qualitative_examples.csv`: exact saved strings and separate human references.
- `independent_checks.csv`, `future_checks.csv`, `radius_sentence_equality.csv`.
- `latent_snapping_audit.csv`, `headline.json`, cal/test summary CSVs.

A future experiment should first expand and audit semantically faithful candidate
coverage and use a predeclared harder update protocol. Do not tune current settings
on the saturated test results or claim a positive-radius gain from this run.
