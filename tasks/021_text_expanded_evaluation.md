# Task021 — Expand text evaluation to 64 unseen factuals

User requests more evaluated sentences. Preserve tasks019/020 and their outputs.
Use 64 classifier-negative, 5–25-word ParaDetox factuals, chosen in seeded random
order (2101), excluding ALL texts in all partitions of both prior runs. Exclude
paired duplicates using the original greedy corpus rule. Never select by Base
success, refinement outcome, or visual appeal. Withhold human neutral references
from candidate generation and optimization.

Eight consecutive populations of eight retain task020's support size and eight
favorable fitting references. Reuse its nominal128 update examples, fitted head,
MiniLM384 encoder, and task019 fine-tuned FLAN-T5-small. Same eight beams and64
samples (.8/1.1 temperatures), lexical dedup, first-four-beam Base rule, fallback,
finite-codebook solver and exact two-anchor geometry. Run frozen beta1.5 with
selected r0 and diagnostic r.25 only. T30 full-batch eta.5, smooth t.05; ten
coordinate sweeps, improvement1e-10, feasibility1e-8. No semantic gate or repair.

Reuse the exact24 task020 held-out future heads, only after output freeze. This
is an expanded factual evaluation under previously examined scenarios, NOT a new
independent shift evaluation or new calibration. No model/hyperparameter tuning.
Use all eight GPUs, one independent worker per eight-sentence population.

Record all64 factuals/Base/outputs, probabilities, FV, embedding L2, word edit,
cosine, nominal loss/sensitivity, geometry, assignment, stopping, duplicate rates,
Base failures and population membership. Independently re-encode actual strings,
recompute complete30-step updates and gradients with autograd, reconstruct all24
future heads, check feasibility/assignment and finite-coordinate stopping.
Preserve provenance hashes and save all-case PDF/HTML with failures. Aggregate
uniformly over64 factuals/24 models; sensitivity averaged across eight populations.
Do not describe classifier scores or cosine as human-rated semantic quality.

Raw: /home/xu.zhao/robust_ce_results/iclr_runs/text_expanded_evaluation/20260925_v1
Results: results/035_text_expanded_evaluation/
Figures: figures/text_expanded_evaluation/
Update state.yaml, AGENTS.md and notes/decisions.md after verification.
