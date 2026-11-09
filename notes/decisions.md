
## Task 002: preserve baselines and require the complete path specification

The current `tasks/002_baseline.md` starts midway through optimization and lacks sections 1–9. Its path objective and representation cannot be reconstructed uniquely from sections 10–20. The missing specification was requested; dependent implementation and experiment launch are pending, rather than substituting an invented objective.

The existing single-budget run remains immutable at `/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1/`. Read-only preflight results are recorded in `results/002_baseline_audit.md` and its JSON/CSV companions. The frozen source endpoint's substantial training-box violations must be reconciled with any full-path box constraint explicitly. Neither the endpoints nor the admissible domain were changed.

This records a specification blocker, not lack of authorization to execute the user's complete task. No manuscript claims or production defaults changed.

## Core code ownership after migration

The imported HELOC/DIABETES workflows now live in `experiments/src/lift_ce/` in
`/home/xu.zhao/iclr_robust_ce/`. Future development of this core should happen here.
The old `/home/xu.zhao/robust_ce/` source remains unchanged as historical reference.
Small RobustX/ElliCE sources and licenses are included locally to avoid relying
on old editable source installations. Import provenance is in
`experiments/provenance/core_import.json`.

Use `experiments/run.py` to select repository-local code and explicit new outputs
under `/home/xu.zhao/robust_ce_results/iclr_runs/`. Historical data, checkpoints and
raw outputs remain external and immutable; this is code consolidation, not an
artifact migration or experiment rerun. Scientific functions are preserved;
administrative changes require explicit core CLI output and report vendor
snapshot revisions correctly. This migration does not resolve task 002's missing
specification or alter any manuscript claims.

## Task 002 preparation supplement (2026-09-16)

The missing original specification was not recovered. A new explicit supplement
is recorded in `tasks/002_path_specification_supplement.md`, with a specification
config in `experiments/configs/path_level_geodesic_wro_v1.yaml`. The default path
objective averages the per-lambda max-model population hinge over the common
grid; K=2 counts intermediate control distributions. These are newly declared
choices, not claims about the missing original text. No solver run has used them.

Keep the training box as a control-particle constraint and complete-path range
violations as diagnostics, matching the inspected generation/control scope.
Do not impose an incompatible full-path box constraint on unchanged sources.
Current validity remains required at every deployment-grid support point.
Read-only exact endpoint checks pass all twelve split/budget cases, without
proving path feasibility. The same absolute B_geo makes beta=1 a training-only
shortest-geodesic special case; held-out batches may have positive slack.

Implementation and small construction smoke checks are next. No new scientific
results, frozen-anchor changes, budget relaxations or GPU runs occurred in this
preparation step. See `results/004_path_preflight.md`.

## Task 003: direct favorable-region WRO (2026-09-16)

Task 002 was cancelled; preserve its partial results and do not reuse its path objective. Task 003 restricts 3336 equal target-mass particles to favorable graph nodes and minimizes the true max-model population hinge subject only to the total intrinsic graph-W2 budget. Unit edge lengths replace the old risk weights. Float64 ReLU affine-piece enumeration rejects 65 inherited edges; the graph remains mass-compatible without further repair.

Predeclared local solver: 32 feasible-descent iterations, 12 batch backtracks, outer-model proposal temperature 100; exact graph OT and true risk decide acceptance. Carry smaller-budget feasible incumbents to respect nested sets. No global solver claim. Query CFs use one nearest admissible frozen-node leaf each, so queries cannot shorten construction routes. The learned Q is fixed; every query batch gets separate OT and explicit absolute-budget checks. No new hard feature box in the inherited all-mutable continuous representation. Four exposed GPUs (physical 1,2,3,5) are available.

Task-003 calibration attachment audit: intrinsic D_cal=8.566627 exceeds even B_geo(beta=1.20)=8.312639. Thus no calibration-batch distribution can satisfy any predeclared absolute budget on this graph/query approximation. Keep the budgets, graph, Q and selection rule unchanged; label heldout operating-point outcomes exploratory and deployment geometry unsupported. This does not prove continuous favorable-region infeasibility.

Task 003 completed: all4 training controls are exactly graph-budget-feasible and all sampled intrinsic outputs are current-valid;5387+540=5927 shared-bank models evaluated. Direct-Q risk falls .488634 to .476140; fresh discovery changes no column maximum. No robustness superiority over the old waypoint/Euclidean controls is demonstrated. Calibration/test graph anchor distances exceed every frozen budget, so query results remain exploratory and deployment unsupported. Detailed evidence: results/006_favorable_region.md. Production defaults and paper claims remain unchanged.

## Task 004: continuous learning and minimal restoration

The raw optimizer retains continuous support coordinates, the previous unregularized Euclidean W2 geometry and optimizer settings, and the observed feature box. It removes all f0 projection/retention logic: f0 is absent from `solve_raw`. The current clean 5,927-model bank is frozen; old continuous feasible supports are included as explicitly rescored numerical incumbents. Geometry-only feasibility selects raw supports before any validity repair.

Restoration enumerates the ReLU logit's affine pieces along each corresponding target segment, identifies the first favorable interval, and refines with the canonical float32 model. Whole-segment bisection alone is insufficient for a non-monotone score. Valid raw supports remain bitwise unchanged. Exact OT is recomputed after repair; over-budget repairs receive a declared constrained invalid-particle-only search or are marked numerically infeasible. No threshold or budget relaxation.

The previous controls were already valid; much of the previous failure was at interpolated outputs. The primary repaired-control path uses exact expanded TRAINING-bank lambda selection restricted to wholly current-valid outputs. A separately labeled diagnostic repairs the historical continuous paths' final outputs at predeclared frozen-old-bank training lambdas, preserving split masses/identities. It is not a newly learned path and receives no targeted fresh search; its conclusions must reflect that limitation. No future outcomes select raw supports, repair sizes, beta, or lambda.

## Future-shift severity figure (2026-09-16)

Use the supplied summary values exactly. At the user's instruction, stop artifact
searches and omit unavailable GRACE S6; do not estimate it from Overall.
The x-axis is external evaluation stress severity, independent of GRACE's
internal Wasserstein radius. No experiment rerun or manuscript claim change.
See `results/future_shift_severity.md` for figure provenance and missingness.

## 2026-09-16 — Task 001 asset discovery and GPU visibility

The genv nvidia-smi shim filters the initial four-device allocation; direct
/usr/bin/nvidia-smi exposes eight physical A100s. The user requests seven GPUs;
allocation IDs remain to be resolved without disturbing other users' jobs.
Discovered `4_to_9` source denotes MNIST digit labels, not classifier replicates.
Saved compatible CNN/AE weights were located and strictly loaded on CPU. Do not
rename MNIST as StyleMNIST without resolving task identity. Discovery and prior
results are recorded in results/008_stylemnist_asset_inventory.md; new external
audit is /home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_asset_audit_01/.
No existing outputs were changed and no full experiment was launched.

## 2026-09-16 — Task 001 identity resolved and protocol frozen

User confirmed four GPUs (1,2,3,5) and that StyleMNIST means existing MNIST
with style shifts. Use the paired saved 4→9 SmallCNN / StrongConvAutoencoder64
weights from mnist_roar_latent_calibrated/payload.pt without retraining. The
new config experiments/configs/stylemnist_grace_v1.json fixes pixel-space
training-data radius 2, full eight-epoch AdamW updates from f0, and existing
latent Euclidean W2 geometry with no current-validity repair. Reuse the raw
solver and historical differentiable AdamW attack surrogate. CNN search uses
rotating minibatches and five-step unrolls; complete eight-epoch hypergradient
refinement is not claimed. Every retained radial candidate gets a full update.
Rotation/translation/noise definitions are historical; the severity grid and
aligned-update replacement for scratch future fitting are explicitly new.
All future evaluation follows frozen shared-bank candidate selection.

## 2026-09-16 — Task 001 completed and independently reviewed

Run /home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1 completed
68 jobs on four A100s, with 24 full-update bank models and 45 future models.
372 retained controls were cross-evaluated before future evaluation. Nominal
future validity: Base .75; beta1 .9791667; beta1.05/1.10 1.0; beta1.20 .9913194.
The SNS-style baseline also reaches 1.0, and exceeds GRACE at severity2.
No universal dominance or guaranteed solver monotonicity is claimed.
Independent checks passed checkpoint/perturbation hashes, data bounds, exact
geometry within tolerance, bank risks, all future validity values, IDs and freeze
ordering. Beta1 geometry exceeds the exact bound by 5.36242e-5, inside the
predeclared relative 1e-6 tolerance. Base encode/decode validity drops from 100%
to12.5%; retain this material representation limitation. Initial strict decoder
reconstruction equality failed: repeated GPU ConvTranspose calls differ by less
than1e-4 per pixel; no future decisions changed when all selected latents were
redecoded. The saved frozen images are the evaluation artifact. This numerical
limitation is recorded, not hidden by changing experiment outputs.
Tables/report/review are in results/009_stylemnist_grace_v1; PDFs in
figures/stylemnist_grace_v1. Historical assets and manuscript claims are unchanged.

## 2026-09-16 — Galaxy task: original multiclass asset/protocol decision

Use the actual submitted LeapFactual VGG20 epoch47 checkpoint, WAE64 version8
last checkpoint and original conditional flow finetune weights. Do not substitute
the later binary 1→2 latent-MLP prototype. Audit/inventory: results/010_galaxy_*.
Run ten target-label tasks with seed-fixed16 calibration/test factuals per target.
Calibration selects r in{2,4} and beta in{1,1.05,1.10,1.20} separately per target.
Original progressive Base and robust-continuation LeapFactual generators are
preserved. Exposure/noise stress and fixed two-epoch full-VGG updates are explicitly
new; optimizer/lr/scheduler follow source, including its effective AdamW .01
weight decay. One-step multiclass minibatch attack gradients are an approximation;
accepted candidates receive full updates on all3181 training examples. Search can
modify128 preselected images per restart; this limits the bank, not the WRO domain
claim. Every geometry leg is latent Euclidean W2 with no current-validity repair.
Use all eight GPUs under12.8GiB allocator caps, without stopping other workloads.

## 2026-09-16: user-requested Galaxy pair scope

The completed ten-target Galaxy run remains immutable at `/home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_full_v1` (75 jobs; independent checks passed). User narrowed the follow-up to Merging–Round Smooth and Round Smooth–Cigar Shaped Smooth, interpreted in both directions and explicitly described to the user. New pair cohorts use only the stated source class; retain the original ten-way classifier and all-class update data. GRACE starts at Base CFs, not the additional robust LeapFactual refinement, as confirmed by the user. New pair-specific attack banks and calibration selections are required. Calibration/future update checkpoints are reused after hash and protocol checks to avoid rerunning identical historical outputs. These shifts have already been evaluated in the ten-target experiment; pair results are not a fresh unseen-shift benchmark. Two disjoint four-GPU controllers collectively use GPUs 0–7. Face task remains queued after the pair runs.

Galaxy pairs completed with independent checks passing (43 jobs each). Nominal macro Future Validity: Merging–Round Base0.40625, selected GRACE0.6328125, LeapFactual0.703125; Round–Cigar Base0.5078125, selected GRACE0.7421875, LeapFactual0.6875. For Cigar source, zero calibration-complement images required reserving16 of33 old test images for calibration and16 disjoint images for final tests; fixed seeded selection, amendment and initial failure recorded. Face audit starts only after both pair runs finish.

## 2026-09-16 FFHQ protocol

Original face assets are generated StyleGAN3 FFHQ-style images, W512, CLIP prompts `smiling face` versus `face`, target0. No real-image encoder, supervised smile checkpoint/training split, or identity metric was found. Reuse original flow Base10 iterations(.8/.8), report extra robust5 iterations(.8/.83) separately. New generated pseudo-label dataset and a fixed projection-only CLIP Update are frozen before running; this limitation is explicit rather than silently replacing f0 with a newly trained classifier. New brightness/noise/blur shifted-image update suite, r1/2, four betas, calibration-only selection. Original reference-operator smoke exceeded12.8GB cap; original fused operators compiled in isolated CUDA environment and checkpointed gradients passed with2.44GB peak. Eight-GPU launcher and reproducibility configs are ready; no outcomes informed protocol choice.

FFHQ completed:75 jobs across GPUs0–7; independent image, future-score, bank-risk, selection and geometry checks passed. Selected r1/beta1, nominal FV1.0 versus Base0.45833 and LeapFactual0.83333; pixel224L2 92.91 versus38.74 and54.64. All betas selected the same geodesic midpoint population; no benefit from extra beta is demonstrated. Fixed visual grid shows substantial appearance changes; identity preservation is not established and no identity metric was invented. Strongest blur reduces selected FV to0.875. Full interpretation and fixed-update limitations in `results/014_ffhq_grace_v1/interpretation.md`. Construction538.55s,0.584 workerGPUhours. All requested Galaxy pair and face outputs are stored; previous runs remain preserved.

## 2026-09-16: supplementary FFHQ low-change selection

At the user’s request, `figures/ffhq_grace_v1/ffhq_qualitative_1.pdf` shows the three lowest factual-to-GRACE native1024 SqueezeNet LPIPS cases among the eight frozen test examples (indices 1, 3, 7; LPIPS 0.214685, 0.258215, 0.373934). Selection is explicitly post-hoc and does not use future validity. Original results and qualitative figure are preserved. All four beta outputs are identical, verified from saved W codes. The supplementary figure does not represent the full cohort or establish identity preservation. Full rankings, source hashes and reproduction command are in `results/014_ffhq_grace_v1/qualitative_1_selection.csv` and `.json`. The saved W codes were rendered with the frozen original generator; no experiments were rerun. The one-page PDF and PNG were visually checked.

## 2026-09-17: FFHQ hard factual-content bottleneck

Task002 explicitly requests the same population-level GRACE objective with a domain-specific conditional realization. Original FFHQ code has no task/content factorization, image encoder or face-identity model. Use the prescribed fallback: uncentered SVD of original smile-probability gradients on all256 training W codes; smallest dimension explaining95% energy, frozen before new test evaluation. This yields29 task dimensions and483 complementary content dimensions. Preserve each ORIGINAL factual's complementary code; optimize task coordinates only. No soft identity penalty, current-validity branch or manuscript-theory change. Calibration64 probe accuracy: task76.5625%, content56.25%; content is not asserted classification-independent.

New raw run: `/home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1`. All eight GPUs are used in capped lanes. Historical Base/factuals/training, original models and sixteen attack/thirty future models are reused by hash; never modified. ArcFace buffalo_l is a new independent evaluation component, using official detector, five-landmark alignment, largest detected face, cosine similarity, detection accounting and source hashes. Save original Base and conditional-Base diagnostics separately. Fix all eight test IDs for qualitative panels before evaluation. Calibration selected r2/beta1.05, so the full-latent comparator at r2 is newly computed to match exactly; original r1 output remains intact. This is a matched follow-up on already evaluated future-model scenarios, not an unseen-shift claim.

FFHQ identity task completed:56 jobs across GPUs0–7, five focused tests and independent source/geometry/content/prediction checks passed. Initial640px ArcFace detector missed5/8 factual faces. A factual-only320px scale audit (same0.5 threshold) detected all8; corrected identity measurements and alignment checks are preserved separately in `identity_detection_audit/`. Final eight-pair means: Base FV0.45833/identity0.82724; full GRACE FV1.0/identity0.27228; bottleneck GRACE FV1.0/identity0.19832. Bottleneck also has higher LPIPS0.48935 versus0.38203 and cost128.82 versus92.91. All betas return the same target-task endpoint. Despite enforcing483 immutable content coordinates and exact reconstruction, this fallback fails the stated identity-preservation success criterion. Final authoritative report: `results/015_ffhq_identity_v1/final_report.md`; no positive identity claim or manuscript change.

## 2026-09-17: calibration-only local smile-editor screen

User approved the proposed calibration-only editor comparison before another GRACE sweep. Compare1/4/8 post-affine StyleGAN3 channels against a one-direction W baseline; leave other styles and the factual input transform fixed. Discover channels on32 training faces using smile-score gradient sensitivity normalized by a three-probe outside-region image-Jacobian RMS estimate. Use all256 training codes for style standard deviations. Select channels only from layers3–11 with at least75% gradient-sign agreement. The fixed ellipse is a coarse mouth/cheek proxy, not semantic segmentation.

Freeze all gates before calibration: smile probability≥0.5 and gain≥0.05, ArcFace≥0.8, independent FaceNet≥0.8, LPIPS≤0.15, outside RMS≤0.04, at least6/8 successful factuals plus qualitative review. These are engineering gates rather than validated same-person thresholds. No final test or future model access, no gate relaxation after failures. Only a passing localized editor may proceed to GRACE. Future hard identity constraints, if used, explicitly alter the domain-specific feasible set and must report infeasibility.

Raw screen v1 preserved successful discovery; evaluation failed on the legacy FaceNet checkpoint's mmap format. v2 also exposed its retained classification-head keys. Strict CPU load and512-embedding smoke passed after loading the full head and switching to embedding mode. Final evaluation runs in `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3`, reusing v1 discovery read-only. Neither failed evaluation produced calibration results. Original run markers/artifacts remain untouched.

Localized editor screen complete on all8 GPUs,192 calibration candidates. Style1/4/8 passed0/8 each; W-direction baseline1/8, below the predeclared6/8 gate. Localized ArcFace means0.958–0.975 and FaceNet0.980–0.987 indicate small-edit preservation, but mean smile scores remain0.269–0.294, below0.5. Visual review shows mostly stable appearance and weak/inconsistent smile edits; factual smile/pseudo-label mismatches persist. No GRACE follow-up started and final test/future models remained untouched. Seven focused tests and independent gate/identity/zero-edit checks passed. Final report: `results/016_ffhq_smile_editor_v1/final_report.md`.

## 2026-09-17: stronger local controls and independent coefficients

User approved the proposed next experiment. Predeclare alpha6/8/12/16 for the same frozen training-selected1/4/8 channel controls. Preserve all original quality and smile gates. If scalar controls fail6/8, permit independent4/8 standardized coordinates with40 normalized-gradient steps, bounded±16 and true measured gates on every accepted update. This is a sequential calibration study following observed weak edits, not independent confirmation. No FFHQ final test or future-model access.

Independent supervised diagnostic: AiresPucrs/LeNNon-Smile-Detector revision e34d94965363927f60dea5a5b76770a8966048c5, model-card preprocessing RGB100×100[0,1]. Released checkpoint has10 logits; retain original argmax>0 convention and expose extra-class predictions rather than silently replacing the head. Audited weights-only conversion and strict state load passed. It classified31/32 first annotated eurecom-ds/celeba test examples correctly, predicting onlyclasses0/1. Training overlap is unknown; this is a small source-domain sanity audit, not validated FFHQ accuracy. The diagnostic does not replace original CLIP or affect optimizer selection.

Extended raw run: `/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v2/20260917_extended_v1`. Stronger scalar controls pass4/8 (style1),1/8 (style4),3/8 (style8), so the predeclared independent-coordinate phase proceeds on eight GPUs. Preserve all accepted/rejected candidates, source hashes and hard-gate evidence.

Extended editor screen complete: stronger scalar controls pass4/8,1/8,3/8; independent4 and independent8 each pass6/8 original numerical gates. Independent4 is the calibration candidate (ArcFace0.8585, FaceNet0.9255, LPIPS0.0300; all preservation limits unchanged). Independent supervised smiles corroborate only2/6 selected original successes for independent4 and1/6 for independent8. Qualitative review shows meaningful changes in some cases but inconsistent source/target smile semantics and disagreement of both classifiers. Do not launch GRACE or claim validated smile recourse yet. Final FFHQ test/future models remain untouched. Nine focused tests and independent candidate/embedding/gate checks passed. Final report: `results/017_ffhq_smile_editor_v2/final_report.md`.


## 2026-09-17: FFHQ smile-label audit

Completed automated audit of 16 saved generated calibration images and 32 reused source-domain examples. All eight CLIP image-gradient checks pass (maximum relative discrepancy 0.7455%). Prompt wording changes positive-edit counts from 6/8 to 2/8 or 5/8; the third wording also marks 4/8 factuals smiling. Do not select a replacement classifier from this small diagnostic or use model/assistant judgments as human labels. Keep the independent4 editor and identity gates fixed. Human validation is pending; shuffled annotation HTML/PDF v2 are ready. No final-test access, GRACE run, old-output modification, or manuscript change. Report: results/018_ffhq_smile_label_audit/report.md. State records automated completion separately from pending domain validation.


## 2026-09-17: GENKI-4K supervised head on FFHQ encoder

User authorized GENKI-4K on the FFHQ infrastructure. Trained a supervised two-class linear head on frozen CLIP image embeddings using official GENKI smile annotations. Selected L2=0.0003 on validation only; held-out GENKI accuracy 94.8505%, balanced accuracy 94.6528%. Exact duplicate groups isolated; subject identities and pretraining overlap unknown. Eight GPUs, 18 completed workers; the evaluation worker saved results but failed three CUDA float32 finite-difference checks. A separate CPU float64 audit passes all eight at the same tolerance and two step sizes; original outputs preserved. Three focused tests pass. Differentiable FFHQ adapter retains smile at output index0. Old generator, editor and identity limits preserved. No inversion of real GENKI photographs, new CFs, new bank, or GRACE result. Generated-face transfer is diagnostic without human labels. Report: results/019_genki_ffhq_classifier/report.md. Raw: /home/xu.zhao/robust_ce_results/genki_ffhq_classifier_v1/20260917_supervised_v1.


## 2026-09-17: GENKI-head GRACE with preservation

Completed new nominal Base CFs, compatible GENKI pixel-attack banks and future updates, and actual population-risk GRACE in the fixed four-channel realization. Selected r=2.0, beta=1.2; future validity Base=0.3167, selected GRACE=0.4500. All 40 returned Base/beta faces pass ArcFace/FaceNet>=.8, LPIPS<=.15, outside RMS<=.04; this does not establish human smile/identity correctness. Exact task-space W2 geometry and max_model mean_CF hinge retained; no nominal-validity repair. Seventy-nine jobs, eight GPUs, seven tests; old outputs preserved. Report: results/020_genki_grace_identity/report.md. Raw: /home/xu.zhao/robust_ce_results/genki_grace_identity_v1/20260917_full_v1.


## Larger-beta GENKI GRACE extension
User chose beta 1.5, 2, 3 with Base and all preservation settings fixed. Existing outputs remain unchanged. All eight cases and uniform saved-image metrics are in results/021_genki_grace_beta_extension/report.md; figures/genki_grace_beta_extension/qualitative.pdf. This is exploratory reuse of prior test faces and future heads.
Larger betas selected byte-identical images to beta=1.2; future validity stayed 45%. All 105 competing screened candidates failed ArcFace on cases 3 and 8; the original winner was the sole population passing all gates. This identifies the rejection reason for this finite search, not a global impossibility.


## Identity-feasible GENKI GRACE block search
User authorized preservation-aware search. Cases 3 and 8 remain fixed while the other six receive feasible block updates, followed by global bank-based coordinate selection. Same radius, beta 3, Base, bank and quality thresholds. Results: results/022_genki_preservation_search/report.md; figures/genki_preservation_search/qualitative.pdf. Exploratory reuse of existing faces and future heads.
Outcome: future validity 45% to 50%, nominal validity 4/8 to 5/8, all original preservation gates passed. Only cases 5 and 7 changed. Worst-case reliability remains unresolved; this is a modest improvement within metric thresholds, not a human identity guarantee.


## GENKI expanded editor and target ablation
User approved 8/16 channels and target 0.7. Completed matched 4/8/16 by 0.5/0.7 with training-only nested channel discovery, common 16-D geometry and fixed preservation gates. All eight faces remain in every arm; no future-based arm selection. Report: results/023_genki_expanded_editor/report.md. Figures: figures/genki_expanded_editor/qualitative.pdf.
Outcome: matched target0.5 future validity = 50%, 51.67%, 60% for 4/8/16 channels; target0.7 = 58.33%, 62.5%, 75%. All quality gates passed. The 16-channel/0.7 arm has six cases valid across all 30 previously seen heads, with cases 3 and 8 still failing. Visual edits are larger, but case 7 has a pink skin-tone shift; classifier gains are not proof of clean smile semantics. Do not strengthen manuscript claims from these exploratory results alone.


## 2026-09-18 — Task 005: fixed-Q HELOC realization comparison

Completed all four beta comparisons using newest seed-1 raw Q, frozen 5,927-model bank and the existing 45 aligned/45 scratch future models. Archived OLD outputs reproduce to 4.44e-16 maximum error. Newest Q had no completed OLD deployment, so replay exact archived weighted path construction and calibration policy on newest Q. No selected beta exists; do not invent one.

The task's equal-cardinality premise is false: each split has 128 factuals/Base CFs, Q has 3,336 particles. Preserve Q using exact weighted Base-to-Q OT; explicitly label this a weighted diagnostic, not one deterministic CF per factual. Do not resample, barycentrically collapse, or retrain Q. Base-directed first-crossing correction restores current validity while leaving valid atoms unchanged.

At beta=1.10, corrected direct transport raises current validity from 97.27% to 100%, future validity from 87.51% to 88.29%, and reduces cost from 4.260 to 4.168. However, correction moves affected atoms about 68% toward Base on average and W2(deployment,Q)=0.696. Other betas show cost tradeoffs; no universal winner is claimed. Training geometry budgets do not transfer to the test population for beta>1. All conclusions concern weighted distributions and retrospective evaluation, not a deterministic deployment guarantee.

Report: `results/heloc_realization_comparison/report.md`; six plots: `figures/heloc_realization_comparison/comparison.pdf`. Raw run: `/home/xu.zhao/robust_ce_results/iclr_runs/heloc_realization_comparison/20260918_v1`. Nine focused tests and independent output/marginal/correction checks pass; original artifacts remain unchanged. Paper discrepancies are documented without altering manuscript claims.


## 2026-09-24 — Revised PDF Method is authoritative

User explicitly designated the Method section of `paper/iclr_robust_ce-62.pdf` as the sole experimental rule. Read Section 3 and its referenced Algorithm 1. PDF SHA256: `57d881807f2c85a501b87a8af28fc80a6ede8077ce90c624b3e313cb26a2d560`. This supersedes conflicting old task/code specifications, while preserving previous runs as historical results.

The numerical objective is nominal population hinge after the prescribed T-step update plus r times the dual norm of training-distribution sensitivity obtained by differentiating through that update. It replaces the previous finite-bank maximization as the optimization objective. Optimize M=n equal-mass support points under the sum-of-W2 budget with an augmented-Lagrangian primal–dual scheme; check final unregularized OT feasibility. Deploy using minimum-cost one-to-one Base-to-Q assignment. Select r and beta using calibration Future Validity, then freeze them before final future-shift evaluation. For D=0 calibrate the absolute budget. No current-validity constraint or post-hoc correction is specified in Section 3; do not silently add the previous repair.

The completed task005 weighted, unequal-cardinality bank-based comparisons do not evaluate this revised method. No new experiments launched by this specification review. The PDF's experimental prose still mentions a shared model bank, and Algorithm 1 contains an unresolved equation reference; Section 3 remains authoritative under the user's instruction.


## 2026-09-24 — New-method MNIST latent pilot, eight GPUs

Created and executed `tasks/011_mnist_update_aware.md` using the revised PDF Section 3. User correction requires latent-space CF optimization and eight GPUs; data-side sensitivity stays in training-image pixel space. Implemented complete three-step gradient-descent differentiation on a predeclared 256-image empirical distribution, equal-mass M=n=64 supports, augmented-Lagrangian geometry, and one-to-one latent assignment. No finite-bank objective or validity correction. All eight GPUs used for calibration jobs; existing classifier/AE and test Base reused; old outputs preserved.

Negative pilot: calibration chooses r=0,beta=1.05. Test Future Validity 51.91%, current validity 53.125%, pixel L2 cost5.943, versus original Base FV96.01%, CV100%, cost5.576. Decoded Base has CV12.5%, FV21.53%; the encode/decode gap is severe despite visually similar digits. Positive-r calibration results are worse. All16 solves exhausted80 iterations; beta1 fails exact geometry in all three calibration runs and test. Do not claim convergence, a successful robust setting, or paper-level improvement. f0 and nominal-update training-subset accuracy remain100%; no held-out model accuracy claim.

Six focused tests pass; actual CNN+decoder outer finite differences agree within1%; original hashes and permutation checks pass. The evaluation-only finalizer fixes retained-graph memory without changing optimized outputs. Report `results/024_mnist_update_aware/report.md`; qualitative `figures/mnist_update_aware/qualitative.pdf`; raw `/home/xu.zhao/robust_ce_results/iclr_runs/mnist_update_aware/20260924_latent_v2`. Recommend next verifying a generator that returns valid decoded images with their original latent codes, and solving to convergence on calibration data. Do not silently apply this change to existing results.


## 2026-09-24 — Smooth hinge clarification and retained latent Base codes

The user clarified that the revised method should use smooth hinge. Adopt t*softplus((0.5-f)/t), t=0.05 in probability space, for both nominal loss and update-aware sensitivity. This explicit user correction overrides the numerical hard-hinge interpretation of the PDF. Keep r=0 as a nominal-only control and choose r using calibration FV, not a requirement to obtain a positive radius.

Found the Base bookkeeping problem: generate_near_boundary_ce decoded valid images but discarded their latent codes; re-encoding those images changed the latent Base population. CEResult now retains its actual codes (including fallback). Calibration replay matches original Base exactly. Test replay differs by at most0.0005848/pixel, so record it as a numerically regenerated Base with its codes and retain Historical-Base separately. Both corrected Base populations are100% current-valid; no GRACE validity repair is used.

The strict recovery attempt and subsequent allocator-cap failure remain preserved. An intermediate smooth run used smooth preparation but hard-hinge optimizer defaults; it is invalidated in results/025_mnist_latent_base_fix/SUPERSEDED.json. The corrected run explicitly passes temperature everywhere and checks each starting objective against the independently audited Base smooth objective. Code changes release unused graphs before final metrics. Active corrected run: /home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/20260924_smooth_v2; task012. Results are pending; do not reuse intermediate figures as evidence.


## 2026-09-24 — Completed corrected smooth-hinge MNIST run

Completed /home/xu.zhao/robust_ce_results/iclr_runs/mnist_latent_base_fix/20260924_smooth_v2 on all eight GPUs. Actual generator codes preserve100% Base current validity; smooth loss is explicit in every objective call and initial objective checks pass. Fourteen tests, finite-difference checks, all24 assignment/geometry audits and input hashes pass. Nine beta>1 test runs meet the stopping criterion; three beta1 runs fail final geometry and are not deployed.

All beta>=1.05/radius configurations attain100% calibration FV at severity1, so the existing cost tie-break selects r0,beta1.05. Selected test FV100%, CV100%, cost5.6046, versus Base FV96.01%, cost5.5755. Positive radius has a real diagnostic benefit under stronger shifts: severity2 FV94.62%/99.83%/100% for r0/.25/.5 at beta1.05; costs5.6046/5.6344/5.6540. Do not reinterpret the positive-radius diagnostic as the calibration-selected method. The saturated calibration criterion, not a broken smooth objective, explains the remaining r0 selection.

Report results/026_mnist_smooth_hinge/report.md; figures/mnist_smooth_hinge/qualitative.pdf and comparison.pdf. Prior results025 are explicitly invalidated for their mixed-loss implementation. Recommend a future predeclared stronger calibration suite and tighter-budget solver work; no post-hoc selection change was made. Loss smoothing is now an explicit user-authorized exception to the original PDF hard-hinge notation.


## 2026-09-24 — HELOC smooth-hinge task013 protocol

User requests HELOC execution/results and explicitly eight GPUs. Created tasks/013_heloc_smooth_hinge.md before execution. Reuse frozen seed1 f0, preprocessing, all6000 training examples,128 Base CFs/calibration-test factuals, and all3336 favorable references. New Q has128 uniform supports and exact one-to-one deployment; do not reuse old3336-support bank-trained Q.

Use smooth hinge temperature0.05, full10-step SGD update (lr0.01, BCE) with complete update-aware differentiation, and new future models generated with that same operator. This is distinct from the historical8-epoch AdamW operator. Radius ratios0/.25/.5/1/3 and beta1/1.05/1.1/1.2 are predeclared; calibrate on all45 scenarios including all severities, not just mild shifts. No current-validity repair; historical continuous all-features-mutable representation remains explicit. All eight GPUs launched; actual gradient finite differences agree within0.02%.90 future models saved. Run /home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/20260924_v1 is in progress; no completed numerical conclusion yet.

## 2026-09-24 — HELOC smooth-hinge task013 completed

Run `/home/xu.zhao/robust_ce_results/iclr_runs/heloc_smooth_hinge/20260924_v1`
completed40 support solves and90 new ten-step future models using physical GPUs0–7.
Report: `results/027_heloc_smooth_hinge/report.md`; plots: `figures/heloc_smooth_hinge/`.
Calibration alone selected r/b_ref=3 (r8.18883183), beta1.1; test FV100%, CV100%,
cost1.6890, sensitivity0.00280652. Base FV82.9861%, cost1.22785; nominal-only
beta1.1 FV99.9826%, cost1.51624. Positive-radius incremental threshold gain is
only0.0174pp at11.39% greater cost than the nominal control; sensitivity drops65.32%.
Smaller positive radii also reach100% test FV; do not revise frozen selection
from this observation. Nominal-only beta1.1 is iteration-limited.

Both calibration/test have15/20 feasible final iterates and13/20 satisfying
successive-objective stopping. All ten beta1 solves fail geometry and are excluded
from deployment. Nine unit tests pass; independent report checks all40 assignments,
geometry totals, objective arithmetic, frozen selection and original/model hashes.
Plots visually checked. No old outputs altered, no validity repair.

Treat this as a retrospective single-seed short-update study, not certified
robustness/actionability or robustness to arbitrary retraining. Base already has
out-of-training-range coordinates in85.94% cal/85.16% test CFs; selected test also
85.16%. Duplicate future distributions and non-monotonic measured difficulty
across severity controls limit generalization. Keep these qualifications with
the result; no manuscript scientific claims changed.

## 2026-09-24 — Predeclare MNIST near-zero radius follow-up

Implement user's16-point grid on MNIST, with absolute radius units (not HELOC
r/b_ref). Freeze the old classifier, Base codes, data and complete T=3 operator.
Broaden calibration to severities .5/1/1.5/2 because previous severity1 FV
saturated; generate fresh stochastic realizations using cal31000/test41000.
Deterministic translations remain duplicates. Select by calibration FV/cost only;
all16 test settings are diagnostics, not a second selection. Task014 describes
32 new solves on eight GPUs and required FV_cal/Cost_cal/sensitivity/Phi_r tables.

## 2026-09-24 — Task014 results and next-step interpretation

Completed32 MNIST solves and72 future checkpoints on8 GPUs; report028 and
figures/mnist_small_radius. Calibration selects absolute r=.25/beta1.02 with
99.9566% FV; test100%, current100%, pixel cost5.6344. At beta1.02,
r0/.05/.10/.25 gives calibration FV98.6979/99.6528/99.7830/99.9566%,
sensitivity .036140/.006905/.004225/.002509, cost5.9665/5.9803/5.9850/5.9931.
Test FV97.0486/99.5660/99.8264/100%, costs5.6046/5.6193/5.6248/5.6344.
The first interval captures86.93% of measured0→.25 sensitivity reduction.
This supports a steep near-zero benefit, not a mathematical discontinuity.
Keep the predeclared FV-maximizing choice; do not post-hoc replace it by r=.05.

Cal13/16 feasible,12 meet stopping; test12/16 feasible,12 meet stopping.
All beta1 test cells fail. Ten tests and32 independent checks pass. Base
byte-identical. Ten of12 repeated Q cells bitwise-identical; two max coordinate
differences .00356293 and .00141287 are disclosed, so do not promise bitwise
solver reproducibility across GPU scheduling. No validity correction.

Recommend independent classifier/data-seed replication at beta1.02,r0/.05/.25
before another broad grid. This recommendation is not executed in task014.
Selection uses fresh stochastic calibration scenarios; deterministic translations
still overlap, and old assets remain retrospective. No manuscript claims changed.

## 2026-09-24 — Matched weaker-classifier task015 result

User requested a weaker classifier and potentially clearer visuals. Predeclared
validation-only checkpoint selection before CF generation:48 candidates from
seeds150–157 and steps4/8/16/32/64/128,128 balanced training examples,512
validation examples disjoint from update/P1. Selected seed153/step16 at89.84%
validation accuracy; official test88.70% versus99.70% strong. Both arms use same
factual IDs, common P1, AE,256-image update distribution and beta1.02/r0,.05,.1,.25.
New Base generated for each current classifier; original runs preserved.

16 solves and144 future checkpoints completed on8 GPUs. All final geometries
feasible;10 tests/16 independent assignment/geometry/decode/metric checks passed.
Strong selects r.25, FV100% versus nominal97.87%/Base85.55%; weak selects r0,
all FV100% including Base. Mean direct image L2 betweenr0/r.25 is.138931 strong
and.000012962 weak. Weak Base cost6.4254 exceeds strong5.4035, so any larger
initial image change is Base-generation behavior, not GRACE refinement.

Crucial qualification: all weak solves stop after2 objective evaluations/1
update under the fixed absolute1e-6 objective-change tolerance at this scale.
Sensitivity is small but notzero, and global/stationary optimality is not
established. This negative visual result does not prove a fully optimized
weak model intrinsically has less visual movement. Recommend a matched solver
tolerance/minimum-iteration audit before further classifier-quality claims;
do not select another weak checkpoint for prettier images. No additional run
launched. Report029 and fixed/all64 figures save the honest negative result.

## 2026-09-24 — Plan task016: lower-margin Base, fixed strong classifier

Saved tasks/016_mnist_low_margin_base.md at user request; no task016 run launched.
Operationalize weaker Base as reduced current-classifier margin while retaining
validity. Keep task015 strong classifier and matched inputs fixed; compare original
Base, precise.52, .505 and.5001 on the same latent path. The precise.52 arm controls
for discrete generator overshoot. Record unbracketed/encoded-start-positive cases
as unchanged rather than dropping factuals. Lower margin is a hypothesis about
future robustness, not a guarantee or an instruction to degrade images.

Plan32 solves on8 GPUs, shared72 new future scenarios, r0/.05/.10/.25 and beta1.02.
Predeclare800 fixed support updates for all arms to avoid task015's one-step
early-stopping artifact; report original stopping diagnostics without claiming
convergence. Separate Base-to-nominal and nominal-to-robust gains, and preserve
all prior outputs. State records this as planned, not completed experimental work.

## 2026-09-24 — Task016 completed: controlled lower-margin Base

Run `.../iclr_runs/mnist_low_margin_base/20260924_v1`, report030 and
figures/mnist_low_margin_base.8 GPUs,32 fixed800-update support solves and72
paired future models.15 tests/32 independent checks passed; all final geometry
constraints feasible. Boundary search attains targets for64/64 cal and63/64 test
cases per refined arm, preserving one original positive-start fallback. Original
Base code/image pairs remain exact; previous outputs unchanged.

Test Base FV original84.2448%, precise.52 79.3403%, .505 78.0816%, .5001 77.3872%.
All current-valid. Calibration selects r.25 for every arm; all selected test FV100%.
Near-boundary nominal-only FV97.9167%, separating20.5295pp nominal gain and
2.0833pp robust increment. Costs Base5.3584/nominal5.4186/selected5.4718.
Base sensitivity original1.0636 versus near-boundary3.6303; near-boundary nominal
.010388 and r.25 .0005783. Model/update protocol remains strong f0, same decoder,
256 training images and complete T=3 SGD, with new cal71000/test81000 realizations.

At r.25, direct image L2 from Base increases original.283568→near-boundary.384345,
while r.25-minus-r0 increases.137360→.172804. Fixed example inspection still shows
subtle stroke/intensity edits, not dramatic semantic changes. Refined.52 has
similar or greater movement, so do not claim monotonic visual effects of margin.
Most recovery remains nominal.800 fixed updates do not establish convergence.
Keep single-checkpoint/retrospective/duplicate-scenario/no semantic-ground-truth
limits. No manuscript claims changed. Task specification remains the hashed
pre-run source; completion status is recorded here and in state.yaml.

## 2026-09-24 — task017 Galaxy update-aware directed pairs

User requested reuse of old Galaxy infrastructure/classifiers for Merging→Round
and Round→Cigar, with a task written before execution. Task017 specifies the new
PDF/smooth-hinge objective, not the historical finite bank. Original ten-class
VGG and WAE64, 16 calibration and16 test cases per direction, and raw nominal Base
codes are frozen. Retain all cases, including initially invalid CFs and one
Merging→Round test factual already target-positive.

To make full differentiation practical, prescribe three lr1e-4 SGD steps of the
last4096→10 layer on40 class-balanced training examples (four per class), with
all earlier weights/BN fixed. This is a head-adaptation pilot, explicitly not a
proxy claim for the old two-epoch full-network update. Pixel sensitivity remains
exact for this operator: frozen-feature Jacobian pullback norms and their adjoint
derivatives are streamed in microbatches. A nonlinear toy model validates the
full-update outer gradient; both real Galaxy finite-difference errors are<0.9%.

Original raw codes replay exactly under original TF32/batch4 settings. Current
float32 decoding differs by max0.002067 pixel; retain original images as
Historical_Base and use exact current Dec(raw codes) as optimization Base. No
re-encoding or current-validity gate. Preserve failed strict-replay audit_v1/v2.
The primary run is20260924_v1, eight GPUs, with GPU4 assigned lighter future-head
jobs due existing memory load. Grid r={0,.25,1,4}, beta={1.02,1.1};160 updates,
primal lr5/rho1/gradient-norm cap.2, best geometry-feasible iterate. Future stress
is brightness/noise, not a claimed inner supremum or common r-ball restriction.
Canonical class/filename checks show no overlap between update data, references,
and factual splits. Sixteen relevant unit tests (3 new,7 OT,6 update-aware) pass.

Task017 completion:32 solves/32 new head models, all saved and final iterates
geometry-feasible;16 tests/32 independent audits pass. Eight GPUs used,23.6min
construction wall time. Calibration choices frozen before test: Merging→Round
r0/beta1.02; Round→Cigar r.25/beta1.02. Both yield100% test FV, and nominal-only
also reaches100% for both. Hence no incremental held-out FV gain from sensitivity
in this pilot. Round→Cigar calibration improves93.75%→100% with r.25, while test
sensitivity falls.001448→.000141 without a further FV increase. r4 at beta1.02
is worse (93.75%/68.75% test FV); smaller sensitivity can accompany confidently
invalid cases. Do not retrofit selection or describe monotonic radius benefits.

Numerical baseline distinction: Historical_Base Merging→Round test FV74.61%,
float32 exact-code replay69.92%; Round→Cigar50% for both. Small replay changes can
move boundary cases and must be reported. All-case PDFs preserve visible inherited
Base artifacts, including Round→Cigar ID3. Current/future classifier validity does
not establish semantic morphology or identity preservation. Final report:
`results/031_galaxy_update_aware/report.md`; state.yaml marked complete.

### Task017 Round→Cigar largest-difference qualitative follow-up

User requested r.25−r0 and r0−Base heatmaps for cases with the largest differences.
Script071 ranks all16 frozen test cases by full RGB pixel L2 separately for each
contrast, at beta1.02. Top4 IDs: [3,0,14,7] and [3,7,11,12], respectively. The
new two-page PDF is `figures/galaxy_update_aware/round_cigar_largest_differences_r025.pdf`.
Both heatmaps appear on the right and share one absolute color scale across pages.
No retraining, output changes, or configuration selection. Original input hashes
verified unchanged. The largest case ID3 contains inherited Base artifacts; figures
are explicitly post-hoc illustrative selections rather than representative quality.

## 2026-09-24 — FFHQ original classifier, full-W update-aware task018

User explicitly selected original CLIP after declining fixed-face-content
constraints. Use original StyleGAN3/CLIP and actual nominal Base W512 codes;
GENKI supervision and the constrained style editor are not part of this run.
Task018 was written before execution. Full W512 Euclidean OT, one-to-one
assignment, no identity/background or current-validity gate. Three complete
SGD updates of original visual projection, lr1e-4,32 balanced generated images.
Smooth hinge.05 and exact empirical-RMS training-pixel sensitivity follow the
revised method. Query feature VJPs stream the full generator one face at a time.

Grid r0/.25/1/4 and beta1.02/1.1,160 fixed updates. New24 calibration and24 test
projection models use brightness/noise/blur shifts; no old bank/checkpoints.
Original audit_v1 passes18.1% at epsilon.01; audit_v2 additionally verifies
0.824% at epsilon.005 (epsilon.002 gives6.71%, reflecting numerical/step-size
sensitivity). Neither audit is an optimization run. Twelve tests pass (2 new
projection,3 pixel pullback,7 OT). Measured memory2.35GiB allows eight-way grid
execution even on GPU4 with lower free memory; cap allocations at6.5% of80GiB.

Raw primary run: /home/xu.zhao/robust_ce_results/iclr_runs/ffhq_update_aware/20260924_v1.
Reporter073 will independently reconstruct and score outputs, make all-eight and
post-hoc top-difference heatmaps, and update completion state. Base replay errors
are retained in diagnostics; original Base/image/checkpoint files are immutable.

### Task018 completion

All16 grid solves and48 new future projection models completed on eight GPUs;
experiment wall time54.1min. Twelve focused tests and16 independent checks pass.
Primary gradient audit errors0.597% at epsilon.005 and0.845% at epsilon.002.
Input hashes, exact OT assignments, decoder replay, costs and per-model scores
were independently verified. All saved outputs pass the declared geometry
tolerance. The cal r0/beta1.02 final iterate fails geometry; the predeclared
best-feasible rule retains iteration155. Six of16 solves retain iteration160.

Calibration FV is100% for all settings; cost tie-break chooses r0/beta1.02 before
test. Test Base/Historical_Base FV82.2917%; all refined settings100%. At beta1.02,
Base/r0/r.25 costs38.7441/47.5033/48.0834 and sensitivity
.452216/.012014/.002436. Thus nominal refinement yields17.7083 percentage points,
with zero measured additional FV gain from the sensitivity term. This eight-face,
24-shift pilot does not establish population-wide robustness or human smile validity.

Report `results/032_ffhq_update_aware/report.md`. Both two-page PDFs under
`figures/ffhq_update_aware/` were rendered and visually checked. Full W refinement
changes hair/background as well as facial features; several CLIP-valid outputs
are not unequivocal smiles by visual inspection. Post-hoc largest-difference
IDs are[6,7,5,0] for both contrasts; their rankings coincide, but the two heatmaps
are computed from different image pairs and are not identical. Shared absolute
RGB scale; no cherry-picked subset replaces the complete eight-case figure.

## 2026-09-24 — Pause text task019 for image work

User requested saving text experiments and resuming later. Raw computation finished
before the pause:16 solves,48 future models, four focused tests. Saved192 artifact
checksums, exact source/config/task snapshot and preliminary summaries. Input and
test-freeze hashes pass; independent scientific output verification and qualitative
PDF remain pending. No active text processes. Resume instructions:
results/033_text_toxicity_update_aware/RESUME.md. Do not rerun completed experiments.

## 2026-09-24 — Text task019 independently verified and completed

User resumed the paused work. Reporter081 checked192 immutable raw-file hashes,
16 saved populations (re-encoding, exact geometry/assignment, costs, full3-step
SGD sensitivity, predictions, finite coordinate stopping) and48 future heads
against independently unrolled autograd updates. Eight-page qualitative PDF and
HTML include all eight test cases and both word-difference panels. Render checked.

Calibration selects r0/beta1.02 on ties; cal FV87.5%, one failed Base remains.
Test Base and all refined settings FV100%. At beta1.02 positive radii yield the
same sentences as r0. Only case3 changes relative to Base, primarily punctuation.
Sensitivity .0132292→.00924136 and embedding cost .4721274→.4724182 do not establish
improved semantic detoxification. Accepted sentences can retain hostile claims;
removing ambiguous names/tokens can damage meaning. No independent human toxicity
or fluency assessment performed. Classifier test-pair accuracy89.84%.

Continuous midpoint geometry1.270212 becomes1.728938 after text-codebook snapping
(test budget1.295616); cal midpoint also fails. This directly supports explicit
realizability checks, not a blanket claim about all text decoders. Finite-codebook
search lacks the ambient geodesic assumptions. Future studies should improve
candidate semantic coverage and predeclare a harder update protocol; do not select
new settings using this held-out pilot. No new experiment is authorized by this
recommendation. Report: results/033_text_toxicity_update_aware/report.md.

## 2026-09-24 — Repair discrete text stagnation, task020

User requested diagnosis and a fix. Read-only diagnostic082 shows no feasible
single-coordinate lexical change at beta1.02 for any of16 prior cases; minimum
budgets for lexical change1.032–1.209 and improving moves up to1.326. This explains
why image-style near-unit budgets permit only formatting changes in that sparse
codebook, without proving no simultaneous feasible moves exist. Mean prior test
absolute Base probability change under future updates is.002463.

Predeclared fresh-cohort task020 expands samples to64, excludes punctuation/case-only
alternatives, sweeps beta1.02/1.1/1.25/1.5 and r0/.25/1/4, and differentiates the
complete30-step update. Preserve classifier/generator and Base construction; no
artificially weakened margins. All old partition texts excluded from new data.
Exact gradient-Gram factors speed enumeration; cancellation test passes. v1 launch
failed before preparation due dependency path; controller corrected, new v2 root.
The final test remains held out; scientific improvement is not assumed.

### Task020 completion and negative semantic finding

All32 new solves and48 future models completed; five focused tests and independent
checks pass (32 outputs,48 reconstructed heads). Report084 and all-eight PDF/HTML
saved. Calibration r0/beta1.5 yields99.48% test FV vs85.42% Base. Seven word sequences
change, but two collapse to the unrelated "I will watch the movies". Cost rises
.391647→.630717; selected population has7 distinct strings. Grammatical regressions
and retained insults also occur. This removes the no-edit search bottleneck but
is not a successful meaning-preserving detoxification fix. At selected beta all
radii return identical outputs; no robust-term FV gain is established.

At beta1.25 case6 has a lexical r.25/r0 difference; at beta1.1 only case4 punctuation
differs because original factual/Base anchor strings were preserved despite
lexical deduplication of generated alternatives. Do not overstate that filter.
Further work needs explicitly declared per-factual semantic correspondence and
candidate-quality safeguards. No new scientific task is silently launched after
observing these held-out results. Parent diagnosis and all outputs preserved.

## 2026-09-25 — expand ParaDetox factual evaluation without retuning
User requested more examples, then explicitly confirmed eight GPUs. Task021
evaluates64 new factuals in eight8-support populations with task020 settings
and existing24 future heads. Retained Base failures and all output examples.
Selected FV78.91→92.45% is a nominal-refinement gain; r.25 remains identical.
Automatic cosine worsens and polarity reversals persist. Report all cases and
label illustrative reasonable rewrites separately from measured quality; do not
claim independent new-shift evaluation or meaning-preserving detoxification.

## 2026-09-25 — task100 and ten-CF runtime extension
Use PDF73 with explicit user overrides: saved MNIST CNN; HELOC continuous and
FICO non-continuous. User requests one seed and eight GPUs. Experiments1/2
reuse seed0 task016 p05001. Horizon sensitivity is mild (99.87→99.61%);
geometry removal has exactly no effect at r.05 because its constraint is inactive.
User adds Experiment4 ten-CF runtime comparison. Do not substitute old finite-bank
LIFT or invent a FICO encoding to fill current Table1. Experiments3/4 require
reference artifacts; original outputs and launch-time task snapshot preserved.

## 2026-09-25 — dictionary-backed mixed FICO representation
User authorized download of categorical FICO. Original dictionary obtained from
pinned challenge-participant mirror identifies two categorical delinquency fields.
Use explicit categorical encoding, not artificial bins. Preserve continuous HELOC
outputs. Mixed FICO has37 standardized coordinates with seed0; document shared
corpus identity and588 retained all-missing rows. Download and preprocessing checks
complete; no model experiment inferred from this staging step.

## 2026-09-25 — Task100 categorical treatment and execution
User permits PDF categorical preprocessing or a justified better method. Chosen:
retain training-only imputation/one-hot/standardization; inverse dummy affine
transform before A.5 softmax-logit perturbations, restore for model evaluation.
Nominal softmax uses epsilon1e-4 interior smoothing, temperature1. Support-domain
projection is weighted simplex projection in standardized coordinates. Hard
projection is a separate diagnostic, not a validity repair or silent substitute.
Tail resampling now preserves feature/label pairs. Preflight corrected positional
PROPLACE output handling and pandas float64 input for official DiCE0.12.
Three numerical tests pass; all8 preflight workers completed. Full controller094
launched in appendix_tabular/20260925_v1; source frozen until all workers exit.
Raw classifier FV and construction/domain failures will both be disclosed.

## ElliCE convergence diagnosis (task101)
User requested resolving ElliCE warnings. Calibration-only screen and independent
MISOCP bound establish German/logistic epsilon.05 infeasibility under legal hard
categories and native ellipsoid (global margin upper bound-.2565138). Increasing
iterations/patience or reducing step size does not solve it. Epsilon.005/.01 pass
all62 calibration cases. Separate explicit calibration-feasibility gate selects
.01;43/43 new test CFs native robust, FV100%, cost10.73994. Preserve original
Task100 selection/results; do not silently substitute this follow-up. Default
ElliCE requires='valid' can return nonrobust outputs without exceptions. Reporter095
now audits the native robust fraction independently. Report:
results/101_ellice_convergence/conclusion.md. Main eight-worker run continues.

Task100 interim verification: six completed workers pass522 checks, including reconstructed updates, all calibration scores, test geometry and ten-CF metrics. German/MLP CPU reconstruction had a2.18e-5 discrepancy; matching GPU arithmetic passes the original threshold, without changing raw outputs or increasing tolerance. FICO/logistic hard-category realization violates geometry and is explicitly labeled infeasible separately from its feasible relaxed output. Remaining HELOC/FICO MLP workers continue.


## Task100 tabular diagnostics verified
Seed0 eight-worker run /home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1 completed. Results and independent checks:
results/exp3_table1_single_run/summary.md; ten-CF timings:
results/exp4_runtime_10cf/summary.md. Separate hard categorical shift diagnostics
are in hard_category_shift_summary.csv. These are reconstructed diagnostics,
not a verified reproduction of the historical Table1. Keep unsupported mixed
ROAR/PROPLACE, infeasible GRACE outputs and native failures visible. Repeated
cal/test model realizations are counted in independent_checks.csv; no claim of
45 independent unseen distributions. No validity correction.


## Task100 recovery completed (2026-09-26)
The interrupted session left the original controller alive; no experiments were
relaunched. All eight workers and final reporting finished. Final completeness
checks confirm726 passing independent checks,62 result rows,24 runtime rows,
and all414 hashed pre-existing completed-worker artifacts unchanged. Three
focused mixed-domain tests pass. Mixed-domain GRACE timings measure relaxed
simplex outputs, not ten legal hard-category counterfactuals. Final links and
limits are saved in results/100_extra_appendix/progress.md.


## Minimal GRACE--Wachter repository (2026-09-26)
User requested only the core GRACE--Wachter implementation and README. Created
`deliverables/grace-wachter/` and `deliverables/grace-wachter.zip`, containing
only core.py, run.py, config.json, requirements.txt, README.md, and .gitignore.
Preserves24 extracted mathematical/data/model routines unchanged; new standalone
orchestration supports four tabular datasets and MLP/logistic classifiers. No
other CF baseline, raw data, model, or saved result is bundled. Three mixed-domain
tests, an exact assignment check, continuous/mixed synthetic end-to-end runs,
and standalone ZIP import/CLI checks pass. No historical experiment was rerun.


## Task102 population-versus-individual adversary ablation (2026-11-08)
Completed18 main solves (HELOC/German n32, seeds3/4/5) plus full Diabetes n10
debug pair;11 unit tests and18 output/9 pairing checks pass. Report:
results/102_population_individual_adversary/report.md. Raw:
/home/xu.zhao/robust_ce_results/iclr_runs/population_individual_adversary/20261108_main_v1.
Only sensitivity aggregation changes; same geometry/matching/update/optimizer.
Default population branch matches legacy objective/gradient/optimizer. Seeds are
POPULATION SUBSAMPLING, not independent model-training seeds: original seed0 MLP,
Base pool and exact45 future checkpoints fixed. All main outputs feasible within
original tolerance and100% current/FV in relaxed evaluation. At r/b_ref3,beta1.1,
HELOC proximity population1.5777 vs individual1.6355; German reverses2.5521 vs
2.4861. No uniform shared-adversary proximity advantage. Original HELOC selected
r0 pair is byte-identical (cost1.3850); positive radius was predeclared diagnostic.
German hard-category results are separate. Kappa below1 is sensitivity cancellation,
not proof of superior recourse. No convergence claim. Optional64-support runs
not executed (German parent has37 factuals; HELOC deferred for cost). All original
outputs preserved. New entry population_individual_adversary, controller106,
reporter107, config population_individual_adversary.json.


## Original Task100 individual proximity-change audit
Saved results/100_extra_appendix/individual_proximity_changes_v1/report.md and
CSV/LaTeX exports. For HELOC/FICO MLP/logistic, compute paired signed changes
in factual-to-CF standardized L2 after the saved Base-to-Q assignment, all128
test cases per worker. Quantiles use linear interpolation of paired changes,
not differences of marginal quantiles. FICO relaxed/hard outputs are separate;
hard FICO/logistic remains geometry-infeasible. Original mean metrics verified
and all12 raw input hashes unchanged. No experiment rerun or model modification.


## Original Task100 proximity-change audit extended to Diabetes/German
Saved results/100_extra_appendix/individual_proximity_changes_diabetes_german_v1.
Cohort sizes: Diabetes MLP50/logistic47; German MLP37/logistic43. Same signed
paired factual-to-assigned-CF L2 changes and linear percentile definition. German
relaxed and hard realizations reported separately. All12 raw hashes unchanged,
original mean costs reproduced; previous HELOC/FICO summaries preserved.

## 2026-11-09 — Task103 PROPLACE repair and four-GPU authorization
User explicitly requested all four attached GPUs (1,2,3,5) for seeds1,2.
Fixed PROPLACE's adversary output-layer interval: it assumed nonnegative inputs
(invalid for standardized logistic features), while generated output weight
variables were disconnected from the objective and could yield incorrect cuts.
Opt-in adapter now enforces the actual perturbed output affine equality and
checks optimal solver status before accessing X or claiming native robustness.
Eight adapter tests pass, including analytic signed linear and small-MLP minima
and reconstruction with returned adversarial weights. Diabetes/logistic seed1
smoke now generates PROPLACE successfully; previous failed run is preserved.
ROAR can still return an invalid CF; record this as failure, without validity repair.

Task103 launched 2026-11-09 on all four attached GPUs 1,2,3,5; raw root
`/home/xu.zhao/robust_ce_results/iclr_runs/tabular_replication/20261109_full_v1`.
Full MLP explicit-weight adversary exceeded restricted quadratic model size.
Equivalent fixed-input box-weight elimination into signed interval MILP resolves
this without reducing architecture or perturbation radius; reconstructed cuts
are verified. Nine adapter tests plus three mixed objective tests pass, and
both Diabetes classifier smoke tests generate PROPLACE successfully. Dynamic
16-job controller runs in tmux task103_full_20261109 and invokes reporter109.
No final two-seed results at launch. Preserve original artifacts and avoid core
edits while the controller runs.

Task103 expanded at user request to GPUs0–7 on 2026-11-09. Supervisor110
preserves four live worker processes and their raw outputs; old scheduler alone
is SIGSTOPped to prevent its stale queue dispatching duplicates. It is retired
only once all adopted workers finish. Four untouched jobs started on0,4,6,7;
one job already complete, eight active, seven queued at handover. Original
scientific code/config/source hashes unchanged. Raw eight_gpu_expansion.json
and eight_gpu_status.json record scheduling; final reporting owned by110.

Task103 scheduling correction: tmux auto-continued SIGSTOPped original scheduler,
causing duplicate launch attempts (guarded before experiment artifact writes),
text-log truncation and four workers interrupted when supervisor110 exited.
Supervisor111 suspends the dedicated task103 tmux server plus obsolete schedulers,
keeping live scientific workers running; this state was verified. Recovery112
queues four distinct recovery directories using saved prepared/base/solve/future
artifacts. It does not alter original raw runs or scientific core. Final reporter
runs after both queues complete. Use handover_v2_status.json and recovery_status.json,
not obsolete controller exit files, for progress. No final result claims yet.

Task103 all16 jobs completed, including four recovered runs. Reporter109 naming
collision fixed by prefixing future scenario seed/realization columns; experiment
identifiers remain unchanged. Independent reporter rerun exited0; final derived
outputs in results/103_tabular_replication/20261109_full_v1. Original raw outputs
and failed automatic reporting records preserved. No claim that every baseline
construction or GRACE geometry check succeeded follows from worker completion.

## Task104 PROPLACE license fix
German MLP seeds1/2 test failures were restricted Gurobi model-size errors at
factual28/41 followed by skipped cases. Equivalent continuous distance auxiliaries,
license-error-only HiGHS fallback, and exact tight ReLU master formulation are
opt-in. Original radius.005/model/geometry/candidate hull preserved.25/26 failed
or skipped cases recovered within30s;94/95 total CFs independently native robust.
Two-seed relaxed FV98.9362±1.5045%, proximity7.06549±.28003. One remaining timeout
gets a separately recorded300s diagnostic; do not pool or conceal budget changes.
Report results/104_proplace_license_fix; all original raw outputs preserved.

Task104 resolved final timeout with explicit300s diagnostic: seed2_v4 factual41
solved in289.186s using six Gurobi master solves then HiGHS. Independent reporter115
verifies95/95 relaxed CFs native robust and exact saved future evaluations. Two-seed
relaxed CV/FV100±0%, proximity7.12888±.36967. All successful prior outputs unchanged.
Keep same-budget30s row (CV/FV98.9362±1.5045%, proximity7.06549±.28003, one timeout)
separate. Extended report results/104_proplace_license_fix/extended_budget/report.md.
