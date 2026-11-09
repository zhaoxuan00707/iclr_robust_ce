# Original LeapFactual Galaxy asset inventory

Inspected 2026-09-16. Original project:
`/mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/` (abbreviated LF below).
Original assets remain read-only. No classifier/encoder/decoder retraining planned.

## Task and data

`LF/data/datasets.py:Galaxy10` wraps ImageFolder; folder labels ans_0 … ans_9
map in numeric order to the 10 DECaLS morphology classes: Disturbed, Merging,
Round Smooth, In-between Round Smooth, Cigar Shaped Smooth, Barred Spiral,
Unbarred Tight Spiral, Unbarred Loose Spiral, Edge-on without Bulge, Edge-on with
Bulge. This is a **10-class prediction task**, not the later binary 1→2 prototype.

`LF/notebooks/Galaxy10_grid_search_robust_multi.py` requests random target labels
0…9 for held-out examples; `Galaxy10_generate_data.py` cycles through every target.
There is no single globally favorable Galaxy label. The replication will preserve
the original multiclass classifier and run each target label as a separate
requested-recourse task. Target probability is the relevant softmax coordinate;
threshold .5 implies target argmax, and both threshold and argmax validity will be
reported. Factual true-label-equals-target cases are excluded before generation.

Exact existing folder split:
- Base classifier training: `/mnt/big_disk/z.cao/Galaxy10/image_data_train_20/`.
- Full training: `/mnt/big_disk/z.cao/Galaxy10/image_data_train/`.
- Test: `/mnt/big_disk/z.cao/Galaxy10/image_data_test/`.
- Original HDF5: `/mnt/big_disk/z.cao/Galaxy10/Galaxy10_DECals.h5`.

The original DataModule uses the test folder for validation as well. There is no
separate original calibration split. The new calibration factual subset will be
seed-selected from full training excluding base-classifier training filenames;
final factuals will come from the original test folder. This is a declared new
split for GRACE selection, not a claim of unseen original checkpoint selection.
Exact file IDs, classes, selection seeds, and hash manifests will be saved.

## Preprocessing

`LF/data/dataset_lit.py` uses RGB ToTensor in [0,1], center crop150, antialiased
resize128 for evaluation. Original training adds RandomAffine(-180,180) before
crop. `WrapedVGG.forward` additionally applies torchvision VGG16_BN ImageNet
transforms (resize256, center crop224, ImageNet normalization). WAE consumes the
RGB128 [0,1] tensor directly. Its latent coordinates are unnormalized 64-vectors.
The kernel prior variance2 is a training regularizer, not a latent rescaling.

## Checkpoints and architectures

The exact checkpoints referenced by the original CF notebook exist in LF:
- `logs/Galaxy10_VGG_20/version_0/checkpoints/epoch=47-step=624.ckpt`:
  `models/classifiers/wraped_vgg.py:WrapedVGG`, torchvision VGG16_BN with final
  Linear4096→10; checkpoint epoch47. This is the predefined f0.
- `logs/Galaxy10_VGG/version_0/checkpoints/epoch=27-step=1764.ckpt`:
  original separate robust/full-data comparator. Not chosen as f0 by new outcomes.
- `logs/Galaxy10_VAE/version_8/checkpoints/last.ckpt`: epoch299,
  `models/generative_models/VAE/wae_mmd.py:WAE_MMD`, convolutional residual
  encoder/decoder widths16,32,64,128,256; 4×4×256 bottleneck, Linear4096→512→64;
  mirrored decoder with sigmoid output. Encoder and decoder share this checkpoint.
- `notebooks/checkpoints/galaxy10_flow_weights_mlp_finetune.pt`:
  conditional time-dependent MLP_CFM; 64 latent +10 class +1 time inputs,
  five width256 SELU hidden layers, 64-vector output.

Other checkpoints exist under Galaxy10_VGG_strong, VGG_CE/robustCE and VAE
versions, but these are not selected using final future validity. Historical
notebooks reference an obsolete `/home/z.cao/massdisk/workspace/OnGoing/LeapFactual`
prefix; the actual identical relative paths are under LF above.

Inference-only WAE definitions and original flow/ODE routines were extracted
into `experiments/src/lift_ce/galaxy/`. Source hashes and changes are recorded in
`experiments/provenance/galaxy_import.json`. Unused WAE perceptual-loss and GAN
training modules are omitted; all encoder/decoder keys are loaded strictly.
VGG architecture uses weights=None before loading the complete trained checkpoint,
avoiding an unnecessary ImageNet download. Its preprocessing is unchanged.
Numerical reconstruction and strict-load checks are stored separately in
`results/010_galaxy_asset_checks.json` after the audit.

## Generator, geometry, update and shift protocols

The original `progressive_leap` uses conditional flow lifting/landing, gamma=.1
for both legs, 250 iterations, 15 trajectory sample times, torchdyn NeuralODE
dopri5, atol=rtol=1e-4. Preserve this Base generator. Its robust continuation uses
gamma_lift1 and gamma_land1.025 with 50 additional iterations; this is a compatible
LeapFactual baseline. No tabular baseline ports are needed.

No validated alternative Galaxy transport ground metric was found; use Euclidean
64-dimensional latent distance for every empirical W2 leg. Original SSIM/PSNR
are available as image diagnostics; no new learned perceptual metric is needed.

`LF/experiments/exp_classifier.py` configures AdamW with LR1e-4 and ExponentialLR
.875. Although YAML declares weight_decay .0005, the code omits that argument,
so AdamW's .01 default is the effective setting. Original training is multiclass
cross entropy, with all VGG parameters trainable. A fixed post-training update
will be declared separately (epoch/batch/augmentation choices included).

The original robustness comparator uses separately trained classifiers and CF
augmentation, not a verified severity-indexed data-shift→fixed-update pipeline.
A minimal exposure/noise stress suite and calibration protocol must therefore be
clearly labeled new. Neither parameter perturbations nor the incompatible later
128-dimensional binary latent MLP will replace the original image classifier.

## Other existing Galaxy runs, preserved

`/mnt/big_disk/xu.zhao/robust_ce_results/galaxy_latent_lift_v1/` and sibling
`galaxy_latent_lift_v1_repr/` use a later binary Merging→Round task, AE128 and a
latent MLP. Existing current_quality.json reports accuracy .91657, AUC .96958.
These are incompatible with the original 10-way image classifier / WAE64 / flow;
no outputs, model weights or original aggregate claims are reused as GRACE results.
Original factual/reference IDs for the requested new frozen cohorts were not
found; seed-fixed cohorts will be declared before any future evaluation.

## Hardware and orchestration

Host ias8026, eight A10080GB devices0–7 attached through genv by user instruction.
Several devices also host existing unrelated workloads; use bounded microbatches
and one worker per GPU without stopping those jobs. The smallest observed free
memory is about18GB. The shell's stale CUDA_VISIBLE_DEVICES=1,2,3,5 must be replaced
per worker. SLURM client exists but external `sinfo` times out; use a local
GPU-aware launcher with disjoint per-job output directories and phase barriers.
Original scripts use multiprocessing and Lightning DDP. No other usable allocated
server has been identified. `hardware_manifest.json` will record the fresh query.
