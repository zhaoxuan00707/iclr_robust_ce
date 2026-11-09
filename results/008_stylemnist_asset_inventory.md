# StyleMNIST GRACE discovery — 2026-09-16

Status: discovery completed for the paths below; task identity and seven-GPU
allocation unresolved. No GRACE experiment has been launched or results claimed.

Audit output: `/home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_asset_audit_01/`.
Reproduce into a **new** directory with:

```bash
OMP_NUM_THREADS=1 /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python \
  experiments/audits/stylemnist_assets.py --output <new-external-audit-directory>
```

## Dataset and task identity

No explicitly named StyleMNIST loader or dataset was found in inspected project
source and user-owned model storage. This does not prove that no such asset exists
on another server. Inspected locations include `/home/xu.zhao/robust_ce/`,
`/home/xu.zhao/robust_ce_results/`, `/home/xu.zhao/robust_ce_data/`,
`/mnt/big_disk/xu.zhao/`, and `/mnt/mass_disk1/shared_models/`.

Historical loader: `/home/xu.zhao/robust_ce/src/lift_ce/data/vision.py`.
`BinaryMNIST` loads torchvision MNIST, restricts two digit labels, maps source to
0 and target to 1, and uses `TF.to_tensor`: grayscale 1×28×28 in [0,1]. Data also
exists at `/mnt/big_disk/xu.zhao/robust_ce_data/MNIST/`.

`/home/xu.zhao/robust_ce/configs/mnist_strong_generator_visual_4_to_9.yaml`
explicitly defines source_digit=4 and target_digit=9. Those numbers denote digits,
not six independently trained classifiers. No separate classifier-ID 4–9 mapping
was recovered. Whether the requested StyleMNIST refers to this task remains open.

## Compatible candidate weights

`/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt`
contains `model`, `ae`, and `sources`. Strict loading succeeded with repository-local
`SmallCNN(embedding_dim=128)` and `StrongConvAutoencoder(latent_dim=64)`.
SmallCNN uses four convolutional layers (32,32,64,64), batch normalization,
7×7 pooled features, a 128-dimensional embedding, and one binary logit.
StrongConvAutoencoder is a convolutional residual encoder/decoder with a
64-dimensional bottleneck and sigmoid image output.

CPU sanity on all 64 saved sources: reconstruction MSE **0.0087675154**;
classifier decision agreement between factual and reconstructed images **98.4375%**.
These are factual-only diagnostics, not a full dataset or counterfactual gate.
The checkpoint SHA-256 is in `classifier_role_manifest.json`.

Producer: `/home/xu.zhao/robust_ce/src/lift_ce/experiments/mnist_roar_latent.py`.
Source defaults indicate seed 0, 5 classifier epochs, 12 AE epochs, binary
training subset 5000, calibration subset 1000, and test subset 512. The original
invocation was not recovered; these defaults are not verified run metadata.
The sibling `mnist_roar_latent_comparison/payload.pt` is another candidate.
Neither historical run was modified or rerun.

## Other AEs and historical failure evidence

`/home/xu.zhao/robust_ce_results/mnistc_latent_lift_v1/` contains
`ae_8.pt`, `ae_16.pt`, `ae_32.pt`, `vae_8.pt`, `vae_16.pt`, `vae_32.pt`,
`reference_classifier.pt`, `latent_current.pt`, and corruption-specific models.
These belong to the historical **3→8** latent experiment, not an established
4→9 StyleMNIST task. AE checkpoints use StrongConvAutoencoder; VAE checkpoints
use ConvVAE; their latent dimensionalities are encoded in their filenames.

`/home/xu.zhao/robust_ce_results/mnistc_latent_lift_v2_repr/` contains AE and
VAE checkpoints for dimensions 16, 32, 64 and `representation_quality.csv`.
Reported reconstruction MSE / semantic agreement:

| Model | MSE | Agreement |
|---|---:|---:|
| AE16 | .00607639 | .976736 |
| AE32 | .00353494 | .995347 |
| AE64 | .00270866 | .995705 |
| VAE16 | .00854407 | .963135 |
| VAE32 | .00732474 | .969578 |
| VAE64 | .00670624 | .977452 |

These are existing results, not newly measured cross-task compatibility.
`/home/xu.zhao/robust_ce_results/mnist_latent_cf_generator_v2/mnist_latent_cf_generator_v2.md`
reports AE64 latent validity 98.44% but decode/re-encode validity only 0.78%.
The historical latent optimizer is not a verified deployable image generator.
Do not silently reuse its latent predictions as decoded-image predictions.

## Existing CF and future-shift code

`/home/xu.zhao/robust_ce/src/lift_ce/counterfactuals/vision.py` implements
`generate_near_boundary_ce`, used in the 4→9 image-classifier experiment.
The old implementation should be imported with provenance into the authoritative
checkout if this is confirmed as the intended task.

`/home/xu.zhao/robust_ce/src/lift_ce/experiments/mnist_method_pilot.py`
uses latent Euclidean empirical W2 (Hungarian matching on equal supports).
Its future protocol retrains classifiers on changed training data: rotation
15 degrees, translation 2 pixels, and noise .08. This is an existing **MNIST**
protocol, not verified as a separately named StyleMNIST protocol. These different
shift types are not a single ordered severity axis.

Existing pilot results in
`/mnt/big_disk/xu.zhao/robust_ce_results/mnist_method_pilot_4_to_9/summary.json`:
Base/LIFT validity rotation .8125/.96875, translation .828125/.9375,
noise .875/.984375; pixel L2 cost 5.57556/7.63274. This historical LIFT result
is **not** the requested GRACE result. Pilot `artifacts.pt` saves CF tensors,
not classifier/AE state dictionaries.

`mnist_roar_latent.py` includes a labeled ROAR-Latent adaptation. It is not
canonical ROAR. Its parameter-noise diagnostic must not become the primary
future-shift evaluation. Compatibility and any baseline inclusion remain pending.

## Hardware and launch infrastructure

Host `ias8026`; physical NVIDIA A100 80GB GPUs 0–7. Initial shell has
`CUDA_VISIBLE_DEVICES=1,2,3,5`, `GENV_GPUS=4`. The PATH-selected `nvidia-smi`
is a genv shim that filters the display. `/usr/bin/nvidia-smi` reveals all eight.
Several GPUs run other users' jobs. User specifies seven GPUs; the intended
seven-device allocation is pending. No existing process was stopped.

SLURM client binaries exist, but `sinfo` and `scontrol show config` timed out
(10 and 5 seconds respectively); scheduler availability is not established.
Historical MNIST scripts use multiprocessing/spawn and seven workers. Current
repository controllers use explicit GPU worker subprocesses. No other authorized
server allocation or SSH alias was identified. Do not infer availability from
physical presence or free memory alone.

## Next steps and scientific constraints

Resolve whether StyleMNIST means discovered MNIST 4→9 with style shifts or a
separate dataset, and identify the seven intended GPUs. Then freeze a compatible
checkpoint and dataset, complete reconstruction/base-CF checks, predeclare update
and attack/search protocols, and implement the requested no-current-validity-check
GRACE branch. Use max-model mean-CF hinge and beta {1,1.05,1.10,1.20}; separate
latent W2 geometry from pixel recourse cost. No future outcome may select assets.

No factual IDs, attack bank, ablation, future validity, runtime results or figures
have been produced for GRACE. No manuscript claims were changed.
