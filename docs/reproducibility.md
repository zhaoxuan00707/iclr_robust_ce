# Reproduction guide

Run commands from the repository root. The [version 88 paper](../iclr_robust_ce-88.pdf) is the method and presentation reference. This guide distinguishes a portable example from research workflows that depend on externally stored assets.

## Environment

The portable example uses the pinned [requirements](../deliverables/grace-wachter/requirements.txt). The full research package declares its dependencies in [pyproject.toml](../experiments/pyproject.toml); the recorded environment is in [the environment manifest](../results/003_core_migration_environment.json).

For the full source tree in a fresh Python 3.11 environment:

```bash
python -m pip install -e './experiments[dev,baselines]'
python -m pip install dice-ml==0.12
python experiments/run.py list
python experiments/run.py run appendix_tabular -- --help
```

The launcher loads the local `lift_ce`, RobustX, and ElliCE sources. Do not install the vendored projects' historical dependency pins wholesale over a working environment. PROPLACE requires solver support; Gurobi license limits affect some problems. The opt-in license fallback and its separately recorded computational budgets are described in [the solver follow-up](../results/104_proplace_license_fix/README.md).

Image dependencies are separate: [Galaxy](../experiments/requirements-galaxy.txt), [FFHQ](../experiments/requirements-ffhq.txt), and [FFHQ identity extensions](../experiments/requirements-ffhq-identity.txt). Identity extensions are historical experiments, not requirements for the paper's unconstrained full-W FFHQ workflow.

A fresh install, downloads, CUDA compatibility on other hardware, and full experimental execution were not revalidated during the documentation-only release preparation.

## Experiment entry points

These are existing scripts, not a promise that their external artifacts are bundled. The linked configuration and result report identify the required data, checkpoint, and original run locations. Preserve completed output directories.

| Workflow | Controller / runner | Configuration | Saved report |
|---|---|---|---|
| Portable Wachter + GRACE | [run.py](../deliverables/grace-wachter/run.py) | [config.json](../deliverables/grace-wachter/config.json) | Output-local metrics; see example README |
| Tabular single-run pipeline | [appendix_tabular.py](../experiments/src/lift_ce/experiments/appendix_tabular.py), through `experiments/run.py` | [tabular](../experiments/configs/appendix_tabular.json) | [single-run summary](../results/exp3_table1_single_run/summary.md) |
| Two additional tabular seeds | [controller108](../experiments/108_run_tabular_replication.py) | [seed 1](../experiments/configs/tabular_replication_seed1.json), [seed 2](../experiments/configs/tabular_replication_seed2.json) | [two-seed report](../results/103_tabular_replication/20261109_full_v1/report.md) |
| MNIST lower-margin Base study | [controller064](../experiments/064_run_mnist_low_margin_base.py) | [MNIST](../experiments/configs/mnist_low_margin_base.json) | [report](../results/030_mnist_low_margin_base/report.md) |
| Galaxy update-aware refinement | [controller069](../experiments/069_run_galaxy_update_aware.py) | [Galaxy](../experiments/configs/galaxy_update_aware.json) | [report](../results/031_galaxy_update_aware/report.md) |
| FFHQ update-aware refinement | [controller072](../experiments/072_run_ffhq_update_aware.py) | [FFHQ](../experiments/configs/ffhq_update_aware.json) | [report](../results/032_ffhq_update_aware/report.md) |
| Population / individual adversary study | [controller106](../experiments/106_run_adversary_ablation.py) | [ablation](../experiments/configs/population_individual_adversary.json) | [report](../results/102_population_individual_adversary/report.md) |

For full tabular baseline reproduction, the runner accepts `--dataset`, `--model`, `--config`, and a fresh `--output`. Unlike the portable example, the research launcher deliberately restricts output paths to its configured external run root. Several configurations also contain original server paths. Inspect and stage those paths before execution; simply cloning the repository does not provide their assets.

The following resolves the existing launcher's permitted root rather than assuming that an arbitrary directory is supported:

```bash
RUN_ROOT=$(python -c "import runpy; print(runpy.run_path('experiments/run.py')['RUN_ROOT'])")
python experiments/run.py run appendix_tabular \
  --output "$RUN_ROOT/grace_new_diabetes_mlp" --dry-run -- \
  --dataset diabetes --model mlp \
  --config experiments/configs/appendix_tabular.json
```

This **dry run writes nothing and launches no experiment**. After staging data and checking path permissions, removing `--dry-run` runs the full pipeline. It uses CUDA and includes calibration; it is not a lightweight test. The standalone example is the recommended entry point on a new machine.

To inspect multi-job and image controller interfaces without launching experiments:

```bash
python experiments/108_run_tabular_replication.py --help
python experiments/069_run_galaxy_update_aware.py --help
python experiments/072_run_ffhq_update_aware.py --help
python experiments/106_run_adversary_ablation.py --help
```

Historical recovery/scheduler scripts reproduce recorded operational events; they are not general-purpose launch commands. In particular, do not rerun fixed-path recovery scripts or previously completed run roots.

## Data and frozen assets

- **Tabular:** the portable runner supports OpenML downloads and a local mixed-FICO CSV. Full-run dataset paths are declared in their configurations. The [FICO audit](../results/100_extra_appendix/fico_data_audit/report.md) records the source hashes and categorical schema.
- **MNIST:** stage the frozen CNN, decoder, actual Base latent codes, training-update images, and reference populations identified in the MNIST configuration/report. CF optimization and geometry are latent-space operations; data-side update sensitivity remains in pixel space.
- **Galaxy:** stage the VGG classifier, frozen generator and actual Base codes, class-balanced update images, and reference population. The update changes the final classification layer.
- **FFHQ:** stage StyleGAN, the original CLIP model, nominal Base W codes, and fixed update/reference assets. The prescribed update changes the CLIP visual projection.
- **ParaDetox:** the appendix adaptation uses a frozen sentence representation and finite candidate codebooks. Its [report](../results/034_text_toxicity_followup/report.md) documents the assets and limitations; it is not continuous token-level optimization.

No checkpoints or original datasets are added to this source release. Public download URLs/checksums for all frozen image assets are not yet packaged in one release manifest.

## Lightweight verification

These tests do not train models or reproduce expensive experiments:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=experiments/src:experiments/vendor/RobustX:experiments/vendor/ellice \
python -m unittest discover -s experiments/tests -p test_mixed_update_aware.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=experiments/src:experiments/vendor/RobustX:experiments/vendor/ellice \
python -m unittest discover -s experiments/tests -p test_sensitivity_aggregation.py
```

The second suite checks sensitivity aggregation and the default population optimizer against the retained implementation. Solver-specific tests additionally require a usable Gurobi installation. Consult each saved report for the independent checks performed on that experiment's artifacts.

## Reading results

Report seed-level mean and sample standard deviation; future-model scenarios are nested evaluations, not additional independent training seeds. The population-versus-individual ablation uses population-subsampling seeds with a fixed classifier, a different statistical unit. Some corrected baseline summaries use two training seeds rather than three.

Do not average relaxed and hard-category outputs together, conceal failed generations, treat solver completion as geometry feasibility, or present an extended-timeout diagnostic as an unchanged runtime budget. Saved result files and scientific provenance are retained without editing their measurements.
