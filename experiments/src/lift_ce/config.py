from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    source_digit: int = 3
    target_digit: int = 8
    train_limit: int | None = None
    test_limit: int | None = None
    reference_size: int = 512
    ce_batch_size: int = 64
    target_batch_size: int = 64
    loader_batch_size: int = 128
    num_workers: int = 0


@dataclass
class ClassifierConfig:
    epochs: int = 8
    future_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class BaseCEConfig:
    threshold: float = 0.5
    margin: float = 0.02
    steps: int = 96
    autoencoder_epochs: int = 15
    autoencoder_lr: float = 1e-3
    autoencoder_type: str = "conv"
    latent_dim: int = 64
    high_confidence_levels: list[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.9]
    )
    trex_noise_sigma: float = 0.05
    trex_samples: int = 16
    trex_beta: float = 2.0
    trex_stability_threshold: float = 0.7
    sns_gradient_weight: float = 0.5


@dataclass
class LiftCEConfig:
    iterations: int = 10
    eta: float = 0.2
    alpha_barycenter: float = 0.5
    sinkhorn_reg: float = 0.05
    sinkhorn_iterations: int = 100
    ot_space: str = "classifier_embedding"
    top_k: int = 4
    lens_epsilon_initial_ratio: float = 0.45
    lens_delta_initial_ratio: float = 0.65
    stop_threshold: float = 1e-3
    lens_alpha: float = 0.5
    lens_slack: float = 0.05
    calibration_coverage: float = 0.8
    calibration_lambdas: list[float] = field(
        default_factory=lambda: [i / 20 for i in range(21)]
    )
    refinement_depths: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0]
    )
    bound_beta_grid: list[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95]
    )
    bound_gamma_grid: list[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
    )


@dataclass
class FutureShiftConfig:
    rotation: list[float] = field(default_factory=lambda: [10, 15, 20])
    translation: list[float] = field(default_factory=lambda: [1, 2, 3])
    gaussian_noise: list[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15]
    )


@dataclass
class CalibrationShiftConfig:
    rotation: list[float] = field(default_factory=lambda: [5.0])
    translation: list[float] = field(default_factory=lambda: [1.0])
    gaussian_noise: list[float] = field(default_factory=lambda: [0.025])


@dataclass
class ExperimentConfig:
    run_name: str = "mnist_3_to_8"
    output_dir: str = "outputs"
    data_dir: str = "data"
    device: str = "auto"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    base_ce: BaseCEConfig = field(default_factory=BaseCEConfig)
    lift_ce: LiftCEConfig = field(default_factory=LiftCEConfig)
    calibration_shifts: CalibrationShiftConfig = field(
        default_factory=CalibrationShiftConfig
    )
    future_shifts: FutureShiftConfig = field(default_factory=FutureShiftConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return ExperimentConfig(
        run_name=raw.get("run_name", "mnist_3_to_8"),
        output_dir=raw.get("output_dir", "outputs"),
        data_dir=raw.get("data_dir", "data"),
        device=raw.get("device", "auto"),
        seeds=list(raw.get("seeds", [0, 1, 2])),
        dataset=DatasetConfig(**raw.get("dataset", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
        base_ce=BaseCEConfig(**raw.get("base_ce", {})),
        lift_ce=LiftCEConfig(**raw.get("lift_ce", {})),
        calibration_shifts=CalibrationShiftConfig(
            **raw.get("calibration_shifts", {})
        ),
        future_shifts=FutureShiftConfig(**raw.get("future_shifts", {})),
    )


def apply_smoke_overrides(config: ExperimentConfig) -> ExperimentConfig:
    config.seeds = config.seeds[:1]
    config.dataset.train_limit = 3000
    config.dataset.test_limit = 512
    config.dataset.reference_size = 64
    config.dataset.ce_batch_size = min(config.dataset.ce_batch_size, 16)
    config.dataset.target_batch_size = min(config.dataset.target_batch_size, 16)
    config.classifier.epochs = 5
    config.classifier.future_epochs = 5
    config.base_ce.steps = 64
    config.base_ce.autoencoder_epochs = 10
    config.lift_ce.iterations = 5
    config.lift_ce.sinkhorn_iterations = 100
    config.future_shifts.rotation = config.future_shifts.rotation[:1]
    config.future_shifts.translation = []
    config.future_shifts.gaussian_noise = []
    config.run_name += "_smoke"
    return config
