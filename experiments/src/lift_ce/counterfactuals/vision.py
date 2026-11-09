from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import BaseCEConfig
from ..models.vision import ConvAutoencoder


@dataclass
class CEResult:
    counterfactuals: torch.Tensor
    final_probabilities: torch.Tensor
    steps_used: int
    latents: torch.Tensor | None = None


def _prototype_latents(
    model: nn.Module,
    autoencoder: ConvAutoencoder,
    source: torch.Tensor,
    references: torch.Tensor,
    target_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        source_latent = autoencoder.encode(source)
        reference_latent = autoencoder.encode(references)
        reconstructed_references = autoencoder.decode(reference_latent)
        reference_probabilities = torch.sigmoid(
            model(reconstructed_references)
        )
        valid_mask = reference_probabilities >= target_probability
        valid_latent = (
            reference_latent[valid_mask]
            if bool(valid_mask.any())
            else reference_latent
        )
        nearest = torch.cdist(source_latent, valid_latent).argmin(dim=1)
    return source_latent, valid_latent[nearest]


def generate_near_boundary_ce(
    model: nn.Module,
    autoencoder: ConvAutoencoder,
    sources: torch.Tensor,
    target_references: torch.Tensor,
    config: BaseCEConfig,
    device: torch.device,
    target_probability: float | None = None,
) -> CEResult:
    """Search a decoded latent path to a valid target-class prototype."""
    model.eval()
    autoencoder.eval()
    source = sources.to(device)
    references = target_references.to(device)
    if target_probability is None:
        target_probability = config.threshold + config.margin
    source_latent, target_latent = _prototype_latents(
        model, autoencoder, source, references, target_probability
    )

    completed = torch.zeros(len(source), dtype=torch.bool, device=device)
    counterfactuals = source.clone()
    returned_latents = torch.full_like(source_latent, float("nan"))
    with torch.no_grad():
        final_probabilities = torch.sigmoid(model(source))
    steps_used = config.steps

    # Decode the latent path directly. No source/target pixel interpolation is
    # used, avoiding the characteristic double-exposure appearance.
    for step in range(1, config.steps + 1):
        alpha = step / config.steps
        with torch.no_grad():
            latent = (1.0 - alpha) * source_latent + alpha * target_latent
            candidates = autoencoder.decode(latent)
            probabilities = torch.sigmoid(model(candidates))
        newly_completed = (~completed) & (probabilities >= target_probability)
        counterfactuals[newly_completed] = candidates[newly_completed]
        returned_latents[newly_completed] = latent[newly_completed]
        final_probabilities[newly_completed] = probabilities[newly_completed]
        completed |= newly_completed
        if bool(completed.all()):
            steps_used = step
            break

    # A rare fallback optimizes only latent coordinates while all generated
    # images still pass through the decoder.
    if not bool(completed.all()):
        latent = source_latent.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=0.03)
        for _ in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            candidates = autoencoder.decode(latent)
            probabilities = torch.sigmoid(model(candidates))
            validity_loss = torch.relu(
                target_probability - probabilities
            )
            source_loss = (latent - source_latent).pow(2).mean(dim=1)
            prototype_loss = (latent - target_latent).pow(2).mean(dim=1)
            active = (~completed).float()
            loss = (
                active
                * (
                    20.0 * validity_loss
                    + 0.2 * source_loss
                    + 0.05 * prototype_loss
                )
            ).sum() / active.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidates = autoencoder.decode(latent)
                probabilities = torch.sigmoid(model(candidates))
                newly_completed = (~completed) & (
                    probabilities >= target_probability
                )
                counterfactuals[newly_completed] = candidates[newly_completed]
                returned_latents[newly_completed] = latent[newly_completed]
                final_probabilities[newly_completed] = probabilities[
                    newly_completed
                ]
                completed |= newly_completed
                if bool(completed.all()):
                    break

    return CEResult(
        counterfactuals.detach().cpu(),
        final_probabilities.detach().cpu(),
        steps_used,
        returned_latents.detach().cpu(),
    )


def generate_stability_baseline(
    model: nn.Module,
    autoencoder: ConvAutoencoder,
    sources: torch.Tensor,
    target_references: torch.Tensor,
    config: BaseCEConfig,
    method: str,
    device: torch.device,
) -> CEResult:
    """Paper-inspired T-Rex/SNS baselines on the shared latent path."""
    model.eval()
    autoencoder.eval()
    source = sources.to(device)
    references = target_references.to(device)
    threshold = config.threshold + config.margin
    source_latent, target_latent = _prototype_latents(
        model, autoencoder, source, references, threshold
    )
    candidates: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for step in range(1, config.steps + 1):
        alpha = step / config.steps
        with torch.no_grad():
            image = autoencoder.decode(
                (1.0 - alpha) * source_latent + alpha * target_latent
            )
            probability = torch.sigmoid(model(image))
        candidates.append(image)
        probabilities.append(probability)
    candidate_tensor = torch.stack(candidates)
    probability_tensor = torch.stack(probabilities)
    valid = probability_tensor >= threshold

    if method == "trex_style":
        robust_scores: list[torch.Tensor] = []
        generator = torch.Generator(device=device).manual_seed(17)
        for image in candidates:
            noisy = image.unsqueeze(0).expand(
                config.trex_samples, *image.shape
            )
            noise = torch.randn(
                noisy.shape, generator=generator, device=device
            ) * config.trex_noise_sigma
            noisy = (noisy + noise).clamp(0.0, 1.0)
            with torch.no_grad():
                noisy_probabilities = torch.sigmoid(
                    model(noisy.flatten(0, 1))
                ).view(config.trex_samples, len(source))
            score = noisy_probabilities.mean(dim=0) - config.trex_beta * (
                noisy_probabilities.std(dim=0)
            )
            robust_scores.append(score)
        scores = torch.stack(robust_scores)
        acceptable = scores >= config.trex_stability_threshold
        selection_mask = acceptable
        objective = torch.arange(
            1, config.steps + 1, device=device
        )[:, None].float()
        objective = objective.expand_as(scores).clone()
        objective[~selection_mask] = float("inf")
    elif method == "sns_style":
        gradient_norms: list[torch.Tensor] = []
        for image in candidates:
            differentiable = image.detach().requires_grad_(True)
            probability = torch.sigmoid(model(differentiable))
            gradient = torch.autograd.grad(
                probability.sum(), differentiable
            )[0]
            gradient_norms.append(gradient.flatten(1).norm(dim=1))
        gradients = torch.stack(gradient_norms)
        normalized_gradient = gradients / gradients.median(
            dim=0
        ).values.clamp_min(1e-8)
        path_cost = torch.arange(
            1, config.steps + 1, device=device
        )[:, None] / config.steps
        objective = (
            path_cost + config.sns_gradient_weight * normalized_gradient
        )
        objective[~valid] = float("inf")
    else:
        raise ValueError(f"Unknown stability baseline: {method}")

    unresolved = torch.isinf(objective).all(dim=0)
    selected_step = objective.argmin(dim=0)
    if method == "trex_style":
        fallback_scores = scores.masked_fill(~valid, float("-inf"))
        selected_step[unresolved] = fallback_scores[:, unresolved].argmax(dim=0)
    else:
        selected_step[unresolved] = config.steps - 1
    batch_indices = torch.arange(len(source), device=device)
    selected = candidate_tensor[selected_step, batch_indices]
    selected_probabilities = probability_tensor[selected_step, batch_indices]
    return CEResult(
        selected.detach().cpu(),
        selected_probabilities.detach().cpu(),
        int(selected_step.max().item() + 1),
    )
