from __future__ import annotations

import copy

import torch

from ..models.tabular import TabularMLP


def generate_gradient_ce(
    model: TabularMLP,
    sources: torch.Tensor,
    threshold: float,
    steps: int,
    learning_rate: float,
    proximity_weight: float,
    target_references: torch.Tensor | None = None,
    realism_weight: float = 0.0,
) -> torch.Tensor:
    model.eval()
    source = sources.detach()
    references = target_references.detach() if target_references is not None else None
    candidate = source.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    completed = torch.zeros(len(source), dtype=torch.bool)
    result = source.clone()
    best_score = torch.full((len(source),), float("inf"))
    use_realism = realism_weight > 0 and references is not None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probability = torch.sigmoid(model(candidate))
        validity_loss = torch.relu(threshold - probability)
        distance = (candidate - source).pow(2).mean(dim=1)
        if use_realism:
            target_distance = torch.cdist(candidate, references).pow(2) / candidate.shape[1]
            realism = target_distance.min(dim=1).values
            active = torch.ones_like(validity_loss)
        else:
            realism = torch.zeros_like(validity_loss)
            active = (~completed).float()
        objective = (
            20 * validity_loss
            + proximity_weight * distance
            + realism_weight * realism
        )
        loss = (active * objective).sum() / active.sum().clamp_min(1)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid = torch.sigmoid(model(candidate)) >= threshold
            if use_realism:
                score = proximity_weight * distance + realism_weight * realism
                improved = valid & (score < best_score)
                result[improved] = candidate[improved]
                best_score[improved] = score[improved]
                completed |= valid
            else:
                newly_valid = (~completed) & valid
                result[newly_valid] = candidate[newly_valid]
                completed |= newly_valid
            if bool(completed.all()) and not use_realism:
                break
    return torch.where(completed[:, None], result, candidate.detach())


def generate_wachter_ce(
    model: TabularMLP,
    sources: torch.Tensor,
    threshold: float,
    steps: int,
    learning_rate: float,
    proximity_weight: float,
    validity_weight: float = 20.0,
) -> torch.Tensor:
    """Wachter-style black-box counterfactual generation.

    This follows the classical objective from Wachter et al. (2017): find a
    minimally changed input whose model output reaches the desired target. We
    optimize only the queried model score and a proximity penalty, without
    using target-class samples or model internals.
    """
    model.eval()
    source = sources.detach()
    candidate = source.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    result = source.clone()
    best_score = torch.full((len(source),), float("inf"))
    completed = torch.zeros(len(source), dtype=torch.bool)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probability = torch.sigmoid(model(candidate))
        validity_loss = torch.relu(threshold - probability).pow(2)
        distance = (candidate - source).pow(2).mean(dim=1)
        objective = validity_weight * validity_loss + proximity_weight * distance
        loss = objective.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probability = torch.sigmoid(model(candidate))
            valid = probability >= threshold
            improved = valid & (distance < best_score)
            result[improved] = candidate[improved]
            best_score[improved] = distance[improved]
            completed |= valid
    return torch.where(completed[:, None], result, candidate.detach())


def generate_apas_original_style_ce(
    model: TabularMLP,
    sources: torch.Tensor,
    delta: float,
    realization_count: int,
    threshold: float,
    steps: int,
    learning_rate: float,
    proximity_weight: float,
    validity_weight: float = 20.0,
    initial_margin: float = 0.01,
    margin_step: float = 0.2,
    max_margin: float = 4.0,
    seed: int = 0,
) -> torch.Tensor:
    """APAS-style robust CE generation under sampled PMS.

    The APAS implementation first computes a standard counterfactual, then
    checks whether it remains valid under plausible model shifts. If not, it
    recomputes the CE with an increasingly larger output margin ``eps`` until
    the PMS robustness check succeeds. Their MILP uses a pre-sigmoid
    constraint logit >= eps for the positive class; here we realize the same
    schedule for our differentiable Wachter generator by converting the margin
    into the equivalent probability threshold sigmoid(eps).

    This baseline is independent of the LIFT lens calibration: ``delta`` is
    the APAS/PMS parameter radius being certified.
    """
    output = sources.detach().clone()
    resolved = torch.zeros(len(sources), dtype=torch.bool)
    best_delta = torch.full((len(sources),), -1.0)
    best_cost = torch.full((len(sources),), float("inf"))
    margin = float(initial_margin)
    iteration = 0
    while margin <= max_margin + 1e-12:
        probability_threshold = float(torch.sigmoid(torch.tensor(margin)).item())
        candidate = generate_wachter_ce(
            model,
            sources,
            probability_threshold,
            steps,
            learning_rate,
            proximity_weight,
            validity_weight,
        )
        certified = aps_robust_mask(
            model,
            candidate,
            delta,
            realization_count,
            threshold,
            seed + 10_000 * iteration,
        )
        cost = (candidate - sources).pow(2).mean(dim=1).sqrt()
        improved = (~resolved & certified) | (
            certified & (best_delta < delta - 1e-12) & (cost < best_cost)
        )
        output[improved] = candidate[improved]
        best_cost[improved] = cost[improved]
        best_delta[improved] = delta
        resolved |= certified
        if bool(resolved.all()):
            break
        margin += margin_step
        iteration += 1
    if not bool(resolved.all()):
        # APAS can return no CE when the robust search fails. For fair metric
        # aggregation in this script, keep the strongest attempted candidate
        # for unresolved points; failures remain visible through validity and
        # APAS-certified-delta metrics.
        fallback_threshold = float(torch.sigmoid(torch.tensor(max_margin)).item())
        fallback = generate_wachter_ce(
            model,
            sources,
            fallback_threshold,
            steps,
            learning_rate,
            proximity_weight,
            validity_weight,
        )
        output[~resolved] = fallback[~resolved]
    return output


def generate_ensemble_wachter_ce(
    models: list[TabularMLP],
    sources: torch.Tensor,
    threshold: float,
    steps: int,
    learning_rate: float,
    proximity_weight: float,
    validity_weight: float = 20.0,
    aggregation: str = "worst",
) -> torch.Tensor:
    """Model-robust Wachter-style CE over a calibration model ensemble.

    This is a lightweight proxy for recent robust recourse methods that seek
    counterfactuals remaining valid under model changes. The CE is optimized
    against the current model plus proxy models trained under calibration
    shifts, using either worst-case or average validity loss across the
    ensemble.
    """
    for model in models:
        model.eval()
    source = sources.detach()
    candidate = source.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=learning_rate)
    result = source.clone()
    best_score = torch.full((len(source),), float("inf"))
    completed = torch.zeros(len(source), dtype=torch.bool)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = torch.stack(
            [torch.sigmoid(model(candidate)) for model in models], dim=0
        )
        losses = torch.relu(threshold - probabilities).pow(2)
        if aggregation == "worst":
            validity_loss = losses.max(dim=0).values
        elif aggregation == "mean":
            validity_loss = losses.mean(dim=0)
        else:
            raise ValueError(f"Unknown ensemble aggregation: {aggregation}")
        distance = (candidate - source).pow(2).mean(dim=1)
        loss = (validity_weight * validity_loss + proximity_weight * distance).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probabilities = torch.stack(
                [torch.sigmoid(model(candidate)) for model in models], dim=0
            )
            valid = (probabilities >= threshold).all(dim=0)
            improved = valid & (distance < best_score)
            result[improved] = candidate[improved]
            best_score[improved] = distance[improved]
            completed |= valid
    return torch.where(completed[:, None], result, candidate.detach())


def sample_parameter_shift_models(
    model: TabularMLP,
    count: int,
    delta: float,
    seed: int,
) -> list[TabularMLP]:
    """Sample PMS realizations around a base model.

    Following the APΔS/PMS setting, each parameter is sampled uniformly from
    the absolute interval [theta_i - delta, theta_i + delta].
    """
    generator = torch.Generator().manual_seed(seed)
    shifted_models: list[TabularMLP] = []
    for _ in range(count):
        shifted = copy.deepcopy(model)
        shifted.eval()
        with torch.no_grad():
            for parameter in shifted.parameters():
                noise = (
                    torch.rand(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 2.0
                    - 1.0
                )
                parameter.add_(delta * noise)
        shifted_models.append(shifted)
    return shifted_models


@torch.no_grad()
def apas_realization_rate(
    model: TabularMLP,
    samples: torch.Tensor,
    delta: float,
    realization_count: int,
    threshold: float,
    seed: int,
) -> float:
    """Estimate APΔS-style realization pass rate for a batch of CEs."""
    shifted_models = sample_parameter_shift_models(
        model, realization_count, delta, seed
    )
    rates = []
    for shifted in shifted_models:
        probabilities = torch.sigmoid(shifted(samples))
        rates.append((probabilities >= threshold).float())
    return float(torch.stack(rates).mean(dim=0).mean().item())


@torch.no_grad()
def aps_robust_mask(
    model: TabularMLP,
    samples: torch.Tensor,
    delta: float,
    realization_count: int,
    threshold: float,
    seed: int,
) -> torch.Tensor:
    """Return which CEs pass all sampled PMS realizations at a fixed delta."""
    shifted_models = sample_parameter_shift_models(
        model, realization_count, delta, seed
    )
    valid = torch.ones(len(samples), dtype=torch.bool)
    for shifted in shifted_models:
        probabilities = torch.sigmoid(shifted(samples))
        valid &= probabilities >= threshold
    return valid


@torch.no_grad()
def aps_certified_deltas(
    model: TabularMLP,
    samples: torch.Tensor,
    realization_count: int,
    threshold: float,
    max_delta: float,
    search_steps: int,
    seed: int,
) -> torch.Tensor:
    """Estimate per-CE maximum PMS radius using APS-style binary search.

    This certification is independent of LIFT's lens calibration: it samples
    plausible model shifts around the current classifier and searches the
    largest delta for which each CE remains valid over sampled realizations.
    """
    base_valid = torch.sigmoid(model(samples)) >= threshold
    low = torch.zeros(len(samples))
    high = torch.full((len(samples),), float(max_delta))
    active = base_valid.clone()
    for step in range(search_steps):
        mid = (low + high) / 2.0
        rounded_mid = mid.round(decimals=8)
        unique_mid = torch.unique(rounded_mid[active])
        robust = torch.zeros(len(samples), dtype=torch.bool)
        for value in unique_mid:
            mask = active & (rounded_mid == value)
            if not bool(mask.any()):
                continue
            checked = aps_robust_mask(
                model,
                samples[mask],
                float(value.item()),
                realization_count,
                threshold,
                seed + 1009 * step + int(float(value.item()) * 1_000_000),
            )
            robust[mask] = checked
        low[active & robust] = mid[active & robust]
        high[active & ~robust] = mid[active & ~robust]
    low[~base_valid] = 0.0
    return low


def generate_mcce_ce(
    model: TabularMLP,
    sources: torch.Tensor,
    target_references: torch.Tensor,
    threshold: float,
    steps: int,
    realism_weight: float = 0.0,
) -> torch.Tensor:
    """MCCE-style data-driven counterfactual search.

    For each source instance, sample a finite candidate set along line segments
    from the source to empirical target-class references. Among candidates that
    satisfy the requested target probability, choose the one with the smallest
    proximity plus optional target-realism score. This is a lightweight
    tabular analogue of Monte-Carlo counterfactual explanation generators: the
    generator searches a plausible empirical target-side candidate set instead
    of following raw classifier gradients.
    """
    model.eval()
    sources = sources.detach()
    references = target_references.detach()
    alphas = torch.linspace(1 / steps, 1.0, steps)
    output = []
    with torch.no_grad():
        for source in sources:
            candidates = (
                (1.0 - alphas[:, None, None]) * source[None, None, :]
                + alphas[:, None, None] * references[None, :, :]
            ).flatten(0, 1)
            probabilities = torch.sigmoid(model(candidates))
            proximity_score = (candidates - source).pow(2).mean(dim=1)
            target_score = torch.cdist(candidates, references).pow(2).min(dim=1).values
            target_score = target_score / source.numel()
            score = proximity_score + realism_weight * target_score
            valid = probabilities >= threshold
            if bool(valid.any()):
                valid_indices = valid.nonzero(as_tuple=False).flatten()
                selected = valid_indices[score[valid_indices].argmin()]
            else:
                # Fallback to the most target-like candidate if no finite sample
                # crosses the threshold. This keeps the generator total and
                # makes failures visible through current_validity.
                best_probability = probabilities.max()
                tied = probabilities >= best_probability - 1e-8
                tied_indices = tied.nonzero(as_tuple=False).flatten()
                selected = tied_indices[score[tied_indices].argmin()]
            output.append(candidates[selected])
    return torch.stack(output)


def select_aps_certified_candidates(
    model: TabularMLP,
    sources: torch.Tensor,
    candidates: dict[float, torch.Tensor],
    realization_count: int,
    threshold: float,
    max_delta: float,
    search_steps: int,
    target_delta: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build APS baseline CEs by certifying and selecting Wachter candidates."""
    ordered_thresholds = sorted(candidates)
    candidate_tensor = torch.stack([candidates[level] for level in ordered_thresholds])
    delta_tensor = torch.stack(
        [
            aps_certified_deltas(
                model,
                candidates[level],
                realization_count,
                threshold,
                max_delta,
                search_steps,
                seed + index * 10_000,
            )
            for index, level in enumerate(ordered_thresholds)
        ]
    )
    cost_tensor = (candidate_tensor - sources[None]).pow(2).mean(dim=2).sqrt()
    best_delta_index = delta_tensor.argmax(dim=0)
    sample_index = torch.arange(len(sources))
    max_delta_samples = candidate_tensor[best_delta_index, sample_index]

    feasible = delta_tensor >= target_delta
    feasible_cost = cost_tensor.clone()
    feasible_cost[~feasible] = float("inf")
    target_index = feasible_cost.argmin(dim=0)
    no_feasible = torch.isinf(feasible_cost).all(dim=0)
    target_index[no_feasible] = best_delta_index[no_feasible]
    target_samples = candidate_tensor[target_index, sample_index]
    return {
        "aps_selected_max_delta": max_delta_samples,
        f"aps_selected_delta_{target_delta:g}": target_samples,
    }


def select_sources(
    model: TabularMLP, test_x: torch.Tensor, test_y: torch.Tensor, count: int, seed: int
) -> torch.Tensor:
    with torch.no_grad():
        negative = ((test_y == 0) & (torch.sigmoid(model(test_x)) < 0.5)).nonzero(
            as_tuple=False
        ).flatten()
    if len(negative) == 0:
        raise ValueError("No correctly classified negative samples available for CE")
    generator = torch.Generator().manual_seed(seed)
    selected = negative[torch.randperm(len(negative), generator=generator)[:count]]
    return test_x[selected]


def reference_targets(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    count: int,
    seed: int,
    model: TabularMLP | None = None,
    minimum_probability: float = 0.7,
) -> torch.Tensor:
    positive = (train_y == 1).nonzero(as_tuple=False).flatten()
    if len(positive) == 0:
        raise ValueError("No positive target-class samples available for CE")
    if model is not None:
        with torch.no_grad():
            accepted = (
                torch.sigmoid(model(train_x[positive]))
                >= minimum_probability
            )
        filtered = positive[accepted]
        if len(filtered) >= count:
            positive = filtered
    generator = torch.Generator().manual_seed(seed)
    selected = positive[torch.randperm(len(positive), generator=generator)[:count]]
    return train_x[selected]


def interpolation_path(
    sources: torch.Tensor,
    endpoints: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    alphas = torch.linspace(1 / steps, 1, steps)[:, None, None]
    return (1 - alphas) * sources[None] + alphas * endpoints[None]


def stability_baseline(
    model: TabularMLP,
    path: torch.Tensor,
    method: str,
    threshold: float = 0.52,
) -> torch.Tensor:
    if method == "trex_style":
        scores = []
        generator = torch.Generator().manual_seed(29)
        for candidates in path:
            noisy = candidates.unsqueeze(0).expand(16, *candidates.shape)
            noisy = noisy + 0.05 * torch.randn(noisy.shape, generator=generator)
            with torch.no_grad():
                probabilities = torch.sigmoid(
                    model(noisy.flatten(0, 1))
                ).view(16, len(candidates))
            scores.append(probabilities.mean(0) - 2 * probabilities.std(0))
        score = torch.stack(scores)
        acceptable = score >= threshold
        objective = torch.arange(len(path))[:, None].float().expand_as(score).clone()
        objective[~acceptable] = float("inf")
    elif method == "sns_style":
        gradients = []
        probabilities = []
        for candidates in path:
            values = candidates.detach().requires_grad_(True)
            probability = torch.sigmoid(model(values))
            gradient = torch.autograd.grad(probability.sum(), values)[0]
            gradients.append(gradient.norm(dim=1))
            probabilities.append(probability.detach())
        gradient = torch.stack(gradients)
        probability = torch.stack(probabilities)
        objective = (
            torch.arange(len(path))[:, None] / len(path)
            + 0.5 * gradient / gradient.median(dim=0).values.clamp_min(1e-8)
        )
        objective[probability < threshold] = float("inf")
    else:
        raise ValueError(method)
    selected = objective.argmin(dim=0)
    unresolved = torch.isinf(objective).all(dim=0)
    selected[unresolved] = len(path) - 1
    return path[selected, torch.arange(path.shape[1])]
