from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from .config import LiftCEConfig
from .models.vision import ConvAutoencoder


def representations(
    samples: torch.Tensor,
    model: nn.Module,
    space: str,
    device: torch.device,
) -> torch.Tensor:
    if space == "pixel":
        return samples.flatten(1).to(device)
    if space == "classifier_embedding":
        model.eval()
        with torch.no_grad():
            _, embedding = model(samples.to(device), return_embedding=True)
        return embedding
    raise ValueError(f"Unsupported OT space: {space}")


def squared_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2).pow(2)


def sinkhorn_coupling(
    cost: torch.Tensor,
    regularization: float,
    iterations: int,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Balanced entropic OT with cost-scale normalization.

    ``regularization`` is interpreted relative to the median positive cost.
    This avoids severe underflow and slow marginal convergence when classifier
    embeddings have a seed-dependent numerical scale.
    """
    n, m = cost.shape
    positive = cost[cost > 0]
    scale = positive.median() if positive.numel() else cost.new_tensor(1.0)
    normalized_cost = cost / scale.clamp_min(1e-12)
    log_a = torch.full(
        (n,), -torch.log(cost.new_tensor(float(n))), device=cost.device
    )
    log_b = torch.full(
        (m,), -torch.log(cost.new_tensor(float(m))), device=cost.device
    )
    log_kernel = -normalized_cost / max(regularization, 1e-8)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    coupling = None
    for step in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        if step % 10 == 9 or step == iterations - 1:
            coupling = torch.exp(
                log_u[:, None] + log_kernel + log_v[None, :]
            )
            row_error = (coupling.sum(dim=1) - torch.exp(log_a)).abs().max()
            column_error = (
                coupling.sum(dim=0) - torch.exp(log_b)
            ).abs().max()
            if max(row_error.item(), column_error.item()) < tolerance:
                break
    if coupling is None:
        coupling = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    return coupling / coupling.sum().clamp_min(1e-12)


def approximate_w2(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    config: LiftCEConfig,
    device: torch.device,
) -> float:
    x_repr = representations(x, model, config.ot_space, device)
    y_repr = representations(y, model, config.ot_space, device)
    cost = squared_cost(x_repr, y_repr)
    coupling = sinkhorn_coupling(
        cost, config.sinkhorn_reg, config.sinkhorn_iterations
    )
    return float(torch.sqrt((coupling * cost).sum().clamp_min(0)).item())


@dataclass
class RefinementResult:
    samples: torch.Tensor
    history: list[dict[str, float]]


def _exact_uniform_w2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Exact empirical W2 for equally weighted supports of equal size."""
    if len(x) != len(y):
        raise ValueError("Exact uniform W2 requires equal support sizes")
    cost = squared_cost(x, y)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    selected = cost[
        torch.as_tensor(rows, device=cost.device),
        torch.as_tensor(columns, device=cost.device),
    ]
    return float(torch.sqrt(selected.mean().clamp_min(0)).item())


def _quantile(values: list[float], coverage: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, min(max(coverage, 0.0), 1.0)).item())


def calibrated_hungarian_refinements(
    counterfactuals: torch.Tensor,
    targets: torch.Tensor,
    proxy_targets: list[torch.Tensor],
    autoencoder: ConvAutoencoder,
    config: LiftCEConfig,
    device: torch.device,
) -> dict[str, RefinementResult]:
    """One-shot calibrated exact-OT refinements.

    Returns both the profiled-bound choice and a direct proxy-minimax choice
    over the same Wasserstein barycentric family.
    """
    z = counterfactuals.detach().to(device)
    n = len(z)
    if len(targets) < n or any(len(proxy) < n for proxy in proxy_targets):
        raise ValueError("Calibration requires at least n samples per support")
    autoencoder.eval()
    with torch.no_grad():
        state = autoencoder.encode(z)
        target_state = autoencoder.encode(targets[:n].to(device))
        proxy_states = [
            autoencoder.encode(proxy[:n].to(device)) for proxy in proxy_targets
        ]

    cost = squared_cost(state, target_state)
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
    match = torch.empty(n, dtype=torch.long, device=device)
    match[torch.as_tensor(rows, device=device)] = torch.as_tensor(
        columns, device=device
    )
    matched_target = target_state[match]
    distance = float(
        torch.sqrt((state - matched_target).pow(2).sum(dim=1).mean()).item()
    )

    delta_values = [_exact_uniform_w2(proxy, state) for proxy in proxy_states]
    epsilon_values = [
        _exact_uniform_w2(proxy, target_state) for proxy in proxy_states
    ]
    delta = _quantile(delta_values, config.calibration_coverage)
    epsilon = _quantile(epsilon_values, config.calibration_coverage)
    if epsilon + delta < distance:
        gap = distance - (epsilon + delta)
        epsilon += gap / 2
        delta += gap / 2

    best_bound: tuple[float, float, float, float] | None = None
    for beta in config.bound_beta_grid:
        for gamma0 in config.bound_gamma_grid:
            for gamma1 in config.bound_gamma_grid:
                a = (1.0 - beta) * (1.0 + gamma0)
                b = beta * (1.0 + gamma1)
                k = (
                    (1.0 - beta) * (1.0 + 1.0 / gamma0) * delta**2
                    + beta * (1.0 + 1.0 / gamma1) * epsilon**2
                )
                certificate = a * b / (a + b) * distance**2 + k
                candidate = (certificate, b / (a + b), a, b)
                if best_bound is None or candidate[0] < best_bound[0]:
                    best_bound = candidate
    assert best_bound is not None
    bound_certificate, bound_lambda, bound_a, bound_b = best_bound

    best_proxy: tuple[float, float] | None = None
    for lam in config.calibration_lambdas:
        center = (1.0 - lam) * state + lam * matched_target
        worst = max(_exact_uniform_w2(center, proxy) ** 2 for proxy in proxy_states)
        candidate = (worst, float(lam))
        if best_proxy is None or candidate[0] < best_proxy[0]:
            best_proxy = candidate
    assert best_proxy is not None
    proxy_risk, proxy_lambda = best_proxy

    results: dict[str, RefinementResult] = {}
    for selector, lam, score in (
        ("bound", bound_lambda, bound_certificate),
        ("proxy_minimax", proxy_lambda, proxy_risk),
    ):
        center = (1.0 - lam) * state + lam * matched_target
        for eta in config.refinement_depths:
            robust_state = (1.0 - eta) * state + eta * center
            with torch.no_grad():
                decoded_before = autoencoder.decode(state)
                decoded_after = autoencoder.decode(robust_state)
                samples = (z + decoded_after - decoded_before).clamp(0.0, 1.0)
            effective_depth = eta * lam
            certificate_before = bound_b * distance**2
            certificate_after = (
                bound_a * (effective_depth * distance) ** 2
                + bound_b * ((1.0 - effective_depth) * distance) ** 2
            )
            history = [
                {
                    "iteration": 1.0,
                    "calibrated_lambda": lam,
                    "eta": float(eta),
                    "effective_depth": effective_depth,
                    "lens_epsilon": epsilon,
                    "lens_delta": delta,
                    "distance_to_target": distance,
                    "selection_score": score,
                    "certificate_before_q_terms": certificate_before,
                    "certificate_after_q_terms": certificate_after,
                    "certificate_decreased": float(
                        certificate_after <= certificate_before + 1e-9
                    ),
                }
            ]
            name = f"calibrated_{selector}_hungarian_eta_{eta:g}"
            results[name] = RefinementResult(samples.cpu(), history)
    return results


def refine_counterfactuals(
    counterfactuals: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    autoencoder: ConvAutoencoder,
    config: LiftCEConfig,
    method: str,
    device: torch.device,
) -> RefinementResult:
    """Refine CEs in pixel or autoencoder latent space."""
    z = counterfactuals.detach().to(device)
    targets = targets.detach().to(device)
    history: list[dict[str, float]] = []

    if method == "base_ce":
        return RefinementResult(z.cpu(), history)

    latent_method = method.startswith("latent_") or method.startswith("exact_lens_")
    if latent_method:
        autoencoder.eval()
        with torch.no_grad():
            state = autoencoder.encode(z)
            target_state = autoencoder.encode(targets)
    else:
        state = z
        target_state = targets

    fixed_matches: torch.Tensor | None = None
    lens_epsilon: float | None = None
    lens_delta: float | None = None
    exact_targets: torch.Tensor | None = None
    exact_target_state: torch.Tensor | None = None
    if method == "exact_lens_hungarian_lift_ce":
        if not latent_method:
            raise ValueError("Exact lens transport requires latent space")
        if len(target_state) < len(state):
            raise ValueError("Exact assignment requires at least as many targets as CEs")
        # Define a fixed, equal-mass empirical target measure. Hungarian then
        # yields an exact Monge map between two n-point uniform measures.
        exact_targets = targets[: len(state)]
        exact_target_state = target_state[: len(state)]
        initial_rows, initial_columns = linear_sum_assignment(
            squared_cost(state, exact_target_state).detach().cpu().numpy()
        )
        initial_match = torch.empty(len(state), dtype=torch.long, device=device)
        initial_match[torch.as_tensor(initial_rows, device=device)] = torch.as_tensor(
            initial_columns, device=device
        )
        initial_squared = (
            state - exact_target_state[initial_match]
        ).pow(2).sum(dim=1)
        initial_distance = torch.sqrt(initial_squared.mean()).item()
        lens_epsilon = (
            config.lens_epsilon_initial_ratio * initial_distance
        )
        lens_delta = config.lens_delta_initial_ratio * initial_distance
        if lens_epsilon + lens_delta < initial_distance:
            raise ValueError("Initial lens is empty: epsilon + delta < W2")

    if method == "latent_hungarian_lift_ce":
        initial_cost = squared_cost(state, target_state)
        rows, columns = linear_sum_assignment(initial_cost.detach().cpu().numpy())
        fixed_matches = torch.empty(len(state), dtype=torch.long, device=device)
        fixed_matches[torch.as_tensor(rows, device=device)] = torch.as_tensor(
            columns, device=device
        )
        if len(rows) < len(state):
            assigned = torch.zeros(len(state), dtype=torch.bool, device=device)
            assigned[torch.as_tensor(rows, device=device)] = True
            fixed_matches[~assigned] = initial_cost[~assigned].argmin(dim=1)

    for iteration in range(config.iterations):
        if method == "exact_lens_hungarian_lift_ce":
            exact_cost = squared_cost(state, exact_target_state)
            rows, columns = linear_sum_assignment(
                exact_cost.detach().cpu().numpy()
            )
            matched = torch.empty(len(state), dtype=torch.long, device=device)
            matched[torch.as_tensor(rows, device=device)] = torch.as_tensor(
                columns, device=device
            )
            destination_target = exact_target_state[matched]
            distance_to_target = torch.sqrt(
                (state - destination_target).pow(2).sum(dim=1).mean()
            ).item()
            if distance_to_target <= 1e-12:
                break
            lower = max(0.0, 1.0 - lens_epsilon / distance_to_target)
            upper = min(1.0, lens_delta / distance_to_target)
            if lower > upper + 1e-10:
                raise RuntimeError("Current geodesic does not intersect the fixed lens")
            alpha = min(max(0.5, lower), upper)
            center = (1.0 - alpha) * state + alpha * destination_target
            previous = z.clone()
            with torch.no_grad():
                decoded_before = autoencoder.decode(state)
                state = (1.0 - config.eta) * state + config.eta * center
                decoded_after = autoencoder.decode(state)
                z = (z + decoded_after - decoded_before).clamp(0.0, 1.0)
            movement = torch.sqrt(
                (z - previous).flatten(1).pow(2).sum(dim=1).mean()
            ).item()
            center_to_current = alpha * distance_to_target
            center_to_target = (1.0 - alpha) * distance_to_target
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "movement_rms": movement,
                    "distance_to_target": distance_to_target,
                    "lens_epsilon": lens_epsilon,
                    "lens_delta": lens_delta,
                    "lens_alpha": alpha,
                    "center_to_current": center_to_current,
                    "center_to_target": center_to_target,
                    "center_feasible": float(
                        center_to_current <= lens_delta + 1e-7
                        and center_to_target <= lens_epsilon + 1e-7
                    ),
                }
            )
            if movement < config.stop_threshold:
                break
            continue

        if latent_method:
            cost = squared_cost(state, target_state)
        else:
            z_repr = representations(z, model, config.ot_space, device)
            target_repr = representations(targets, model, config.ot_space, device)
            cost = squared_cost(z_repr, target_repr)
        coupling = sinkhorn_coupling(
            cost, config.sinkhorn_reg, config.sinkhorn_iterations
        )

        if method == "pixel_soft_lift_ce":
            row_mass = coupling.sum(dim=1, keepdim=True).clamp_min(1e-12)
            destination = (coupling / row_mass) @ targets.flatten(1)
            destination = destination.view_as(z)
        elif method == "latent_soft_lift_ce":
            row_mass = coupling.sum(dim=1, keepdim=True).clamp_min(1e-12)
            destination = (coupling / row_mass) @ target_state
        elif method == "latent_topk_lift_ce":
            k = min(config.top_k, coupling.shape[1])
            values, indices = coupling.topk(k, dim=1)
            normalized = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
            selected = target_state[indices]
            destination = (normalized.unsqueeze(-1) * selected).sum(dim=1)
        elif method == "latent_hungarian_lift_ce":
            destination = target_state[fixed_matches]
        else:
            raise ValueError(f"Unknown refinement method: {method}")

        step = config.eta * config.alpha_barycenter
        previous = z.clone()
        if latent_method:
            with torch.no_grad():
                decoded_before = autoencoder.decode(state)
                state = (1.0 - step) * state + step * destination
                decoded_after = autoencoder.decode(state)
                # Apply only the manifold-informed displacement. Re-decoding
                # the whole current CE introduces an unrelated reconstruction
                # jump that can destroy validity at the first iteration.
                z = (z + decoded_after - decoded_before).clamp(0.0, 1.0)
        else:
            z = ((1.0 - step) * z + step * destination).clamp(0.0, 1.0)
        movement = torch.sqrt(
            (z - previous).flatten(1).pow(2).sum(dim=1).mean()
        ).item()
        distance_to_target = torch.sqrt((coupling * cost).sum()).item()
        epsilon = config.lens_alpha * (1.0 + config.lens_slack) * distance_to_target
        delta = (
            (1.0 - config.lens_alpha)
            * (1.0 + config.lens_slack)
            * distance_to_target
        )
        history.append(
            {
                "iteration": float(iteration + 1),
                "movement_rms": movement,
                "distance_to_target": distance_to_target,
                "lens_epsilon": epsilon,
                "lens_delta": delta,
            }
        )
        if movement < config.stop_threshold:
            break
    return RefinementResult(z.detach().cpu(), history)
