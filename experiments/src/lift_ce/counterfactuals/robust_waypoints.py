"""Training-only intermediate populations for a fixed empirical OT coupling."""
import copy

import torch
from torch.func import functional_call, stack_module_state, vmap


class ModelBank:
    """Frozen classifiers with chunked population-level risk evaluation."""

    def __init__(self, models, device):
        models = [copy.deepcopy(m).to(device).eval().requires_grad_(False) for m in models]
        self.params, self.buffers = stack_module_state(models)
        self.base, self.count = models[0], len(models)

    def logits(self, points, start=0, stop=None):
        stop = self.count if stop is None else stop
        params = {k: v[start:stop] for k, v in self.params.items()}
        buffers = {k: v[start:stop] for k, v in self.buffers.items()}
        return vmap(lambda p, b: functional_call(self.base, (p, b), (points,)))(params, buffers)

    @torch.no_grad()
    def risks(self, points, threshold=.5, weights=None, chunk=32):
        risks = []
        for start in range(0, self.count, chunk):
            losses = torch.relu(threshold-torch.sigmoid(self.logits(points, start, start+chunk)))
            risks.append(losses.mean(1) if weights is None else (losses*weights.to(points)).sum(1))
        return torch.cat(risks)


def fit_waypoints(bank, current, target, lower, upper, radius=3., steps=300, seed=0):
    """Fit a bounded intermediate support, using no calibration observations.

    Path-learning surrogate: max_model(mean_target hinge at .7), plus nominal
    confidence and displacement regularization. Lambda ranking subsequently
    uses the separate original .5 true-hinge objective.
    """
    if radius <= 0 or steps < 0:
        raise ValueError("positive radius and nonnegative steps are required")
    current = copy.deepcopy(current).to(target.device).eval().requires_grad_(False)
    generator = torch.Generator(device=target.device).manual_seed(seed)
    start = target.clone()
    if seed:
        start += .03*torch.randn(target.shape, generator=generator, device=target.device)
    points = start.clamp(lower, upper).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([points], lr=.05)
    best_loss, best = float("inf"), target.detach().clone()
    history = []
    for step in range(steps+1):
        risks = bank.risks(points, threshold=.7)
        worst = int(risks.argmax())
        risk = torch.relu(.7-torch.sigmoid(bank.logits(points, worst, worst+1))).mean()
        nominal = torch.relu(.7-torch.sigmoid(current(points))).mean()
        movement = (points-target).square().sum(1).mean()
        objective = risk+2*nominal+.001*movement
        if not torch.isfinite(objective):
            raise ValueError("nonfinite training waypoint objective")
        value = float(objective.detach())
        if value < best_loss:
            best_loss, best = value, points.detach().clone()
        history.append({"iteration": step, "training_objective": value,
                        "bank_margin_risk": float(risk.detach()),
                        "nominal_margin_risk": float(nominal.detach()),
                        "mean_squared_displacement": float(movement.detach())})
        if step % 25 == 0:
            print(f"seed={seed} step={step} training_objective={value:.6f}", flush=True)
        if step == steps:
            break
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            points.clamp_(lower, upper)
            delta = points-target
            norm = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
            points.copy_(target+delta*torch.minimum(torch.ones_like(norm), radius/norm)[:, None])
    return best.cpu(), best_loss, history
