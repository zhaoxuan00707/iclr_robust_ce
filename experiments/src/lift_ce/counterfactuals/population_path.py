"""Finite OT displacement paths with exact empirical endpoint marginals."""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def population_mean(values: torch.Tensor, weights: torch.Tensor | None = None):
    """Expectation under uniform samples or explicit probability masses."""
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    if (values.ndim != 1 or weights.shape != values.shape
            or not torch.isfinite(weights).all() or (weights < 0).any()
            or not torch.isclose(weights.sum(), weights.new_tensor(1.), atol=1e-6)):
        raise ValueError("weights must be finite probability masses matching the values")
    return (values * weights).sum()


@dataclass
class PopulationPath:
    """Sparse coupling pushforward, with one weighted atom per transport edge.

    Unlike a barycentric projection, splitting source mass retains both
    marginals when the source and target populations have different sizes.
    """

    base: torch.Tensor
    target: torch.Tensor
    source_index: torch.Tensor
    target_index: torch.Tensor
    weights: torch.Tensor

    def at(self, lam: float) -> torch.Tensor:
        if not 0 <= lam <= 1:
            raise ValueError("lambda must be in [0, 1]")
        return (1-lam)*self.base[self.source_index] + lam*self.target[self.target_index]

    def state_dict(self):
        return dict(vars(self))


@dataclass
class WaypointPopulationPath:
    """Two-segment coupling path through training-derived target waypoints."""

    population: PopulationPath
    waypoints: torch.Tensor

    def __post_init__(self):
        if (self.waypoints.shape != self.population.target.shape
                or not torch.isfinite(self.waypoints).all()):
            raise ValueError("waypoints must match the finite target support")

    @property
    def weights(self):
        return self.population.weights

    def at(self, lam: float):
        if not 0 <= lam <= 1:
            raise ValueError("lambda must be in [0, 1]")
        p = self.population
        middle = self.waypoints[p.target_index]
        if lam <= .5:
            return (1-2*lam)*p.base[p.source_index] + 2*lam*middle
        return (2-2*lam)*middle + (2*lam-1)*p.target[p.target_index]


def empirical_ot_path(base: torch.Tensor, target: torch.Tensor) -> PopulationPath:
    """Solve balanced squared-Euclidean OT and retain its sparse coupling.

    HiGHS solves the finite transport LP; this is exact empirical OT up to
    numerical tolerance, not a certificate for the adversarial supremum.
    """
    if (base.ndim != 2 or target.ndim != 2 or base.shape[1] != target.shape[1]
            or len(base) == 0 or len(target) == 0
            or not torch.isfinite(base).all() or not torch.isfinite(target).all()):
        raise ValueError("OT endpoints must be nonempty finite feature matrices")
    n, m = len(base), len(target)
    columns = np.arange(n*m)
    source_index = np.repeat(np.arange(n), m)
    target_index = np.tile(np.arange(m), n)
    constraints = coo_matrix((np.ones(2*n*m),
        (np.concatenate([source_index, n+target_index]), np.tile(columns, 2))),
        shape=(n+m, n*m)).tocsr()
    cost = torch.cdist(base.detach().cpu().double(), target.detach().cpu().double()).square()
    solution = linprog(cost.numpy().ravel(), A_eq=constraints,
        b_eq=np.concatenate([np.full(n, 1/n), np.full(m, 1/m)]),
        bounds=(0, None), method="highs")
    if not solution.success:
        raise RuntimeError(f"empirical OT failed: {solution.message}")
    active = solution.x > 0
    masses = solution.x[active]
    rows, cols = source_index[active], target_index[active]
    if (not np.allclose(np.bincount(rows, weights=masses, minlength=n), 1/n, atol=1e-8)
            or not np.allclose(np.bincount(cols, weights=masses, minlength=m), 1/m, atol=1e-8)):
        raise RuntimeError("OT solution failed its endpoint marginal check")
    return PopulationPath(base.detach().cpu(), target.detach().cpu(),
        torch.from_numpy(rows), torch.from_numpy(cols), torch.from_numpy(masses).to(base.dtype))
