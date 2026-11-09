"""Natural Euclidean transport spline on a frozen empirical OT coupling."""
from dataclasses import dataclass

import torch

from .population_path import PopulationPath, WaypointPopulationPath


def bending_basis(t):
    """Natural three-knot cubic basis; knots are 0, 1/2, 1."""
    t = torch.as_tensor(t)
    if not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError('lambda must be finite and in [0,1]')
    s = torch.minimum(t, 1-t)
    return 3*s-4*s**3


@dataclass
class TransportSplinePath(WaypointPopulationPath):
    """Same endpoints/coupling and target-indexed waypoint as the old path."""

    def at(self, lam: float):
        p = self.population
        z, y = p.base[p.source_index], p.target[p.target_index]
        u = self.waypoints[p.target_index]-(z+y)/2
        return p.at(lam)+bending_basis(lam).to(z)*u


def bend_diagnostics(population: PopulationPath, waypoints, lambdas):
    """Same-coupling distances and conservative speed, not spline W2 equality."""
    p = population
    z, y = p.base[p.source_index].double(), p.target[p.target_index].double()
    u = waypoints[p.target_index].double()-(z+y)/2
    weights = p.weights.double()
    norm = lambda v: float((v.square().sum(-1)*weights).sum().sqrt())
    bend, distance = norm(u), norm(y-z)
    times = torch.tensor(lambdas, dtype=torch.float64)
    points = (1-times[:,None,None])*z+times[:,None,None]*y+bending_basis(times)[:,None,None]*u
    speeds = [norm(points[i+1]-points[i])/float(times[i+1]-times[i]) for i in range(len(times)-1)]
    deviations = [norm(points[i]-((1-t)*z+t*y)) for i,t in enumerate(times)]
    if max(speeds) > distance+3*bend+1e-9:
        raise AssertionError('spline speed bound failed')
    if any(d > abs(float(bending_basis(t)))*bend+1e-9 for t,d in zip(times,deviations)):
        raise AssertionError('same-coupling bending bound failed')
    return {'E_bend':bend, 'D':distance, 'L_path_emp':max(speeds),
            'L_path_bound':distance+3*bend, 'maximum_deviation':max(deviations),
            'spline_integrated_squared_acceleration':48*bend**2,
            'waypoint_velocity_jump_rms':4*bend,
            'piecewise_H2_energy':'infinite if E_bend > 0 (velocity jump); zero otherwise',
            'deviations':deviations}
