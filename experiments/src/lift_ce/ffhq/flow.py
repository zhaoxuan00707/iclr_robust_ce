"""Exact original FFHQ progressive LeapFactual inference functions."""
import torch
from torchdyn.core import NeuralODE

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, c, step_size=1.0):
        super().__init__()
        self.model = model
        self.c = c
        self.step_size = step_size

    def forward(self, t, x, *args, **kwargs):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return self.step_size * self.model(t, x, self.c)

@torch.no_grad()
def leap(model, source_points, source_y, target_y, gamma_lift=1.0, gamma_land=1.0, n_steps=10, t_lower=0.0, t_upper=1.0):
    if gamma_lift > 0:
        node_lift = NeuralODE(torch_wrapper(model, source_y, gamma_lift), solver='dopri5', sensitivity='adjoint', atol=0.0001, rtol=0.0001)
        traj_lift = node_lift.trajectory(source_points, t_span=torch.linspace(t_upper, t_lower, n_steps))
    land_start_points = traj_lift[-1].detach().clone() if gamma_lift > 0 else source_points
    node_land = NeuralODE(torch_wrapper(model, target_y, gamma_land), solver='dopri5', sensitivity='adjoint', atol=0.0001, rtol=0.0001)
    traj_land = node_land.trajectory(land_start_points, t_span=torch.linspace(t_lower, t_upper, n_steps))
    return traj_land

@torch.no_grad()
def progressive_leap(model, source_points, classifier, target_y, gamma_lift=0.1, gamma_land=0.1, max_itr=10, n_steps=10, t_lower=0.0, t_upper=1.0):
    traj_list = []
    lift_start_points = source_points.detach().clone()
    for itr in range(max_itr):
        source_y = classifier(lift_start_points)
        print(f'Processing itration {itr}, predicted label {source_y.detach().cpu().numpy()}')
        traj_land = leap(model, lift_start_points, source_y, target_y, gamma_lift=gamma_lift, gamma_land=gamma_land, n_steps=n_steps, t_lower=t_lower, t_upper=t_upper)
        lift_start_points = traj_land[-1].detach().clone()
        traj_list.append(lift_start_points)
    print(f'Finish, predicted label {classifier(lift_start_points).detach().cpu().numpy()}')
    return traj_list
