"""Original Galaxy conditional flow and progressive LeapFactual generator."""
import torch
from torchdyn.core import NeuralODE

class MLP_CFM(torch.nn.Module):

    def __init__(self, dim, out_dim=None, w=64, time_varying=False, cond=False, cond_dim=None):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        if cond and cond_dim is None:
            cond_dim = dim
        self.net = torch.nn.Sequential(torch.nn.Linear(dim + (1 if time_varying else 0) + (cond_dim if cond else 0), w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, w), torch.nn.SELU(), torch.nn.Linear(w, out_dim))

    def forward(self, t, x):
        while t.dim() > 1:
            t = t[:, 0]
        if t.dim() == 0:
            t = t[:, None]
        if t.dim() == 1 and t.shape[0] == 1:
            t = t.repeat(x.shape[0])
        return self.net(torch.cat([x, t[:, None]], dim=1))

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
        return self.step_size * self.model(t, torch.cat([x, self.c], dim=-1))

@torch.no_grad()
def leap(model, source_points, source_y, target_y, gamma_lift=1.0, gamma_land=1.0, n_steps=10, t_lower=0.0, t_upper=1.0):
    if gamma_lift > 0.0:
        node_lift = NeuralODE(torch_wrapper(model, source_y, step_size=gamma_lift), solver='dopri5', sensitivity='adjoint', atol=0.0001, rtol=0.0001)
        traj_lift = node_lift.trajectory(source_points, t_span=torch.linspace(t_upper, t_lower, n_steps))
    land_start_points = traj_lift[-1].detach().clone() if gamma_lift > 0.0 else source_points
    node_land = NeuralODE(torch_wrapper(model, target_y, step_size=gamma_land), solver='dopri5', sensitivity='adjoint', atol=0.0001, rtol=0.0001)
    traj_land = node_land.trajectory(land_start_points, t_span=torch.linspace(t_lower, t_upper, n_steps))
    return traj_land

@torch.no_grad()
def progressive_leap(model, source_points, classifier, target_y, gamma_lift=0.1, gamma_land=0.1, max_itr=10, n_steps=10, t_lower=0.0, t_upper=1.0, mute=False):
    traj_list = []
    lift_start_points = source_points
    for itr in range(max_itr):
        source_y = classifier(lift_start_points)
        if not mute:
            print('Processing itration {:03d}, predicted label {}'.format(itr, torch.argmax(source_y, dim=1).detach().cpu().numpy()), end='\r')
        traj_land = leap(model, lift_start_points, source_y, target_y, gamma_lift=gamma_lift, gamma_land=gamma_land, n_steps=n_steps, t_lower=t_lower, t_upper=t_upper)
        lift_start_points = traj_land[-1]
        traj_list.append(lift_start_points)
    return traj_list
