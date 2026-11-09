import unittest
import torch
from lift_ce.counterfactuals.path_level_solver import (
    path_state, weighted_path_risk, audit_path, solve_path, straight_derived_initial)
from lift_ce.counterfactuals.robust_waypoints import ModelBank


def model(slope=0., bias=10.):
    net = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad(): net[0].weight.fill_(slope); net[0].bias.fill_(bias)
    return net.requires_grad_(False)


class PathLevelTests(unittest.TestCase):
    def test_split_gluing_endpoints_and_factual_mass(self):
        source = torch.tensor([[0.], [2.]])
        target = torch.tensor([[3.], [4.], [5.]])
        controls = torch.stack([target-2, target-1])
        state = path_state(source, controls, target, [0., .2, .5, 1.])
        w = state['weights']
        torch.testing.assert_close(state['q'][0], source[state['source_index']])
        torch.testing.assert_close(state['q'][-1], target[state['target_index']])
        self.assertAlmostEqual(float(w.sum()), 1.)
        torch.testing.assert_close(torch.bincount(state['source_index'], weights=w), torch.full((2,), .5, dtype=torch.double))
        torch.testing.assert_close(torch.bincount(state['target_index'], weights=w), torch.full((3,), 1/3, dtype=torch.double))

    def test_zero_length_legs_and_knots(self):
        source, target = torch.tensor([[0.]]), torch.tensor([[2.]])
        controls = torch.tensor([[[.5]], [[.5]]])
        state = path_state(source, controls, target, [0., .25, .5, 1.])
        torch.testing.assert_close(state['q'].flatten(), torch.tensor([0., .5, 1., 2.]))
        constant = path_state(source, torch.stack([source, source]), source, [0., .5, 1.])
        self.assertEqual(constant['total_length'], 0.)
        self.assertTrue(torch.isfinite(constant['q']).all())

    def test_population_reduction_and_streamed_gradient(self):
        bank = ModelBank([model(4, 0), model(-4, 0)], 'cpu')
        q = torch.tensor([[[-1.], [1.]], [[.2], [.7]]], requires_grad=True)
        mass = torch.tensor([.3, .7], dtype=torch.double)
        smooth, true, grad = weighted_path_risk(bank, q, mass, chunk=1)
        score = torch.sigmoid(bank.logits(q.reshape(-1, 1))).reshape(2, 2, 2)
        true_expected = (torch.relu(.5-score).double()*mass).sum(-1).amax(0)
        soft = (torch.nn.functional.softplus(50*(.5-score)).double()/50*mass).sum(-1)
        objective = (torch.logsumexp(100*soft, 0)/100).mean()
        torch.testing.assert_close(true, true_expected)
        torch.testing.assert_close(smooth.mean(), objective)
        torch.testing.assert_close(grad, torch.autograd.grad(objective, q)[0])
        self.assertLess(float(true[0]), float((torch.relu(.5-score[:, 0]).amax(0)*mass).sum()))

    def test_arclength_gradient_matches_finite_difference(self):
        source, target = torch.tensor([[0., 0.]], dtype=torch.double), torch.tensor([[2., 0.]], dtype=torch.double)
        controls = torch.tensor([[[.3, .7]], [[1.3, .4]]], dtype=torch.double, requires_grad=True)
        loss = path_state(source, controls, target, [.17, .61])['q'].square().sum()
        gradient = torch.autograd.grad(loss, controls)[0]
        plus, minus = controls.detach().clone(), controls.detach().clone()
        plus[0, 0, 1] += 1e-5; minus[0, 0, 1] -= 1e-5
        diff = (path_state(source, plus, target, [.17, .61])['q'].square().sum()-path_state(source, minus, target, [.17, .61])['q'].square().sum())/2e-5
        self.assertAlmostEqual(float(gradient[0, 0, 1]), float(diff), places=6)

    def test_valid_knots_do_not_hide_invalid_interior(self):
        class Bowl(torch.nn.Module):
            def forward(self, x): return x[:, 0].square()-.25
        source, target = torch.tensor([[-1.]]), torch.tensor([[1.]])
        controls = torch.stack([target, target])
        state = path_state(source, controls, target, [0., .5, 1.])
        audit = audit_path(state, controls, Bowl(), torch.tensor([-2.]), torch.tensor([2.]), 3., 1e-6)
        self.assertFalse(audit['feasible'])
        self.assertEqual(audit['maximum_grid_invalid_mass'], 1.)

    def test_solver_retains_true_feasible_and_rejects_heldout_budget(self):
        current = model()
        bank = ModelBank([model(1, 0)], 'cpu')
        source, target = torch.tensor([[0.]]), torch.tensor([[1.]])
        controls = straight_derived_initial(source, target)
        settings = dict(steps=2, learning_rate=.01, beta_h=50., beta_m=100., rho_initial=1., dual_interval=1, stall_interval=2)
        args = dict(bank=bank, current=current, target=target, initial=controls, budget=1., D_train=1., lower=torch.tensor([-2.]), upper=torch.tensor([2.]), lambdas=[0., .5, 1.], settings=settings)
        result = solve_path(source=source, **args)
        self.assertEqual(result['status'], 'feasible')
        self.assertTrue(result['candidate']['audit']['feasible'])
        rejected = solve_path(source=torch.tensor([[-1.]]), **args)
        self.assertEqual(rejected['status'], 'infeasible_endpoints')
        self.assertIsNone(rejected['candidate'])

    def test_weighted_same_coupling_speed_is_bounded_by_total_length(self):
        source = torch.tensor([[0., 0.], [1., .3]])
        target = torch.tensor([[3., 0.], [3., 1.], [2., 1.]])
        controls = torch.stack([target-1., target-.3])
        grid = [i/20 for i in range(21)]
        state = path_state(source, controls, target, grid)
        increments = state['q'][1:]-state['q'][:-1]
        speeds = ((increments.double().square().sum(-1)*state['weights']).sum(-1)).sqrt()*20
        self.assertLessEqual(float(speeds.max()), state['total_length']+1e-5)

    def test_no_feasible_iterate_is_not_returned_as_an_incumbent(self):
        class Bowl(torch.nn.Module):
            def forward(self, x): return x[:, 0].square()-.25
        source, target = torch.tensor([[-1.]]), torch.tensor([[1.]])
        result = solve_path(ModelBank([model()], 'cpu'), Bowl(), source, target,
                            torch.stack([target, target]), 2., 2., torch.tensor([-2.]),
                            torch.tensor([2.]), [0., .5, 1.],
                            dict(steps=0, learning_rate=.01, beta_h=50., beta_m=100.,
                                 rho_initial=1., dual_interval=1, stall_interval=2))
        self.assertEqual(result['status'], 'no_feasible_iterate')
        self.assertIsNone(result['candidate'])
        self.assertFalse(result['trace'][0]['feasible'])
