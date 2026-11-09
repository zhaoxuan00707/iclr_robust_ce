"""Regression checks for the isolated single-total-budget solver."""
import unittest
from unittest.mock import patch
import torch
from lift_ce.counterfactuals.single_budget_solver import solve_control, geometry_record, bank_value_gradient
from lift_ce.counterfactuals.robust_waypoints import ModelBank


def model(slope, bias):
    net = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        net[0].weight.fill_(slope)
        net[0].bias.fill_(bias)
    return net


class SingleBudgetTests(unittest.TestCase):
    def test_total_budget_is_not_an_endpoint_displacement_bound(self):
        source, target = torch.tensor([[0.]]), torch.tensor([[1.]])
        _, _, audit = geometry_record(source, torch.tensor([[1.2]]), target,
                                     1., 1.3, model(0, 10), torch.tensor([-10.]), torch.tensor([10.]))
        self.assertAlmostEqual(audit['total_length'], 1.4, places=6)
        self.assertAlmostEqual(audit['slack'], -.1, places=6)
        self.assertGreater(audit['total_length'], audit['B_geo'])

    def test_true_objective_reduces_population_before_model(self):
        bank = ModelBank([model(20, 0), model(-20, 0)], 'cpu')
        q = torch.tensor([[-1.], [1.]], requires_grad=True)
        _, risk, grad = bank_value_gradient(bank, q, chunk=1)
        self.assertAlmostEqual(risk, .25, places=6)
        score = torch.sigmoid(bank.logits(q))
        smooth = torch.logsumexp(100 * (torch.nn.functional.softplus(50 * (.5-score))/50).double().mean(1), 0)/100
        torch.testing.assert_close(grad, torch.autograd.grad(smooth, q)[0])

    def test_exact_feasible_nested_incumbents_and_unit_budget(self):
        bank = ModelBank([model(1, -2)], 'cpu')
        current = model(0, 10)
        source, target = torch.tensor([[0.]]), torch.tensor([[1.]])
        settings = {'steps':20, 'learning_rate':.03, 'beta_h':50., 'beta_m':100.,
                    'rho_initial':1., 'dual_interval':5, 'stall_interval':20,
                    'feasibility_relative_tolerance':1e-6}
        pool = []
        best = []
        for budget in [1., 1.2, 1.4]:
            candidates, _ = solve_control(bank, current, source, target, target, 1., budget,
                                          torch.tensor([-10.]), torch.tensor([10.]), settings)
            self.assertTrue(all(c['total_length'] <= budget + 1e-6 and c['current_invalid_mass']==0 for c in candidates))
            pool.extend(candidates)
            best.append(min(c['true_bank_hinge'] for c in pool if c['total_length'] <= budget+1e-6))
        self.assertGreater(best[0], best[-1])
        self.assertTrue(all(b <= a for a, b in zip(best, best[1:])))

    def test_reference_cpu_validity_precedes_incumbent_selection(self):
        bank = ModelBank([model(1, -2)], 'cpu')
        # Optimization model accepts all points; canonical reference rejects
        # target-near points below 1.01. No such candidate may be retained.
        settings = {'steps':0, 'learning_rate':.03, 'beta_h':50., 'beta_m':100.,
                    'rho_initial':1., 'dual_interval':5, 'stall_interval':20,
                    'feasibility_relative_tolerance':1e-6}
        with patch('lift_ce.counterfactuals.single_budget_solver.copy.deepcopy', return_value=model(1, -1.01)):
            candidates, _ = solve_control(bank, model(0, 10), torch.tensor([[0.]]), torch.tensor([[1.]]),
                                          torch.tensor([[1.]]), 1., 1.2, torch.tensor([-10.]), torch.tensor([10.]), settings)
        self.assertEqual(candidates, [])


if __name__ == '__main__':
    unittest.main()
