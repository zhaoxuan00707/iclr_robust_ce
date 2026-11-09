import unittest
import torch
from lift_ce.ffhq.bottleneck import FrozenContent, ConditionalBank, gradient_basis
from lift_ce.counterfactuals.single_budget_solver import bank_value_gradient

class TestBottleneck(unittest.TestCase):
    def test_energy_rule_and_content_invariance(self):
        torch.manual_seed(71)
        g = torch.randn(20, 16, dtype=torch.double)
        u, v, s, k = gradient_basis(g)
        self.assertGreaterEqual(float(s[:k].square().sum()/s.square().sum()), .95)
        self.assertLess(float(s[:k-1].square().sum()/s.square().sum()), .95)
        factual = torch.randn(5, 16)
        r = FrozenContent(u, factual)
        torch.testing.assert_close(r.realize(r.encode(factual)), factual)
        changed = r.realize(torch.randn(5, k))
        torch.testing.assert_close(changed.double() @ v, factual.double() @ v, atol=2e-7, rtol=1e-6)
        task = torch.randn(5, k, requires_grad=True)
        direction = torch.randn(5, 16)
        got = torch.autograd.grad((r.realize(task)*direction).sum(), task)[0]
        torch.testing.assert_close(got, (direction.double() @ u).float())

    def test_population_max_not_per_person_max(self):
        class Bank:
            count = 2
            def logits(self, x, start=0, end=None):
                return torch.stack([x[:,0], -x[:,0]])[start:end]
        x = torch.tensor([[2.], [-2.]], requires_grad=True)
        _, risk, grad = bank_value_gradient(Bank(), x)
        self.assertAlmostEqual(risk, float(torch.relu(.5-x.sigmoid()).mean()), places=7)
        self.assertTrue(torch.isfinite(grad).all())

if __name__ == '__main__': unittest.main()
