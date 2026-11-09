import unittest

import torch

from lift_ce.experiments.full_update_adversary_v1 import differentiable_full_update
from lift_ce.experiments.heloc_gradient_sensitive_adversary_v1 import fine_tune
from lift_ce.models.tabular import TabularMLP


class FullUpdateAdversaryTests(unittest.TestCase):
    def test_forward_matches_official_update_across_minibatches(self):
        torch.manual_seed(14)
        model = TabularMLP(3, [5, 4])
        x, y = torch.randn(270, 3), torch.randint(0, 2, (270,)).float()
        delta, q = torch.randn_like(x)*.03, torch.randn(7, 3)
        replay = differentiable_full_update(model, x, y, delta, q, seed=31, epochs=3)
        official = fine_tune(model, x+delta, y, 31, "cpu", epochs=3)
        torch.testing.assert_close(replay, torch.sigmoid(official(q)), atol=1e-6, rtol=1e-5)

    def test_dead_units_have_finite_outer_gradients(self):
        torch.manual_seed(9)
        model = TabularMLP(2, [4])
        with torch.no_grad():
            model.features[0].weight[0].zero_()
            model.features[0].bias[0] = -10.
        x, y = torch.randn(9, 2), torch.randint(0, 2, (9,)).float()
        delta = torch.zeros_like(x, requires_grad=True)
        pred = differentiable_full_update(model, x, y, delta, x, 13, epochs=2)
        gradient = torch.autograd.grad(pred.sum(), delta)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.)


if __name__ == "__main__":
    unittest.main()
