import unittest

import torch

from lift_ce.experiments.update_alignment_feasibility_v1 import (
    FrozenEnsemble, find_joint_witnesses, project_witness,
)


def linear_model(weight, bias):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        model[0].weight.fill_(weight)
        model[0].bias.fill_(bias)
    return model


class JointFeasibilityTests(unittest.TestCase):
    def test_vectorized_logits_match_individual_models(self):
        models = [linear_model(1., -.2), linear_model(-2., .4)]
        points = torch.tensor([[-1.], [.3], [2.]])
        ensemble = FrozenEnsemble(models, "cpu")
        torch.testing.assert_close(ensemble(points), torch.stack([m(points) for m in models]))

    def test_projection_respects_box_and_factual_cost(self):
        source = torch.tensor([[0.], [.5]])
        projected = project_witness(torch.tensor([[10.], [-10.]]), source,
                                    torch.tensor([.2, .3]), torch.tensor([-1.]), torch.tensor([1.]))
        torch.testing.assert_close(projected, torch.tensor([[.2], [.2]]))

    def test_joint_witness_satisfies_every_model(self):
        models = [linear_model(1., -.2), linear_model(1., -.4)]
        source = torch.zeros(2, 1)
        points, margin = find_joint_witnesses(FrozenEnsemble(models, "cpu"), source,
            source, torch.ones(2), torch.tensor([-2.]), torch.tensor([2.]), steps=60)
        self.assertTrue((margin > 0).all())
        self.assertTrue(all((model(points) > 0).all() for model in models))

    def test_conflicting_models_never_produce_false_witness(self):
        models = [linear_model(1., -1.), linear_model(-1., -1.)]
        source = torch.zeros(2, 1)
        _, margin = find_joint_witnesses(FrozenEnsemble(models, "cpu"), source,
            source, torch.ones(2)*2, torch.tensor([-2.]), torch.tensor([2.]), steps=30)
        self.assertTrue((margin < 0).all())


if __name__ == "__main__":
    unittest.main()
