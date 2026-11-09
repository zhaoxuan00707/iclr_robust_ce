import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from lift_ce.counterfactuals.population_path import empirical_ot_path, WaypointPopulationPath
from lift_ce.counterfactuals.robust_waypoints import ModelBank, fit_waypoints
from lift_ce.experiments import robust_waypoint_path_v1 as experiment


def model(weight, bias):
    result = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        result[0].weight.fill_(weight)
        result[0].bias.fill_(bias)
    return result


class RobustWaypointTests(unittest.TestCase):
    def test_endpoints_and_target_mass_are_preserved(self):
        population = empirical_ot_path(torch.tensor([[-1.], [0.]]), torch.tensor([[1.], [2.], [3.]]))
        path = WaypointPopulationPath(population, population.target+2.)
        torch.testing.assert_close(path.at(0.), population.at(0.))
        torch.testing.assert_close(path.at(1.), population.at(1.))
        torch.testing.assert_close(path.at(.5), path.waypoints[population.target_index])
        torch.testing.assert_close(path.weights, population.weights)

    def test_bank_risks_average_before_maximizing(self):
        bank = ModelBank([model(1., 0.), model(-1., 0.)], "cpu")
        points = torch.tensor([[-2.], [2.]])
        risks = bank.risks(points, chunk=1)
        expected = torch.relu(.5-torch.sigmoid(points[:, 0])).mean()
        torch.testing.assert_close(risks, expected.expand(2))

    def test_training_improves_objective_within_displacement_bound(self):
        target = torch.tensor([[.1], [.2]])
        bank = ModelBank([model(1., -1.), model(1., -.5)], "cpu")
        result, objective, history = fit_waypoints(bank, model(1., 0.), target,
            torch.tensor([-2.]), torch.tensor([2.]), radius=1.5, steps=50)
        self.assertLess(objective, history[0]["training_objective"])
        self.assertTrue(((result-target).abs() <= 1.50001).all())
        self.assertTrue(((result >= -2.) & (result <= 2.)).all())

    def test_selection_compares_paths_on_same_bank_without_final_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/"run"
            out.mkdir()
            original = root/"original"
            original.mkdir()
            p = empirical_ot_path(torch.tensor([[-1.], [-.5]]), torch.tensor([[1.], [2.]]))
            population = {"path": p.state_dict(), "source": p.base}
            torch.save(population, root/"cal_population.pt")
            (root/"cal_population.sha256").write_text(experiment.protocol.file_hash(root/"cal_population.pt"))
            torch.save({"waypoints": p.target+2.}, out/"frozen_path.pt")
            digest = experiment.protocol.file_hash(out/"frozen_path.pt")
            (out/"path_manifest.json").write_text(json.dumps({"path_sha256": digest}))
            torch.save(model(1., -.5).state_dict(), out/"model.pt")
            record = {"attack_id": "training_attack", "radius_ratio": 1.,
                      "model_path": str(out/"model.pt"), "model_sha256": experiment.protocol.file_hash(out/"model.pt")}
            (out/"bank_manifest.json").write_text(json.dumps({"path_sha256": digest, "attacks": [record]}))
            cal = root/"update_alignment_feasibility_v1/matched_models"
            cal.mkdir(parents=True)
            torch.save(model(1., -.5).state_dict(), cal/"model0.pt")
            pd.DataFrame({"path": ["unused_scratch_model"]}).to_csv(original/"future_calibration_manifest.csv", index=False)
            meta = {"original_instance": str(original), "lambdas": [0., .5, 1.], "ratios": [1.]}
            context = (root, meta, {"x": p.target}, model(1., 0.), out, {}, out)
            with patch.object(experiment, "context", return_value=context), patch.object(
                    experiment.protocol.legacy, "load_model", side_effect=lambda *a: model(1., -.5)):
                experiment.select(SimpleNamespace(device="cpu"))
            selected = experiment.verified_selection(root, out)
            decisions = {row["path"]: row["lambda"] for row in selected["selected"]}
            self.assertEqual(decisions, {"learned_waypoint": .5, "linear_control": 1.})
            rows = pd.read_csv(out/"calibration_attack_cross_evaluation.csv")
            self.assertEqual(len(rows), 6)
            self.assertEqual(set(rows.attack_id), {"training_attack"})


if __name__ == "__main__":
    unittest.main()
