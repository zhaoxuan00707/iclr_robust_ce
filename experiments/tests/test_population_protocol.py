"""Endpoint, split-isolation, and frozen-bank protocol regression tests."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from lift_ce.counterfactuals.population_path import empirical_ot_path, population_mean
from lift_ce.experiments import heloc_diabetes_bank_protocol_v3 as protocol
from lift_ce.experiments import heloc_diabetes_official_lift_v1 as legacy
from lift_ce.models.tabular import TabularMLP


class PopulationPathTests(unittest.TestCase):
    def test_unequal_size_endpoints_preserve_entire_empirical_distributions(self):
        base = torch.tensor([[-2.], [0.]])
        target = torch.tensor([[1.], [3.], [8.]])
        path = empirical_ot_path(base, target)
        # Nonlinear expectations distinguish coupling interpolation from a
        # barycentric destination that collapses the target distribution.
        self.assertAlmostEqual(float(population_mean(path.at(0.)[:, 0].square(), path.weights)),
                               float(base.square().mean()), places=6)
        self.assertAlmostEqual(float(population_mean(path.at(1.)[:, 0].square(), path.weights)),
                               float(target.square().mean()), places=5)
        source_masses = torch.zeros(len(base)).scatter_add_(0, path.source_index, path.weights)
        target_masses = torch.zeros(len(target)).scatter_add_(0, path.target_index, path.weights)
        torch.testing.assert_close(source_masses, torch.full((2,), .5))
        torch.testing.assert_close(target_masses, torch.full((3,), 1/3))

    def test_weighted_hinge_agrees_with_empirical_endpoint_risk(self):
        path = empirical_ot_path(torch.tensor([[-1.], [0.]]), torch.tensor([[.1], [.3], [.9]]))
        risk = population_mean(torch.relu(.5-path.at(1.)[:, 0]), path.weights)
        self.assertAlmostEqual(float(risk), .2, places=6)

    def test_invalid_weights_and_empty_endpoints_fail(self):
        with self.assertRaises(ValueError):
            population_mean(torch.ones(2), torch.tensor([.4, .4]))
        with self.assertRaises(ValueError):
            empirical_ot_path(torch.empty(0, 1), torch.ones(2, 1))


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.args = SimpleNamespace(output=Path(self.tmp.name)/"new", instance_root=Path(self.tmp.name)/"old",
            dataset="heloc", seed=0, data_home=self.tmp.name, count=2, device="cpu",
            all_grid=True, radius_index=0, lambda_index=0, steps=1,
            generation="first", bank_name="first")
        self.old = legacy.seed_dir(self.args.instance_root, "heloc", 0)
        (self.old/"models").mkdir(parents=True)
        self.model = TabularMLP(1, [64, 32])
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()
            self.model.features[0].weight[0, 0] = 1.
            self.model.features[0].bias[0] = 2.
            self.model.features[2].weight[0, 0] = 1.
            self.model.classifier.weight[0, 0] = 1.
            self.model.classifier.bias[0] = -2.
        self.model.eval()
        self.d = {"train_x": torch.tensor([[-1.], [-.5], [.5], [1.]]),
            "train_y": torch.tensor([0., 0., 1., 1.]), "train_ids": np.arange(4),
            "cal_x": torch.tensor([[-.9], [-.4], [.6]]), "cal_ids": np.arange(4, 7),
            "test_x": torch.tensor([[-.8], [-.3], [.7]]), "test_ids": np.arange(7, 10),
            "feature_names": ["x"]}
        self.common = {"source": self.d["test_x"][:2], "source_ids": torch.tensor([7, 8]),
                       "test_indices": torch.tensor([0, 1]), "b_ref": .1}
        torch.save(self.model.state_dict(), self.old/"models/current.pt")
        torch.save(self.common, self.old/"common.pt")
        torch.save({"counterfactuals": torch.tensor([[.1], [.2]])}, self.old/"lift_original.pt")
        pd.DataFrame({"path": [str(self.old/"models/current.pt")]}).to_csv(
            self.old/"future_calibration_manifest.csv", index=False)
        self.stack.enter_context(patch.object(legacy, "load_common", return_value=(self.old, self.d, self.common, self.model)))
        self.generator = self.stack.enter_context(patch.object(legacy, "generate_wachter_ce",
            side_effect=lambda current, source, *a: torch.full_like(source, .1)))
        self.stack.enter_context(patch.object(legacy, "LAMBDAS", (0., 1.)))
        self.stack.enter_context(patch.object(legacy, "RATIOS", (1.,)))
        self.stack.enter_context(patch.object(legacy, "RESTARTS", 1))
        protocol.prepare(self.args)
        self.root = protocol.root_for(self.args)

    def test_cpu_pipeline_is_split_isolated_and_freezes_before_calibration(self):
        generated_from = self.generator.call_args.args[1]
        self.assertEqual(set(generated_from.flatten().tolist()), {-1., -.5})
        with self.assertRaises(FileNotFoundError):
            protocol.select(self.args)
        with self.assertRaises(ValueError):
            protocol.freeze(self.args)
        # Discovery must not even open the held-out data files.
        original_load = torch.load

        def train_only_load(path, *args, **kwargs):
            if Path(path).name in ("cal.pt", "test.pt", "cal_population.pt", "test_population.pt"):
                raise AssertionError("discovery consumed holdout data")
            return original_load(path, *args, **kwargs)

        with patch.object(torch, "load", side_effect=train_only_load):
            protocol.discover(self.args)
            protocol.freeze(self.args)
        self.assertFalse((self.root/"cal_population.pt").exists())
        bank = self.root/"banks/first"
        manifest_hash = protocol.file_hash(bank/"manifest.json")
        # No final manifest exists yet: selection must not depend on it.
        protocol.select(self.args)
        self.assertEqual(protocol.file_hash(bank/"manifest.json"), manifest_hash)
        population = torch.load(self.root/"cal_population.pt", weights_only=True)
        self.assertEqual(set(population["source_ids"].tolist()), {4, 5})
        evaluations = pd.read_csv(bank/"calibration_attack_evaluation.csv")
        for _, rows in evaluations.groupby("attack_id"):
            self.assertEqual(set(rows["lambda"]), {0., 1.})
        selection_hash = protocol.file_hash(bank/"selection.json")
        pd.DataFrame({"path": [str(self.old/"models/current.pt")]}).to_csv(
            self.old/"future_final_manifest.csv", index=False)
        protocol.evaluate(self.args)
        self.assertEqual(protocol.file_hash(bank/"selection.json"), selection_hash)
        test = torch.load(self.root/"test_population.pt", weights_only=True)
        self.assertEqual(set(test["source_ids"].tolist()), {7, 8})
        self.assertTrue((bank/"final_results.csv").exists())
        # A new generation appends compatible attacks; the old frozen bank
        # and selection remain immutable while a new bank contains a superset.
        self.args.generation, self.args.bank_name = "second", "second"
        protocol.discover(self.args)
        self.args.reference_bank = "first"
        protocol.freeze(self.args)
        _, meta, _, _ = protocol.load_instance(self.args)
        _, old_bank = protocol.verify_bank(self.root, meta, "first")
        _, new_bank = protocol.verify_bank(self.root, meta, "second")
        self.assertLess({a["attack_id"] for a in old_bank["attacks"]},
                        {a["attack_id"] for a in new_bank["attacks"]})
        self.args.reference_bank = "first"
        protocol.audit(self.args)
        comparison = pd.read_csv(self.root/"banks/second/strengthening_vs_first.csv")
        self.assertTrue((comparison.risk_increase >= 0).all())
        self.assertEqual(protocol.file_hash(bank/"selection.json"), selection_hash)

    def test_freeze_rejects_test_derived_attack(self):
        _, meta, _, _ = protocol.load_instance(self.args)
        (self.root/"attacks").mkdir()
        torch.save({"instance_id": meta["instance_id"], "discovery_split": "test"},
                   self.root/"attacks/invalid.pt")
        with self.assertRaisesRegex(ValueError, "non-training"):
            protocol.freeze(self.args)

    def test_evaluation_requires_selection(self):
        with self.assertRaises(FileNotFoundError):
            protocol.evaluate(self.args)
        self.assertFalse((self.root/"test_population.pt").exists())


if __name__ == "__main__":
    unittest.main()
