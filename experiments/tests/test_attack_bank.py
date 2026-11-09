"""Regression coverage for deployment-level attack-bank ranking (CPU only)."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from lift_ce.experiments import heloc_diabetes_official_lift_v1 as driver
from lift_ce.experiments.heloc_conditional_adversary_strengthened_v1 import (
    differentiable_adamw_update,
    radial_full_evaluation,
)


class ProbabilityModel(torch.nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = torch.tensor(probabilities)

    def forward(self, q):
        return torch.logit(self.probabilities.expand(len(q)))


class PathModel(torch.nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start, self.end = start, end

    def forward(self, q):
        return torch.logit(self.start + (self.end-self.start)*q[:, 0])


def attack(source_lambda, model, delta=0.0):
    return (source_lambda, 0, {
        "delta": torch.full((2, 1), delta), "source_full_metrics": {},
    }, model)


class AttackBankTests(unittest.TestCase):
    def test_maximum_is_after_population_mean(self):
        bank = [attack(0., ProbabilityModel([.1, .9])),
                attack(1., ProbabilityModel([.9, .1]))]
        rows, worst = driver.evaluate_attack_bank(bank, torch.zeros(2, 1), 1.)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(worst["true_hinge"], .2, places=6)
        # Per-CF maximum would incorrectly give .4.

    def test_cross_lambda_attack_can_remove_artificial_endpoint_minimum(self):
        bank = [attack(0., PathModel(.3, .05)), attack(1., PathModel(.4, .49))]
        q0, q1 = torch.zeros(2, 1), torch.ones(2, 1)
        own0 = driver.evaluate_attack_bank(bank[:1], q0, 1.)[1]["true_hinge"]
        own1 = driver.evaluate_attack_bank(bank[1:], q1, 1.)[1]["true_hinge"]
        self.assertLess(own1, own0)
        pooled0 = driver.evaluate_attack_bank(bank, q0, 1.)[1]["true_hinge"]
        pooled1 = driver.evaluate_attack_bank(bank, q1, 1.)[1]["true_hinge"]
        self.assertGreater(pooled1, pooled0)

    def test_stronger_bank_is_monotone_and_endpoint_can_still_win(self):
        old = [attack(0., PathModel(.2, .45))]
        stronger = old + [attack(1., PathModel(.1, .4))]
        risks = []
        for lam in (0., .5, 1.):
            q = torch.full((2, 1), lam)
            old_risk = driver.evaluate_attack_bank(old, q, 1.)[1]["true_hinge"]
            new_risk = driver.evaluate_attack_bank(stronger, q, 1.)[1]["true_hinge"]
            self.assertGreaterEqual(new_risk, old_risk)
            risks.append(new_risk)
        self.assertEqual(int(np.argmin(risks)), 2)

    def test_invalid_banks_fail_instead_of_silently_ranking(self):
        q = torch.zeros(2, 1)
        for bank in ([], [attack(0., ProbabilityModel([.2]), delta=2.)],
                     [attack(0., ProbabilityModel([float("nan")]))]):
            with self.assertRaises(ValueError):
                driver.evaluate_attack_bank(bank, q, 1.)

    def test_zero_adam_moments_have_finite_outer_gradients(self):
        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(),
                                    torch.nn.Linear(3, 1), torch.nn.Flatten(0))
        with torch.no_grad():
            model[0].weight[0].zero_()
            model[0].bias[0] = -10.  # Dead unit guarantees zero second moments.
        x, y = torch.randn(4, 2), torch.tensor([0., 1., 0., 1.])
        delta = torch.zeros_like(x, requires_grad=True)
        pred = differentiable_adamw_update(model, x, y, delta, x, steps=3)
        grad = torch.autograd.grad(torch.nn.functional.softplus(.5-pred).mean(), delta)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.)

    def test_radial_search_retains_every_discovered_candidate(self):
        retained = []
        with patch("lift_ce.experiments.heloc_conditional_adversary_strengthened_v1.full_metrics",
                   return_value=(ProbabilityModel([.2]), {"true_hinge": .3, "invalidity": 1.})):
            _, radial = radial_full_evaluation(None, torch.zeros(2, 1), torch.zeros(2),
                torch.zeros(2, 1), torch.full((2, 1), .05), 1., 0, "cpu",
                candidate_sink=lambda row, delta: retained.append(delta.clone()))
        self.assertEqual(len(retained), len(radial))
        self.assertEqual(len(retained), 5)

    def test_aggregate_unions_legacy_and_new_radial_attacks(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            sd = Path(tmp)
            x = torch.zeros(2, 1)
            d = {"train_x": x, "train_y": torch.zeros(2), "feature_names": ["x"]}
            common = {"source": x, "b_ref": 1.}
            original = {"native_failure": torch.zeros(2, dtype=torch.bool),
                        "runtime_seconds": torch.zeros(2)}
            path = {"0.00": x, "1.00": torch.ones_like(x)}
            seeds = []

            def update(current, train_x, train_y, seed, device):
                seeds.append(seed)
                return ProbabilityModel([.4-float(train_x.mean())*.1])

            for lam in (0., 1.):
                ad = sd/"attacks/lift/r1"/f"lambda{lam:.2f}"
                ad.mkdir(parents=True)
                saved = {"delta": x, "search_seed": int(lam), "source_lambda": lam,
                         "radius_ratio": 1., "source_full_metrics": {}}
                torch.save(saved, ad/"restart0.pt")
                if lam == 0.:
                    torch.save({**saved, "attack_version": driver.ATTACK_VERSION,
                        "instance_id": "fixed", "full_update_seed": 123,
                        "candidates": [{"delta": x+.25, "source_full_metrics": {}},
                                       {"delta": x+.5, "source_full_metrics": {}}]},
                        ad/f"restart0_{driver.ATTACK_VERSION}.pt")
            for split in ("calibration", "final"):
                pd.DataFrame({"path": ["mock-model"]}).to_csv(sd/f"future_{split}_manifest.csv", index=False)
            replacements = {
                "RATIOS": (1.,), "LAMBDAS": (0., 1.), "RESTARTS": 1,
                "load_common": lambda *a: (sd, d, common, ProbabilityModel([.6])),
                "path_artifact": lambda *a: (x, x+1, path, original),
                "attack_instance_id": lambda *a: "fixed", "fine_tune": update,
                "load_model": lambda *a: ProbabilityModel([.6]),
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(driver, name, value))
            driver.aggregate(SimpleNamespace(output=sd, dataset="heloc", seed=0,
                                            data_home=tmp, method="lift", device="cpu"))
            rows = pd.read_csv(sd/"lift_conditional_attack_evaluation.csv")
            curves = pd.read_csv(sd/"lift_conditional_robustness_cost_curves.csv")
            self.assertEqual(len(rows), 8)  # Four attacks times two lambdas.
            self.assertEqual(sorted(seeds), [123, 123, 900000, 900001])
            self.assertTrue((curves.bank_size == 4).all())
            self.assertTrue((curves.bank_strengthening_risk_increase > 0).all())
            self.assertTrue((sd/"attacks/lift/r1/lambda0.00/restart0.pt").exists())


if __name__ == "__main__":
    unittest.main()
