"""Regression coverage for union selection and frozen collection integrity."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from lift_ce.counterfactuals.population_path import empirical_ot_path
from lift_ce.experiments import finite_waypoint_paths_v2 as experiment


def model():
    result = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Flatten(0))
    with torch.no_grad():
        result[0].weight.fill_(1.)
        result[0].bias.fill_(-.5)
    return result


class FinitePathTests(unittest.TestCase):
    def test_joint_minimum_includes_straight_and_endpoints(self):
        rows = [dict(k=k, **{'lambda': lam}, r_over_b_ref=r,
                     pooled_worst_hinge=risk, recourse_cost=cost)
                for r in (.5, 1.)
                for k, lam, risk, cost in [(0, 1., .1, 1.), (1, .5, .2, 0.), (2, 0., .15, .5)]]
        chosen = experiment.joint_choices(pd.DataFrame(rows))
        self.assertTrue((chosen.k == 0).all())
        self.assertTrue((chosen['lambda'] == 1.).all())
        rows.append(dict(k=2, **{'lambda': .25}, r_over_b_ref=1., pooled_worst_hinge=.0999999,
                         recourse_cost=100.))
        chosen = experiment.joint_choices(pd.DataFrame(rows))
        self.assertEqual(chosen[chosen.r_over_b_ref == 1.].iloc[0].k, 2)

    def test_selection_pools_radii_and_all_paths_without_final_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, original = root/'out', root/'original'
            out.mkdir(); original.mkdir()
            p = empirical_ot_path(torch.tensor([[-1.], [-.5]]), torch.tensor([[1.], [2.]]))
            population = {'path': p.state_dict(), 'source': p.base}
            torch.save(population, root/'cal_population.pt')
            (root/'cal_population.sha256').write_text(experiment.protocol.file_hash(root/'cal_population.pt'))
            torch.save({'waypoints': p.target+2.}, out/'path1.pt')
            experiment.protocol.write_json(out/'paths_manifest.json', {'reference_model_count': 1, 'paths': [
                {'k': 0, 'type': 'linear'}, {'k': 1, 'file': 'path1.pt',
                'sha256': experiment.protocol.file_hash(out/'path1.pt')}]})
            torch.save(model().state_dict(), out/'model.pt')
            records = [{'attack_id': f'a{i}', 'radius_ratio': radius, 'model_path': str(out/'model.pt'),
                        'model_sha256': experiment.protocol.file_hash(out/'model.pt')}
                       for i, radius in enumerate((.5, 1.))]
            experiment.protocol.write_json(out/'bank_manifest.json', {
                'paths_sha256': experiment.protocol.file_hash(out/'paths_manifest.json'), 'attacks': records})
            cal = root/'update_alignment_feasibility_v1/matched_models'
            cal.mkdir(parents=True)
            torch.save(model().state_dict(), cal/'model0.pt')
            pd.DataFrame({'path': ['unused']}).to_csv(original/'future_calibration_manifest.csv', index=False)
            meta = {'original_instance': str(original), 'lambdas': [0., .5, 1.], 'ratios': [.5, 1.]}
            context = (root, meta, {'x': p.target}, model(), root/'unused', out)
            with patch.object(experiment, 'context', return_value=context), patch.object(
                    experiment.protocol.legacy, 'load_model', side_effect=lambda *a: model()):
                experiment.select(SimpleNamespace(device='cpu'))
            selection = experiment.verified_selection(root, out)
            self.assertEqual(len(selection['selected']), 2)
            frame = pd.read_csv(out/'calibration_curves.csv')
            self.assertEqual(set(frame[frame.r_over_b_ref == 1.].bank_size), {2})
            self.assertEqual(set(frame[frame.r_over_b_ref == .5].bank_size), {1})
            cross = pd.read_csv(out/'calibration_attack_cross_evaluation.csv')
            self.assertEqual(len(cross), 12)
            self.assertTrue((cross.groupby(['k', 'attack_id']).size() == 3).all())
            candidates = pd.read_csv(out/'joint_candidates.csv')
            for _, row in candidates.iterrows():
                self.assertEqual(row.pooled_worst_hinge,
                                 frame[frame.r_over_b_ref == row.r_over_b_ref].pooled_worst_hinge.min())
            # Frozen content mutations must not be silently accepted.
            torch.save({'waypoints': p.target}, out/'path1.pt')
            with self.assertRaisesRegex(ValueError, 'frozen path changed'):
                experiment.load_paths(out, population)


if __name__ == '__main__':
    unittest.main()
