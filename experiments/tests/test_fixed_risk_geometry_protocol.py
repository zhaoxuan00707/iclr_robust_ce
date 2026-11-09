import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from lift_ce.experiments import fixed_risk_geometry_v1 as e


class FixedGeometryProtocolTests(unittest.TestCase):
    def test_exact_hinge_selection_ignores_cost_and_near_optimality(self):
        rows = pd.DataFrame([{'method': 'C', 'r_over_b_ref': 1., 'lambda': .0, 'bank_hinge': .10000001, 'recourse_cost': 0.},
                             {'method': 'C', 'r_over_b_ref': 1., 'lambda': 1., 'bank_hinge': .1, 'recourse_cost': 100.}])
        self.assertEqual(e.exact_lambda_choices(rows)[0]['lambda'], 1.)
        rows.loc[0, 'bank_hinge'] = .1
        chosen = e.exact_lambda_choices(rows)[0]
        self.assertEqual(chosen['lambda'], 0.)
        self.assertEqual(chosen['exact_ties'], [0., 1.])

    def test_feasibility_provenance_and_perturbation_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            meta, train = {'full_update_seed': 7, 'b_ref': 1.}, {'x': torch.zeros(2, 1)}
            obj = {'candidates': [torch.full((2, 1), .25)], 'discovery_split': 'train', 'full_update_seed': 7}
            torch.save(obj, out/'attack.pt'); torch.save({}, out/'model.pt')
            row = {'split': 'train', 'full_update_seed': 7, 'nominal_radius': 1., 'delta_file': str(out/'attack.pt'),
                   'delta_index': 0, 'model_path': str(out/'model.pt'), 'model_sha256': e.p.file_hash(out/'model.pt')}
            result = e.verified_records([row], meta, train)[0]
            self.assertEqual(result['measured_budget'], .25)
            with self.assertRaisesRegex(ValueError, 'non-training'):
                e.verified_records([{**row, 'split': 'test'}], meta, train)
            obj['candidates'] = [torch.ones(2, 1)*2]
            torch.save(obj, out/'attack.pt')
            with self.assertRaisesRegex(ValueError, 'perturbation changed'):
                e.verified_records([result], meta, train)
            with self.assertRaisesRegex(ValueError, 'infeasible'):
                e.verified_records([row], meta, train)

    def test_fresh_discoveries_cannot_redefine_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            names = {'geometry_bank.json': 'geometry_bank_sha256', 'graph.pt': 'graph_sha256',
                     'paths_train.pt': 'train_paths_sha256', 'resolved_config.json': 'settings_sha256'}
            for name in names:
                (out/name).write_text('frozen')
            e.p.write_json(out/'frozen_geometry.json', {key: e.p.file_hash(out/name) for name, key in names.items()})
            e.verify_geometry(out)
            (out/'graph.pt').write_text('new attack tries to change graph')
            with self.assertRaisesRegex(ValueError, 'frozen geometry changed'):
                e.verify_geometry(out)

    def test_full_bank_weight_bounds_and_dominance_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            e.p.write_json(out/'geometry_bank.json', {'candidates': [
                {'model_path': 'first', 'candidate_id': 'dominant'}, {'model_path': 'second', 'candidate_id': 'other'}]})
            class Predictor(torch.nn.Module):
                def __init__(self, bias):
                    super().__init__(); self.bias = torch.nn.Parameter(torch.tensor(float(bias)))
                def forward(self, x):
                    return x[:, 0]+self.bias
            with patch.object(e.legacy, 'load_model', side_effect=[Predictor(-2.), Predictor(2.)]):
                w, winner, stats = e.score_geometry(out, {'x': torch.zeros(2, 1)}, np.array([[-1.], [0.], [1.]]), 'test', 'cpu')
            self.assertTrue(((w >= 1) & (w <= 2)).all())
            self.assertTrue((winner == 0).all())
            self.assertEqual(stats['dominant_model_fraction'], 1.)
            self.assertEqual(stats['models'], 2)
            cache = np.load(out/'scores/test.npy')
            np.testing.assert_allclose(w, 2-cache.min(1), rtol=1e-7)

    def test_enlarging_bank_maximum_cannot_reduce_risk(self):
        old = np.array([[.2, .4, .1], [.3, .1, .2]])
        additional = np.array([[.1, .5, .3]])
        self.assertTrue((np.r_[old, additional].max(0) >= old.max(0)).all())
        self.assertNotEqual(float(np.r_[old, additional].max(0).mean()),
                            float(np.r_[old, additional].mean(1).max()))


if __name__ == '__main__':
    unittest.main()
