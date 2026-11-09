import unittest
import torch
from lift_ce.counterfactuals.direct_realization import direct_realization

class TestDirectRealization(unittest.TestCase):
    def test_global_assignment_does_not_duplicate_nearest_neighbor(self):
        z=torch.tensor([[0.],[.1]]);q=torch.tensor([[0.],[10.]])
        r=direct_realization(z,q,True)
        self.assertTrue(r['one_per_factual'])
        self.assertEqual(sorted(r['points'].flatten().tolist()),[0.,10.])
        self.assertAlmostEqual(r['plan']['distance']**2,9.9**2/2,places=5)
    def test_unequal_counts_preserve_both_marginals_not_barycenters(self):
        z=torch.tensor([[0.],[3.]]);q=torch.tensor([[0.],[1.],[3.]])
        r=direct_realization(z,q)
        self.assertFalse(r['one_per_factual'])
        self.assertEqual(len(r['points']),4)
        self.assertTrue(all(float(v) in [0.,1.,3.] for v in r['points']))
        torch.testing.assert_close(q[r['control_index']],r['points'],rtol=0,atol=0)
    def test_refuse_false_deterministic_claim(self):
        with self.assertRaisesRegex(ValueError,'equal support counts'):
            direct_realization(torch.zeros(2,1),torch.zeros(3,1),True)
if __name__=='__main__':unittest.main()
