import unittest
import numpy as np
import torch
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.local_graph_repair import minimal_component_bridges, relu_segment_minimum


class LocalGraphRepairTests(unittest.TestCase):
    def test_only_one_accepted_bridge_is_added(self):
        nodes=np.array([[0.],[1.],[2.]])
        edges=np.array([[1,2]])
        original=edges.copy()
        added,attempts,audit=minimal_component_bridges(nodes,edges,[0],[1,2],
            lambda u,v: {'allowed':True} if v==2 else None)
        np.testing.assert_array_equal(added,[[0,2]])
        np.testing.assert_array_equal(edges,original)
        self.assertTrue(audit['compatible'])
        self.assertEqual(len(attempts),2)

    def test_no_forced_connection_when_edges_forbidden(self):
        with self.assertRaisesRegex(ValueError,'No admissible'):
            minimal_component_bridges(np.array([[0.],[1.],[2.]]),np.array([[1,2]]),[0],[1,2],lambda *args:None)

    def test_piecewise_check_detects_invalid_interior(self):
        model=TabularMLP(1,[2])
        with torch.no_grad():
            model.features[0].weight.copy_(torch.tensor([[1.],[-1.]]))
            model.features[0].bias.zero_()
            model.classifier.weight.fill_(1.)
            model.classifier.bias.fill_(-.25)
        self.assertTrue((model(torch.tensor([[-1.],[1.]]))>0).all())
        check=relu_segment_minimum(model,np.array([-1.]),np.array([1.]))
        self.assertAlmostEqual(check['minimum_logit'],-.25)
        self.assertEqual(check['affine_pieces'],2)

if __name__=='__main__':unittest.main()
