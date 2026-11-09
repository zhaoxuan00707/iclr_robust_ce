import unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra
from lift_ce.counterfactuals.favorable_region import graph_ot, bank_risk, solve_nodes, intrinsic_path
from lift_ce.counterfactuals.risk_geometry import sparse_graph


class FavorableRegionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
        self.graph = sparse_graph(4, np.array([[0, 1], [1, 2], [2, 3]]), np.ones(3))
        self.dist, self.pred = dijkstra(self.graph, return_predecessors=True)

    def test_population_max_not_pointwise_max(self):
        risk, _ = bank_risk(np.array([[.4, 0.], [0., .4]]), [0, 1])
        self.assertAlmostEqual(risk, .2)

    def test_disconnected_mass_is_not_given_finite_cost(self):
        d = np.array([[0., np.inf], [np.inf, 0.]])
        with self.assertRaisesRegex(ValueError, 'incompatible component masses'):
            graph_ot(d, [0, 0, 1], [0, 1])
        self.assertEqual(graph_ot(d, [0, 1], [0, 1]).distance, 0.)

    def test_duplicates_preserve_target_masses(self):
        p = graph_ot(self.dist, [0], [2, 3, 3])
        np.testing.assert_allclose(np.bincount(p.target_index, weights=p.weights), 1/3)
        self.assertAlmostEqual(p.distance, np.sqrt(22/3))

    def test_route_uses_graph_not_euclidean_shortcut(self):
        path = intrinsic_path(self.nodes, self.dist, self.pred, [0], [2], [3], [0., .5, 2/3, 1.])
        np.testing.assert_allclose(path['q'][:, 0], [[0, 0], [.5, 1], [1, 1], [1, 0]])
        self.assertAlmostEqual(path['lambda_Q'], 2/3)
        self.assertAlmostEqual(path['total_length'], 3)

    def test_feasible_search_and_nested_incumbent(self):
        losses = np.array([[.3, .1, 0., .4]])
        first = solve_nodes(self.dist, self.graph, losses, [0], [3], 3., steps=8)
        second = solve_nodes(self.dist, self.graph, losses, [0], [3], 3.6,
                             incumbent=first['candidate']['node_ids'], steps=8)
        self.assertLessEqual(first['candidate']['total_length'], 3+1e-6)
        self.assertLessEqual(second['candidate']['true_bank_hinge'], first['candidate']['true_bank_hinge'])
        self.assertEqual(first['candidate']['true_bank_hinge'], 0.)
        self.assertAlmostEqual(first['candidate']['total_length'], 3.)

    def test_budget_below_anchor_distance_is_infeasible(self):
        result = solve_nodes(self.dist, self.graph, np.zeros((1, 4)), [0], [3], 2.)
        self.assertIsNone(result['candidate'])

if __name__ == '__main__':
    unittest.main()
