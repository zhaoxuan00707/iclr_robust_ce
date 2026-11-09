import unittest

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

from lift_ce.counterfactuals import risk_geometry as g


class RiskGeometryTests(unittest.TestCase):
    def setUp(self):
        self.nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        self.edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        self.ones = np.ones((4, 9))

    def test_unit_constant_and_bounded_metric(self):
        unit = g.edge_lengths(self.nodes, self.edges, self.ones)
        np.testing.assert_allclose(unit, 1.)
        graph = g.sparse_graph(4, self.edges, unit)
        self.assertEqual((graph-graph.T).nnz, 0)
        self.assertTrue((graph.data > 0).all())
        d0 = dijkstra(graph, directed=False)
        constant = g.sparse_graph(4, self.edges, g.edge_lengths(self.nodes, self.edges, self.ones*1.7))
        np.testing.assert_allclose(dijkstra(constant, directed=False), d0*1.7)
        bounded = self.ones*np.array([1., 1.2, 1.8, 2.])[:, None]
        weighted = g.sparse_graph(4, self.edges, g.edge_lengths(self.nodes, self.edges, bounded))
        dw = dijkstra(weighted, directed=False)
        self.assertTrue((dw >= d0-1e-10).all())
        self.assertTrue((dw <= 2*d0+1e-10).all())

    def test_quadrature_arclength_not_euclidean_fraction(self):
        points = np.array([[0.], [2.]])
        position = g.traverse_polyline(points, np.array([1., 2.]), .5)
        np.testing.assert_allclose(position, [2*(np.sqrt(2.5)-1)])
        self.assertLess(float(position[0]), 1.2)
        np.testing.assert_array_equal(g.traverse_polyline(points, [1., 2.], 0.), points[0])
        np.testing.assert_array_equal(g.traverse_polyline(points, [1., 2.], 1.), points[-1])

    def test_balanced_split_endpoints_fixed_routes_and_speed(self):
        nodes = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        base = torch.tensor(nodes[[0, 1]], dtype=torch.float32)
        target = torch.tensor(nodes[[1, 2, 3]], dtype=torch.float32)
        path, audit, _ = g.route_population(nodes, edges, np.ones((3, 9))*1.5,
                                           np.array([0, 1]), np.array([1, 2, 3]), base, target)
        np.testing.assert_allclose(np.bincount(path.source_index, weights=path.weights), [.5, .5])
        np.testing.assert_allclose(np.bincount(path.target_index, weights=path.weights), [1/3]*3)
        torch.testing.assert_close(path.at(0), base[path.source_index])
        torch.testing.assert_close(path.at(1), target[path.target_index])
        weights = path.weights.clone()
        for lam in np.linspace(0, 1, 21):
            path.at(float(lam))
        torch.testing.assert_close(path.weights, weights)
        displacement = torch.sqrt((path.weights*(path.at(.8)-path.at(.2)).double().square().sum(1)).sum())
        self.assertLessEqual(float(displacement), .6*path.speed_bound+1e-7)
        self.assertLess(audit['marginal_residual'], 1e-8)

    def test_disconnected_mass_is_not_replaced_with_penalty(self):
        edges = np.array([[0, 1], [2, 3]])
        graph = g.sparse_graph(4, edges, np.ones(2))
        audit = g.component_mass_audit(graph, [0], [1, 3])
        self.assertFalse(audit['compatible'])
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            g.balanced_plan(np.array([[1., np.inf]]))

    def test_weighting_can_change_route_without_extra_anchor(self):
        weight = self.ones.copy()
        weight[[0, 2]] = 2.
        graph = g.sparse_graph(4, self.edges, g.edge_lengths(self.nodes, self.edges, weight))
        _, pred = g.distances(graph, [0], [3])
        self.assertEqual(g.reconstruct_route(pred[0], 0, 3).tolist(), [0, 2, 3])

    def test_endpoint_checks_do_not_replace_edge_interior_checks(self):
        samples = g.edge_samples(np.array([[-1.], [1.]]), np.array([[0, 1]]), 9)
        allowed = np.abs(samples[..., 0]) >= .5
        self.assertTrue(allowed[0, 0] and allowed[0, -1])
        self.assertFalse(allowed.all())


if __name__ == '__main__':
    unittest.main()
