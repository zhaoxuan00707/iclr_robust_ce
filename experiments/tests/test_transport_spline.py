import unittest
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog
from lift_ce.counterfactuals.population_path import empirical_ot_path, WaypointPopulationPath
from lift_ce.counterfactuals.transport_spline import TransportSplinePath, bending_basis, bend_diagnostics


class SplineTests(unittest.TestCase):
    def test_straight_geodesic_identity_with_independent_weighted_ot(self):
        p = empirical_ot_path(torch.tensor([[0.], [1.]]), torch.tensor([[2.], [3.], [4.]]))
        a, b = p.at(.2).double(), p.at(.7).double()
        n = len(a)
        constraints = np.zeros((2*n, n*n))
        for i in range(n):
            constraints[i, i*n:(i+1)*n] = 1
            constraints[n+i, i::n] = 1
        result = linprog(torch.cdist(a,b).square().numpy().ravel(),
                         A_eq=constraints, b_eq=np.tile(p.weights.numpy(),2),
                         bounds=(0,None), method='highs')
        self.assertTrue(result.success)
        distance = float(((p.at(1)-p.at(0)).double().square().sum(1)*p.weights.double()).sum().sqrt())
        self.assertAlmostEqual(np.sqrt(result.fun), distance*.5, places=6)

    def test_natural_variational_solution_matches_independent_scipy(self):
        t = np.linspace(0, 1, 101)
        knots = np.array([[0.,2.],[3.,-1.],[1.,4.]])
        curve = CubicSpline([0,.5,1],knots,bc_type='natural')
        actual = (1-t[:,None])*knots[0]+t[:,None]*knots[2]+bending_basis(torch.tensor(t)).numpy()[:,None]*(knots[1]-(knots[0]+knots[2])/2)
        np.testing.assert_allclose(actual,curve(t),atol=1e-12)
        np.testing.assert_allclose(curve([0,1],2),0,atol=1e-12)
        np.testing.assert_allclose(curve(.5-1e-9,1),curve(.5+1e-9,1),atol=1e-6)

    def test_split_pairing_knots_and_bounds(self):
        p = empirical_ot_path(torch.tensor([[0.],[1.]]),torch.tensor([[2.],[3.],[4.]]))
        w = p.target+1
        curve = TransportSplinePath(p,w)
        wp = WaypointPopulationPath(p,w)
        for t in [0,.5,1]:torch.testing.assert_close(curve.at(t),wp.at(t))
        grid = [i/20 for i in range(21)]
        diag = bend_diagnostics(p,w,grid)
        self.assertLessEqual(diag['L_path_emp'],diag['L_path_bound'])
        for index, mass in [(p.source_index,1/2),(p.target_index,1/3)]:
            torch.testing.assert_close(torch.bincount(index,weights=curve.weights),torch.full((int(index.max())+1,),mass))

    def test_midpoint_reduces_to_straight_and_invalid_time_rejected(self):
        p = empirical_ot_path(torch.tensor([[0.]]),torch.tensor([[2.]]))
        curve = TransportSplinePath(p,torch.tensor([[1.]]))
        for t in [0,.1,.5,.8,1]:torch.testing.assert_close(curve.at(t),p.at(t))
        for t in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):curve.at(t)

if __name__ == '__main__':unittest.main()
