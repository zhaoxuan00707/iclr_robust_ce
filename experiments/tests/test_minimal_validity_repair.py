import unittest
import numpy as np
import torch
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.minimal_validity_repair import minimal_restore,valid_intervals,restore_with_budget

class MinimalRepairTests(unittest.TestCase):
    def monotone_model(self):
        model=TabularMLP(1,[2])
        with torch.no_grad():
            model.features[0].weight.copy_(torch.tensor([[1.],[-1.]]));model.features[0].bias.zero_()
            model.classifier.weight.copy_(torch.tensor([[1.,-1.]]));model.classifier.bias.zero_()
        return model

    def test_minimum_crossing_and_valid_points_unchanged(self):
        model=self.monotone_model();raw=torch.tensor([[-1.],[.2]])
        repaired,rows=minimal_restore(model,raw,torch.ones_like(raw))
        self.assertAlmostEqual(rows[0]['alpha'],.5,places=6)
        self.assertEqual(rows[1]['alpha'],0.)
        torch.testing.assert_close(repaired[1],raw[1],atol=0,rtol=0)
        self.assertTrue((torch.sigmoid(model(repaired))>=.5).all())

    def test_invalid_anchor_is_explicit_failure(self):
        with self.assertRaisesRegex(ValueError,'invalid corresponding anchors'):
            minimal_restore(self.monotone_model(),torch.tensor([[-1.]]),torch.tensor([[-.5]]))

    def test_nonmonotone_segment_uses_first_interval(self):
        # -0.2 + x - 3 relu(x-.4) + 4 relu(x-.7): first valid at .2,
        # then invalid, then valid again. Whole-segment bisection can miss .2.
        model=TabularMLP(1,[3])
        with torch.no_grad():
            model.features[0].weight.fill_(1.)
            model.features[0].bias.copy_(torch.tensor([0.,-.4,-.7]))
            model.classifier.weight.copy_(torch.tensor([[1.,-3.,4.]]));model.classifier.bias.fill_(-.2)
        intervals=valid_intervals(model,np.array([0.]),np.array([1.]))
        self.assertEqual(len(intervals),2)
        repaired,rows=minimal_restore(model,torch.tensor([[0.]]),torch.tensor([[1.]]))
        self.assertAlmostEqual(rows[0]['alpha'],.2,places=6)

    def test_over_budget_repair_is_rejected_without_moving_valid_points(self):
        model=TabularMLP(2,[3])
        with torch.no_grad():
            model.features[0].weight.copy_(torch.tensor([[0.,-1.],[0.,1.],[1.,0.]]))
            model.features[0].bias.copy_(torch.tensor([.75,-1.25,0.]))
            model.classifier.weight.fill_(1.);model.classifier.bias.fill_(-.2)
        source=torch.tensor([[0.,0.]]);raw=torch.tensor([[1.,0.],[0.,1.]])
        target=torch.tensor([[2.,0.],[0.,2.]])
        result,rows,info=restore_with_budget(model,source,raw,target,2.,2.)
        self.assertIsNone(result)
        self.assertTrue(info['fallback_used'])
        self.assertLess(info['initial_minimal_repair_geometry']['slack'],0)
        self.assertEqual(rows[0]['alpha'],0.)
        self.assertEqual(info['status'],'infeasible_under_declared_joint_repair_search')

    def test_raw_optimizer_can_retain_current_invalid_points(self):
        from lift_ce.counterfactuals.minimal_validity_repair import solve_raw
        from lift_ce.counterfactuals.robust_waypoints import ModelBank
        import copy
        current=self.monotone_model();adverse=copy.deepcopy(current)
        with torch.no_grad():adverse.classifier.weight.mul_(-1)
        bank=ModelBank([adverse],'cpu')
        settings={'steps':0,'learning_rate':.03,'rho_initial':1.,'beta_h':50.,'beta_m':100.,
                  'feasibility_relative_tolerance':1e-6,'restoration_steps':2,'dual_interval':5,'stall_interval':20}
        retained,_=solve_raw(bank,torch.tensor([[0.]]),torch.tensor([[1.]]),torch.tensor([[-.5]]),
                             1.,3.,torch.tensor([-2.]),torch.tensor([2.]),settings)
        chosen=min(retained,key=lambda r:r['true_bank_hinge'])
        self.assertLess(float(current(chosen['points'])[0]),0.)
        self.assertGreaterEqual(chosen['slack'],-1e-6)

    def test_budget_rechecked_and_valid_mass_preserved(self):
        model=self.monotone_model();source=torch.tensor([[.1]]);raw=torch.tensor([[-.1]])
        repaired,rows,info=restore_with_budget(model,source,raw,torch.tensor([[1.]]),.9,1.1)
        self.assertIsNotNone(repaired)
        self.assertGreaterEqual(info['initial_minimal_repair_geometry']['slack'],-1e-6)
        self.assertEqual(len(repaired),1)

if __name__=='__main__':unittest.main()
