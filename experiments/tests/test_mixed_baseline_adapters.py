import unittest
import torch
from lift_ce.baselines.mixed_tabular_adapters import categorical_projector, add_categorical_constraints
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain

class MixedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.train=torch.tensor([[0.,-1.,2.],[1.,2.,-1.]],dtype=torch.float64)
        self.groups={'category':[1,2]}
        self.domain=MixedDomain(self.train,self.groups)
    def test_relaxed_projection(self):
        x=torch.tensor([7.,4.,3.],dtype=torch.float64)
        y=categorical_projector(self.train,self.groups)(x)
        p=(y[1:]+1)/3
        self.assertAlmostEqual(float(y[0]),7.)
        self.assertAlmostEqual(float(p.sum()),1.,places=10)
        self.assertTrue((p>=0).all() and (p<=1).all())
    def test_hard_projection(self):
        x=torch.tensor([7.,4.,3.],dtype=torch.float64)
        y=categorical_projector(self.train,self.groups,True)(x)
        self.assertTrue(torch.equal(y,torch.tensor([7.,2.,-1.],dtype=torch.float64)))
    def test_continuous_identity(self):
        x=torch.tensor([7.,4.,3.],dtype=torch.float64)
        self.assertTrue(torch.equal(categorical_projector(self.train,{})(x),x))
    def test_solver_affine_domain(self):
        import gurobipy as gp
        for hard in [False,True]:
            model=gp.Model();model.Params.OutputFlag=0
            inputs={f'v_0_{i}':model.addVar(lb=-10,ub=10,name=f'v_0_{i}') for i in range(3)}
            add_categorical_constraints(model,inputs,self.domain,hard)
            model.setObjective((inputs['v_0_1']-.5)**2+(inputs['v_0_2']-.5)**2)
            model.optimize();self.assertEqual(model.Status,gp.GRB.OPTIMAL)
            p=torch.tensor([(inputs[f'v_0_{j}'].X+1)/3 for j in [1,2]])
            self.assertAlmostEqual(float(p.sum()),1.,places=6)
            if hard:self.assertTrue(all(float(v) in [0.,1.] for v in p))
            else:self.assertTrue(torch.allclose(p,torch.tensor([.5,.5])))
            model.dispose()
if __name__=='__main__':unittest.main()
