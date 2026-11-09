import unittest
import numpy as np
from gurobipy import Model,GRB
from lift_ce.baselines.proplace_license_fallback import solve_linear_highs

class SolverTests(unittest.TestCase):
 def test_same_mixed_integer_optimum(self):
  with Model() as m:
   m.Params.OutputFlag=0;x=m.addVar(lb=-2,ub=3);b=m.addVar(vtype=GRB.BINARY)
   m.addConstr(x>=1-3*b);m.addConstr(x<=2-b);m.setObjective(x+.2*b);m.optimize()
   z,info=solve_linear_highs(m)
   self.assertAlmostEqual(info['objective'],m.ObjVal,places=6)
   np.testing.assert_allclose(z,[v.X for v in m.getVars()],atol=1e-6)
 def test_over_license_constraint_count(self):
  with Model() as m:
   m.Params.OutputFlag=0;x=m.addVar(lb=-2,ub=3)
   for _ in range(2100):m.addConstr(x>=1)
   m.setObjective(x);z,info=solve_linear_highs(m)
   self.assertAlmostEqual(z[0],1.);self.assertEqual(info['constraints'],2100)
if __name__=='__main__':unittest.main()

class TightMasterTests(unittest.TestCase):
 def test_matches_original_master(self):
  import torch,pandas as pd
  from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
  from robustx.lib.tasks.ClassificationTask import ClassificationTask
  from lift_ce.experiments.run_official_proplace import FrozenFrameDataset
  from lift_ce.baselines.tabular_replication_adapters import corrected_proplace
  for seed in range(4):
   net=SimpleNNModel(2,[3,2],1,seed=seed)
   with torch.no_grad():
    for layer in net.model:
     if isinstance(layer,torch.nn.Linear):layer.bias.fill_(.3)
    [layer for layer in net.model if isinstance(layer,torch.nn.Linear)][-1].bias.fill_(1.)
   data=FrozenFrameDataset(seed=seed);data.data=pd.DataFrame({'a':[-2.,0.,2.],'b':[1.,0.,2.],'target':[0,1,1]})
   task=ClassificationTask(net,data);train=torch.tensor(data.X.to_numpy(),dtype=torch.float32);candidates=train.numpy()
   original=corrected_proplace(task,train,{})
   fixed=corrected_proplace(task,train,{},license_fallback=True)
   original.neg_value=fixed.neg_value=0
   a=original._master_prob([-1.,-1.],candidates);b=fixed._master_prob([-1.,-1.],candidates)
   self.assertFalse(a.empty);self.assertFalse(b.empty)
   self.assertAlmostEqual(np.abs(a.to_numpy()+1).sum(),np.abs(b.to_numpy()+1).sum(),places=4)
