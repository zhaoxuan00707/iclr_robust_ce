import unittest,torch
from lift_ce.counterfactuals.adjoint_factor import factor_adjoint
class TestFactor(unittest.TestCase):
 def test_population_norm_and_cancellation(self):
  torch.manual_seed(4);a=torch.randn(12,37,dtype=torch.double);a[1]=-a[0];f=factor_adjoint(a)
  for ids in [[0,1],[3,4,7],[2,2,8,10]]:
   self.assertAlmostEqual(float(a[ids].sum(0).norm()),float(f[ids].sum(0).norm()),places=6)
if __name__=='__main__':unittest.main()
