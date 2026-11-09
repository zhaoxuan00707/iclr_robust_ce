import unittest
import torch
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain,mixed_surrogate
from lift_ce.counterfactuals.update_aware import surrogate
from lift_ce.models.tabular import TabularMLP

class MixedTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(0)
  self.x=torch.tensor([[.2,-1.,2.],[-.7,2.,-1.],[1.,-1.,2.],[-1.,2.,-1.]],dtype=torch.float64)
  self.d=MixedDomain(self.x,{'category':[1,2]})
 def test_simplex_projection_and_vertices(self):
  z=self.d.project(self.x+torch.randn_like(self.x)*3)
  p=(z[:,1:]-self.d.low[1:])/self.d.scale[1:]
  self.assertLess(float((p.sum(1)-1).abs().max()),1e-10)
  self.assertGreaterEqual(float(p.min()),-1e-12)
  self.assertTrue(torch.allclose(self.d.project(self.x),self.x,atol=1e-10))
  hard=self.d.project(z,True)
  self.assertTrue(all(any(torch.equal(a,b) for b in self.x[:,1:]) for a in hard[:,1:]))
 def test_full_update_gradient(self):
  model=TabularMLP(3,[4]).double();y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
  q=self.x[:2].clone().requires_grad_(True)
  def f(q):return mixed_surrogate(model,self.x,y,q,.3,self.d,steps=3)[0]
  grad=torch.autograd.grad(f(q),q)[0];v=torch.randn_like(q);v/=v.norm();eps=1e-5
  fd=(f(q+eps*v)-f(q-eps*v))/(2*eps)
  self.assertLess(float((fd-(grad*v).sum()).abs()),1e-7)
 def test_continuous_matches_original(self):
  model=TabularMLP(3,[4]).double();y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64);q=self.x[:2].clone().requires_grad_(True)
  a=mixed_surrogate(model,self.x,y,q,.2,MixedDomain(self.x,{}),steps=3)
  b=surrogate(model,self.x,y,q,.2,steps=3,hinge_temperature=.05)
  for v,w in zip(a,b):self.assertTrue(torch.allclose(v,w,atol=1e-12))
if __name__=='__main__':unittest.main()
