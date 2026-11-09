import unittest,math
import torch
from torch import nn
from torch.func import functional_call
from lift_ce.counterfactuals.update_aware import surrogate,unroll,differentiable_w2
class SmoothClassifier(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(2,3),nn.Tanh(),nn.Linear(3,1));self.double()
  with torch.no_grad():self.net[-1].bias.fill_(-1)
 def forward(self,x):return self.net(x).squeeze(-1)
class TestUpdateAware(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(4);self.m=SmoothClassifier();self.x=torch.randn(5,2,dtype=torch.double);self.y=torch.tensor([0.,1.,0.,1.,1.],dtype=torch.double);self.q=torch.randn(3,2,dtype=torch.double)
 def test_dual_norm_and_training_direction(self):
  h=torch.zeros_like(self.x,requires_grad=True);p,b=unroll(self.m,self.x+h,self.y,3,.03);f=torch.relu(.5-functional_call(self.m,(p,b),(self.q,)).sigmoid()).mean();g=torch.autograd.grad(f,h)[0];direction=math.sqrt(len(h))*g/g.norm();eps=1e-5
  def value(t):
   p,b=unroll(self.m,self.x+t*direction,self.y,3,.03);return torch.relu(.5-functional_call(self.m,(p,b),(self.q,)).sigmoid()).mean().item()
  fd=(value(eps)-value(-eps))/(2*eps);self.assertAlmostEqual(fd,math.sqrt(len(h))*g.norm().item(),places=8)
 def test_full_outer_gradient_includes_sensitivity(self):
  q=self.q.clone().requires_grad_();f,n,s=surrogate(self.m,self.x,self.y,q,.5,3,.03);gn=torch.autograd.grad(n,q,retain_graph=True)[0];g=torch.autograd.grad(f,q)[0];self.assertGreater((g-gn).norm().item(),1e-8);v=torch.randn_like(q);v/=v.norm();eps=1e-5
  a=surrogate(self.m,self.x,self.y,q.detach()+eps*v,.5,3,.03)[0].item();b=surrogate(self.m,self.x,self.y,q.detach()-eps*v,.5,3,.03)[0].item();self.assertAlmostEqual((a-b)/(2*eps),(g*v).sum().item(),places=7)
 def test_valid_population_flat_hinge_for_all_radii(self):
  with torch.no_grad():self.m.net[-1].bias.fill_(10.)
  for r in [0.,.25,.5,10.]:
   q=self.q.clone().requires_grad_();f,n,s=surrogate(self.m,self.x,self.y,q,r,3,.03)
   self.assertEqual(f.item(),0.);self.assertEqual(n.item(),0.);self.assertEqual(s.item(),0.);self.assertTrue(torch.equal(torch.autograd.grad(f,q)[0],torch.zeros_like(q)))
 def test_smooth_hinge_full_outer_gradient(self):
  q=self.q.clone().requires_grad_();f,n,s=surrogate(self.m,self.x,self.y,q,.5,3,.03,hinge_temperature=.05);g=torch.autograd.grad(f,q)[0];v=torch.randn_like(q);v/=v.norm();eps=1e-5
  a=surrogate(self.m,self.x,self.y,q.detach()+eps*v,.5,3,.03,hinge_temperature=.05)[0].item();b=surrogate(self.m,self.x,self.y,q.detach()-eps*v,.5,3,.03,hinge_temperature=.05)[0].item();self.assertAlmostEqual((a-b)/(2*eps),(g*v).sum().item(),places=7)
 def test_smooth_loss_has_gradient_above_validity_threshold(self):
  with torch.no_grad():self.m.net[-1].bias.fill_(.3)
  q=self.q.clone().requires_grad_();f,n,s=surrogate(self.m,self.x,self.y,q,.5,3,.03,hinge_temperature=.05)
  self.assertGreater(n.item(),0.);self.assertGreater(s.item(),0.);self.assertGreater(torch.autograd.grad(f,q)[0].norm().item(),0.)
 def test_geometry_gradient_and_zero(self):
  a=torch.tensor([[0.],[2.]],requires_grad=True);b=torch.tensor([[1.],[4.]])
  d,_=differentiable_w2(a,b);self.assertAlmostEqual(d.item(),math.sqrt(2.5),places=6);g=torch.autograd.grad(d,a)[0];self.assertTrue(torch.isfinite(g).all());zero,_=differentiable_w2(a,a.detach());self.assertEqual(zero.item(),0);self.assertTrue(torch.equal(torch.autograd.grad(zero,a)[0],torch.zeros_like(a)))
if __name__=='__main__':unittest.main()
