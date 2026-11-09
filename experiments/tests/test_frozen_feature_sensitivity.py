import unittest
import torch
from lift_ce.counterfactuals.frozen_feature_sensitivity import pixel_sensitivity,head_update,smooth_target

class SensitivityTests(unittest.TestCase):
 def test_full_update_value_and_outer_derivative(self):
  torch.manual_seed(17);dtype=torch.float64
  a=torch.randn(3,5,dtype=dtype);extract=lambda x:torch.tanh(x@a)
  x=torch.randn(4,3,dtype=dtype);y=torch.tensor([0,1,2,1]);w=torch.randn(3,5,dtype=dtype);b=torch.randn(3,dtype=dtype)
  q=torch.randn(3,3,dtype=dtype,requires_grad=True)
  h=x.clone().requires_grad_();ww,bb=head_update(extract(h),y,w,b,3,.04)
  f=smooth_target(torch.nn.functional.linear(extract(q),ww,bb),1)
  g=torch.autograd.grad(f,h,create_graph=True)[0];direct=2*g.norm();expected=torch.autograd.grad(f+.7*direct,q)[0]
  feat=extract(x).detach().requires_grad_();ww,bb=head_update(feat,y,w,b,3,.04)
  f=smooth_target(torch.nn.functional.linear(extract(q),ww,bb),1)
  adj=torch.autograd.grad(f,feat,create_graph=True)[0];s=pixel_sensitivity(adj,x,extract,2)
  actual=torch.autograd.grad(f+.7*s,q)[0]
  torch.testing.assert_close(s,direct);torch.testing.assert_close(actual,expected)
 def test_pixel_not_feature_norm(self):
  x=torch.zeros(2,3,dtype=torch.double);u=torch.ones(2,3,dtype=torch.double,requires_grad=True)
  s=pixel_sensitivity(u,x,lambda z:3*z,1)
  torch.testing.assert_close(s,3*(2**.5)*u.norm())
 def test_zero(self):
  x=torch.zeros(2,3,dtype=torch.double);u=torch.zeros_like(x,requires_grad=True)
  s=pixel_sensitivity(u,x,lambda z:2*z,1);g=torch.autograd.grad(s,u)[0]
  self.assertEqual(float(s),0);self.assertTrue(torch.isfinite(g).all());self.assertEqual(float(g.norm()),0)
if __name__=='__main__':unittest.main()
