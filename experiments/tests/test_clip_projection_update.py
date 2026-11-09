import unittest
import torch
import torch.nn.functional as F
from lift_ce.counterfactuals.clip_projection_update import projection_update
from lift_ce.counterfactuals.frozen_feature_sensitivity import pixel_sensitivity,smooth_target
class ProjectionTests(unittest.TestCase):
 def test_full_pixel_and_Q_gradient(self):
  torch.manual_seed(18);dt=torch.float64
  A=torch.randn(3,5,dtype=dt);extract=lambda x:torch.tanh(x@A)
  x=torch.randn(4,3,dtype=dt);y=torch.tensor([0,1,0,1]);p0=torch.randn(5,4,dtype=dt);text=F.normalize(torch.randn(2,4,dtype=dt),dim=-1);q=torch.randn(3,3,dtype=dt,requires_grad=True)
  def loss(z):return smooth_target(2*F.normalize(extract(q)@z,dim=-1)@text.T,0)
  h=x.clone().requires_grad_();p=projection_update(extract(h),y,p0,text,2.,3,.01);n=loss(p);s=2*torch.autograd.grad(n,h,create_graph=True)[0].norm();expected=torch.autograd.grad(n+.25*s,q)[0]
  features=extract(x).detach().requires_grad_();p=projection_update(features,y,p0,text,2.,3,.01);n=loss(p);adj=torch.autograd.grad(n,features,create_graph=True)[0];v=pixel_sensitivity(adj,x,extract,2);actual=torch.autograd.grad(n+.25*v,q)[0]
  torch.testing.assert_close(s,v);torch.testing.assert_close(expected,actual)
 def test_matches_optimizer(self):
  torch.manual_seed(2);x=torch.randn(6,5,dtype=torch.double);y=torch.arange(6)%2;p0=torch.randn(5,4,dtype=torch.double);t=F.normalize(torch.randn(2,4,dtype=torch.double),dim=-1)
  actual=projection_update(x,y,p0,t,2.,3,.01,False);p=p0.clone().requires_grad_();opt=torch.optim.SGD([p],lr=.01)
  for _ in range(3):opt.zero_grad();F.cross_entropy(2*F.normalize(x@p,dim=-1)@t.T,y).backward();opt.step()
  torch.testing.assert_close(actual,p)
if __name__=='__main__':unittest.main()
