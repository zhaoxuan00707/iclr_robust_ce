import unittest
import torch
from lift_ce.experiments.ffhq_grace_v1 import FeatureGradient,logits,surrogate
class Toy:
 def __init__(self):
  self.base_projection=torch.randn(3,2);self.text=torch.eye(2);self.scale=torch.tensor(2.)
 def features(self,x):return x.tanh()
 def canonical(self,x):return x
 def logits_features(self,f,projection=None):
  return self.scale*torch.nn.functional.normalize(f@(self.base_projection if projection is None else projection),dim=-1)@self.text.T
class TestFFHQ(unittest.TestCase):
 def test_recomputed_feature_gradient(self):
  class G:
   def decode(self,x):return x.sin()
  class B:
   g=G();c=Toy()
   def cached_features(self,w):return self.c.features(self.g.decode(w)).detach()
  w=torch.randn(4,3,requires_grad=True);direction=torch.randn(4,3)
  actual=torch.autograd.grad((FeatureGradient.apply(w,B())*direction).sum(),w)[0]
  expected=torch.autograd.grad((w.sin().tanh()*direction).sum(),w)[0]
  torch.testing.assert_close(actual,expected)
 def test_vectorized_bank_matches_separate_models(self):
  c=Toy();p=torch.randn(4,3,2);f=torch.randn(5,3)
  torch.testing.assert_close(logits(c,f,p),torch.stack([c.logits_features(f,a) for a in p]))
 def test_projection_update_pixel_hypergradient(self):
  c=Toy();x=torch.rand(3,3);d=torch.zeros_like(x,requires_grad=True);p=surrogate(c,x,torch.tensor([0,1,0]),d,torch.randn(2,3),.01)
  gradient=torch.autograd.grad(p.mean(),d)[0];self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)
if __name__=='__main__':unittest.main()
