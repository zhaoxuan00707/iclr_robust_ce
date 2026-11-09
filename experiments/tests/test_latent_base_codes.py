import unittest
import torch
from torch import nn
from lift_ce.config import BaseCEConfig
from lift_ce.counterfactuals.vision import generate_near_boundary_ce
class AE(nn.Module):
 def encode(self,x):return x*.5
 def decode(self,z):return z*3
class Classifier(nn.Module):
 def forward(self,x):return (x[:,0]-.5)*10
class TestLatentCodes(unittest.TestCase):
 def test_preserve_generator_code_not_reencoding(self):
  ae=AE();m=Classifier();r=generate_near_boundary_ce(m,ae,torch.tensor([[0.],[.1]]),torch.tensor([[1.]]),BaseCEConfig(steps=20),torch.device('cpu'))
  self.assertTrue(torch.isfinite(r.latents).all());torch.testing.assert_close(ae.decode(r.latents),r.counterfactuals,rtol=0,atol=0);self.assertFalse(torch.equal(ae.encode(r.counterfactuals),r.latents));self.assertTrue((r.final_probabilities>=.52).all())
if __name__=='__main__':unittest.main()
