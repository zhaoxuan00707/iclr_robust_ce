import unittest
import torch
from lift_ce.counterfactuals.latent_boundary import first_sampled_crossing
class BoundaryTests(unittest.TestCase):
 def test_valid_upper_endpoint(self):
  r=first_sampled_crossing(lambda a:.2+.7*a,3,.5001,intervals=64)
  self.assertTrue(r['success'].all());self.assertTrue((r['lower_probability']<.5001).all());self.assertTrue((r['upper_probability']>=.5001).all());self.assertTrue((r['upper_probability']<=.5002).all())
 def test_nonmonotonic_first_sampled_crossing(self):
  def probability(a):return torch.where(a<.4,.1+2*a,torch.where(a<.8,.1+0*a,.9+0*a))
  r=first_sampled_crossing(probability,1,.5,intervals=100)
  self.assertTrue(r['success'].item());self.assertAlmostEqual(r['alpha'].item(),.2,places=5)
 def test_already_positive_keeps_original_endpoint(self):
  r=first_sampled_crossing(lambda a:.7+.1*a,2,.52,intervals=16)
  self.assertFalse(r['success'].any());self.assertTrue(torch.equal(r['alpha'],torch.ones(2)));self.assertEqual(r['reason'],['start_already_positive']*2)
 def test_unbracketed_endpoint_keeps_original(self):
  r=first_sampled_crossing(lambda a:.1+.2*a,1,.52,intervals=16)
  self.assertFalse(r['success'].item());self.assertEqual(r['alpha'].item(),1);self.assertEqual(r['reason'][0],'endpoint_below_target')
 def test_discontinuous_crossing_fails_probability_tolerance(self):
  r=first_sampled_crossing(lambda a:torch.where(a<.5,.1,.9),1,.52,intervals=16)
  self.assertFalse(r['success'].item());self.assertEqual(r['alpha'].item(),1)
if __name__=='__main__':unittest.main()
