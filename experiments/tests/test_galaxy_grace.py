import unittest
import torch
from lift_ce.experiments.galaxy_grace_v1 import target_logit,surrogate_scores,project_delta,paired_distance
from lift_ce.counterfactuals.single_budget_solver import bank_value_gradient

class TestGalaxy(unittest.TestCase):
    def test_target_score_preserves_multiclass_probability(self):
        logits=torch.randn(9,10)
        for target in range(10):torch.testing.assert_close(target_logit(logits,target).sigmoid(),logits.softmax(1)[:,target])
    def test_data_projection_uses_full_training_population(self):
        x=torch.rand(8,3,4,4);d=project_delta(x,torch.randn_like(x),.2,100)
        self.assertLessEqual(float(paired_distance(d,100)),.200001)
        self.assertTrue(((x+d>=0)&(x+d<=1)).all())
    def test_multiclass_hypergradient(self):
        m=torch.nn.Linear(4,10);x=torch.randn(3,4);d=torch.zeros_like(x,requires_grad=True)
        p=surrogate_scores(m,x,torch.tensor([1,3,7]),d,torch.randn(2,4),torch.tensor([4,2]))
        g=torch.autograd.grad(p.mean(),d)[0];self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
    def test_streaming_population_aggregation(self):
        class Bank:
            count=2;chunk_size=1
            def logits(self,x,start,end):
                self_assert=(end-start)==1
                if not self_assert:raise AssertionError('did not respect streaming')
                return ((1 if start==0 else -1)*3*x[:,0])[None]
        q=torch.tensor([[-1.],[1.]],requires_grad=True)
        _,risk,gradient=bank_value_gradient(Bank(),q)
        expected=float(torch.relu(.5-torch.sigmoid(3*q[:,0])).mean())
        self.assertAlmostEqual(risk,expected,places=7);self.assertTrue(torch.isfinite(gradient).all())
if __name__=='__main__':unittest.main()
