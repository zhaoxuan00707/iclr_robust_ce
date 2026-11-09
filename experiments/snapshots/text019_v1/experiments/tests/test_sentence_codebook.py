import unittest,torch,numpy as np
from lift_ce.counterfactuals.sentence_codebook import update,losses,cache,objective,solve,geometry,w2
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(7);self.x=torch.randn(6,3,dtype=torch.double);self.y=torch.tensor([0,1,0,1,0,1],dtype=torch.double);self.w=torch.randn(3,dtype=torch.double);self.b=torch.tensor(.2,dtype=torch.double);self.q=torch.randn(5,3,dtype=torch.double)
 def test_full_sgd(self):
  w=self.w.clone().requires_grad_();b=self.b.clone().requires_grad_();o=torch.optim.SGD([w,b],lr=.5)
  for _ in range(3):o.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(self.x@w+b,self.y).backward();o.step()
  a,c=update(self.x,self.y,self.w,self.b);torch.testing.assert_close(a,w);torch.testing.assert_close(c,b)
 def test_cache_and_directional(self):
  f,g=cache(self.x,self.y,self.q,self.w,self.b);ids=torch.tensor([0,2,2,4]);x=self.x.clone().requires_grad_();w,b=update(x,self.y,self.w,self.b);nom=losses(self.q[ids],w,b).mean();grad=torch.autograd.grad(nom,x)[0];o,n,s=objective(ids,f,g,6,.25);torch.testing.assert_close(n,nom);torch.testing.assert_close(s,6**.5*grad.norm());d=torch.randn_like(x);d/=d.norm();e=1e-5
  def fun(xx):ww,bb=update(xx,self.y,self.w,self.b);return losses(self.q[ids],ww,bb).mean()
  torch.testing.assert_close((fun(x+e*d)-fun(x-e*d))/(2*e),(grad*d).sum(),atol=1e-9,rtol=1e-6)
 def test_cancellation(self):
  _,_,s=objective(torch.tensor([0,1]),torch.zeros(2),torch.tensor([[1.,0.],[-1.,0.]]),2,1);self.assertEqual(float(s),0)
 def test_discrete_geometry(self):
  codes=np.array([[0.,0.],[1.,0.],[.2,0.],[.8,0.],[.5,.1]]);base=[0,1];ref=codes[[2,3]];f=np.array([1,1,.1,.2,.05]);g=np.zeros((5,4));q,deployed,tr,stop=solve(codes,base,ref,f,g,2,0,1.1);self.assertLessEqual(geometry(codes[base],codes[q],ref),1.1*w2(codes[base],ref)[0]+1e-8);self.assertLessEqual(f[q].mean(),1);self.assertEqual(sorted(q),sorted(deployed));self.assertTrue(stop)
if __name__=='__main__':unittest.main()
