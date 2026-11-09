import copy
import unittest
import torch
from torch.func import functional_call
from lift_ce.models.tabular import TabularMLP
from lift_ce.counterfactuals.update_aware import unroll
from lift_ce.counterfactuals.mixed_update_aware import (
    MixedDomain, mixed_surrogate, sensitivity_chunks, sensitivity_aggregation_diagnostics)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.x=torch.tensor([[.2,-1.,2.],[-.7,2.,-1.],[1.,-1.,2.],[-1.,2.,-1.]],dtype=torch.float64)
        self.y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64)
        self.model=TabularMLP(3,[4]).double()
        self.domain=MixedDomain(self.x,{'category':[1,2]})
        self.q=(self.x[:3]+.13).requires_grad_(True)
    def value(self,q,mode='population',chunk=1,r=.3):
        return mixed_surrogate(self.model,self.x,self.y,q,r,self.domain,steps=3,
                               adversary_mode=mode,sensitivity_chunk_size=chunk)
    def test_original_population_value_and_gradient(self):
        u=torch.zeros_like(self.x,requires_grad=True)
        p,b=unroll(self.model,self.domain.perturb(self.x,u),self.y,3,.01,True)
        nominal=(.05*torch.nn.functional.softplus((.5-functional_call(self.model,(p,b),(self.q,)).sigmoid())/.05)).mean()
        sensitivity=len(self.x)**.5*torch.autograd.grad(nominal,u,create_graph=True)[0].norm()
        old=nominal+.3*sensitivity;new=self.value(self.q)[0]
        self.assertTrue(torch.equal(old,new))
        self.assertTrue(torch.equal(torch.autograd.grad(old,self.q)[0],torch.autograd.grad(new,self.q)[0]))
    def test_mean_vectors_and_triangle(self):
        d=sensitivity_aggregation_diagnostics(self.model,self.x,self.y,self.q,self.domain,steps=3,chunk_size=2)
        self.assertLess(d['mean_vector_max_error'],1e-14)
        self.assertLessEqual(d['population_sensitivity'],d['individual_sensitivity']+1e-14)
        self.assertAlmostEqual(d['population_sensitivity'],float(self.value(self.q)[2]),places=12)
    def test_single_cf(self):
        for a,b in zip(self.value(self.q[:1]),self.value(self.q[:1],'individual')):
            self.assertTrue(torch.allclose(a,b,atol=1e-14,rtol=1e-12))
    def test_chunk_value_and_gradient(self):
        a=self.value(self.q,'individual',1)[0];b=self.value(self.q,'individual',3)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-14,rtol=1e-12))
        self.assertTrue(torch.allclose(torch.autograd.grad(a,self.q)[0],torch.autograd.grad(b,self.q)[0],atol=1e-13,rtol=1e-10))
    def test_individual_directional_derivative(self):
        value=self.value(self.q,'individual',2)[0];g=torch.autograd.grad(value,self.q)[0]
        v=torch.randn_like(self.q);v/=v.norm();eps=1e-5
        fd=(self.value(self.q+eps*v,'individual',2)[0]-self.value(self.q-eps*v,'individual',2)[0])/(2*eps)
        self.assertLess(float((fd-(g*v).sum()).abs()),1e-8)
    def test_zero_radius(self):
        a=self.value(self.q,r=0)[0];b=self.value(self.q,'individual',r=0)[0]
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(torch.autograd.grad(a,self.q)[0],torch.autograd.grad(b,self.q)[0]))
    def test_zero_denominator(self):
        # Zero classifier weights and zero update steps make loss data-independent.
        # A zero learning rate retains a graph, with mathematically zero pullback.
        d=sensitivity_aggregation_diagnostics(self.model,self.x,self.y,self.q,self.domain,steps=1,lr=0.)
        self.assertTrue(d['zero_denominator']);self.assertIsNone(d['kappa'])
    def test_same_geometry_and_old_optimizer(self):
        from lift_ce.experiments.appendix_tabular import grace
        import ast
        from pathlib import Path
        import lift_ce.experiments.appendix_tabular as module
        source=Path(__file__).parent.joinpath('fixtures/task102_legacy_grace.py').read_text()
        old_node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='grace')
        namespace=vars(module).copy();exec(compile(ast.Module(body=[old_node],type_ignores=[]),'old_grace','exec'),namespace)
        cfg=dict(solver=dict(steps=3,primal_lr=.1,rho=1.,obj_tol=1e-6,feas_tol=1e-5),update=dict(steps=3,lr=.01),smooth_temperature=.05,categorical_temperature=1.,categorical_smoothing=1e-4)
        base=self.x[:2].clone();p1=self.x[2:].clone()
        old=namespace['grace'](self.model,self.x,self.y,base,p1,self.domain,.3,2.,cfg)
        new=grace(self.model,self.x,self.y,base,p1,self.domain,.3,2.,cfg)
        self.assertTrue(torch.equal(old['q'],new['q']))
        self.assertEqual(old['history'],new['history'])
        ind=grace(self.model,self.x,self.y,base,p1,self.domain,.3,2.,cfg,'individual',2)
        for o in [old,new,ind]:
            self.assertTrue(o['feasible']);self.assertEqual(o['budget'],new['budget'])
            self.assertTrue(torch.equal(o['q'][o['assignment']],o['cf']))
            self.assertEqual(len(o['assignment'].unique()),len(base))

if __name__=='__main__':unittest.main()
