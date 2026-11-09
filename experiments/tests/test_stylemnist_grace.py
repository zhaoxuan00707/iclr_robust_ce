import copy
import unittest
import torch
from lift_ce.experiments.stylemnist_grace_v1 import project_delta,rms,data_w2,ImageBank,full_update
from lift_ce.counterfactuals.single_budget_solver import bank_value_gradient
from lift_ce.counterfactuals.minimal_validity_repair import solve_raw

class IdentityDecoder(torch.nn.Module):
    def decode(self,x):return x

class TestStyleMNIST(unittest.TestCase):
    def test_population_max_not_pointwise_max(self):
        a=torch.nn.Linear(1,1);b=torch.nn.Linear(1,1)
        with torch.no_grad():a.weight.fill_(3);b.weight.fill_(-3);a.bias.zero_();b.bias.zero_()
        q=torch.tensor([[-1.],[1.]],requires_grad=True)
        bank=ImageBank([a,b],IdentityDecoder())
        _,risk,grad=bank_value_gradient(bank,q)
        pointwise=torch.relu(.5-torch.sigmoid(bank.logits(q))).max(0).values.mean()
        self.assertAlmostEqual(risk,float(pointwise)/2,places=6)
        self.assertTrue(torch.isfinite(grad).all())
    def test_projection_and_exact_data_distance(self):
        x=torch.rand(8,1,3,3);delta=project_delta(x,torch.randn_like(x),.2)
        self.assertLessEqual(float(rms(delta)),.200001)
        self.assertTrue(((x+delta>=0)&(x+delta<=1)).all())
        distance=data_w2(x,x+delta,torch.tensor([0.,1.]*4),'cpu')
        self.assertLessEqual(distance,float(rms(delta))+1e-6)
    def test_raw_solver_has_no_current_model(self):
        model=torch.nn.Linear(1,1)
        with torch.no_grad():model.weight.fill_(1);model.bias.zero_()
        bank=ImageBank([model],IdentityDecoder())
        settings={'steps':2,'learning_rate':.03,'beta_h':50.,'beta_m':100.,'rho_initial':1.,'dual_interval':1,'stall_interval':2,'feasibility_relative_tolerance':1e-6,'restoration_steps':10}
        z=torch.tensor([[-2.],[-1.]]);target=torch.tensor([[1.],[2.]])
        rows,_=solve_raw(bank,z,target,(z+target)/2,3.,3.3,float('-inf'),float('inf'),settings)
        self.assertTrue(rows)
        self.assertTrue(all(r['slack']>=-3e-6 for r in rows))
    def test_update_fixed_seed_and_frozen_base(self):
        base=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,1),torch.nn.Flatten(0))
        old=copy.deepcopy(base.state_dict());x=torch.rand(8,1,2,2);y=torch.tensor([0.,1.]*4)
        c={'update':{'seed':42,'lr':.001,'weight_decay':.0001,'epochs':2,'batch_size':4}}
        a=full_update(base,x,y,c,'cpu');b=full_update(base,x,y,c,'cpu')
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        for k,v in old.items():torch.testing.assert_close(v,base.state_dict()[k],atol=0,rtol=0)

if __name__=='__main__':unittest.main()
