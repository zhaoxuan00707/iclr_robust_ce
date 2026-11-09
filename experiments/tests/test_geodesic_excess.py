import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from lift_ce.counterfactuals.geodesic_excess import exact_w2,excess,glued_arclength_path,project_basic
from lift_ce.counterfactuals.geodesic_excess_solver import bank_value_gradient,solve_control
from lift_ce.counterfactuals.robust_waypoints import ModelBank

class GeodesicExcessTests(unittest.TestCase):
    def test_final_selection_rechecks_current_validity_in_reference_model(self):
        from lift_ce.experiments import geodesic_excess_wro_v1 as s
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'optimization').mkdir()
            (out/'resolved_config.json').write_text(json.dumps({'D':2.}))
            torch.save({'path':{'base':torch.tensor([[-1.]])}},out/'train_population.pt')
            model=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Flatten(0))
            with torch.no_grad():model[0].weight.fill_(1);model[0].bias.fill_(-.5)
            train={'target':torch.tensor([[1.]]),'x':torch.tensor([[-2.],[2.]])}
            for k in [0.,.2]:
                # Both cached flags claim validity. The lower-risk point fails
                # the actual reporting model and must not win either budget.
                candidates=[{'points':torch.tensor([[x]]),'true_bank_hinge':risk,
                    'E_geo':0.,'current_invalid_mass':0.,'box_violation_mass':0.,'tag':'test','step':0}
                    for x,risk in [(0.,.1),(1.,.3)]]
                file=out/'optimization'/f'k{int(k*100):03d}_target.pt'
                torch.save({'construction':{},'candidates':candidates},file)
                pd.DataFrame([{'step':0,'nu':0.,'rho_aug':1.,'g':0.,'kappa_ratio':k,'initialization':'target'}]).to_csv(file.with_suffix('.csv'),index=False)
            with patch.object(s.e,'context',return_value=(out,{},train,model)),patch.object(s,'verify_construction',return_value={}),patch.object(s,'KAPPAS',[0.,.2]),patch.object(s,'INITIALIZATIONS',['target']),patch.object(s,'build_paths',side_effect=lambda *a,**kw:torch.save({},out/'paths_train.pt')):
                s.freeze_controls(SimpleNamespace())
            choices=json.loads((out/'controls_manifest.json').read_text())
            self.assertTrue(all(c['true_bank_hinge']==.3 and c['current_invalid_mass']==0 for c in choices))

    def test_streamed_objective_is_population_then_model_max(self):
        models=[]
        for sign in [-1.,1.]:
            m=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Flatten(0))
            with torch.no_grad():m[0].weight.fill_(20*sign);m[0].bias.zero_()
            models.append(m)
        bank=ModelBank(models,'cpu');q=torch.tensor([[-1.],[1.]],requires_grad=True)
        value,risk,gradient=bank_value_gradient(bank,q,chunk=1)
        self.assertAlmostEqual(risk,.25,places=6)
        pred=torch.sigmoid(bank.logits(q))
        loss=torch.logsumexp(100*(torch.nn.functional.softplus(50*(.5-pred))/50).double().mean(1),0)/100
        torch.testing.assert_close(gradient,torch.autograd.grad(loss,q)[0])

    def test_primal_dual_solver_returns_feasible_improvement_in_one_dimension(self):
        model=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Flatten(0))
        current=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Flatten(0))
        with torch.no_grad():
            model[0].weight.fill_(1);model[0].bias.fill_(-2)
            current[0].weight.zero_();current[0].bias.fill_(10)
        bank=ModelBank([model],'cpu');z=torch.tensor([[0.]]);y=torch.tensor([[1.]])
        settings={'steps':30,'learning_rate':.03,'beta_h':50.,'beta_m':100.,'rho_initial':1.,
                  'dual_interval':5,'stall_interval':20,'feasibility_relative_tolerance':1e-6}
        candidates,_=solve_control(bank,current,z,y,y,1.,.4,torch.tensor([-10.]),torch.tensor([10.]),settings)
        best=min(candidates,key=lambda a:a['true_bank_hinge'])
        self.assertLess(best['true_bank_hinge'],.5-float(torch.sigmoid(torch.tensor(-1.))))
        self.assertLessEqual(best['E_geo'],.4+1e-6)

    def test_assignment_matches_network_simplex_and_zero_distance_gradient(self):
        import ot
        x=torch.tensor([[0.,0.],[2.,1.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([[3.,2.],[1.,0.]],dtype=torch.float64)
        p=exact_w2(x,y)
        expected=ot.emd2([.5,.5],[.5,.5],torch.cdist(x,y).square().detach().numpy())
        self.assertAlmostEqual(p.distance**2,expected)
        zero=exact_w2(y,y).differentiable_distance(y.clone().requires_grad_(),y)
        self.assertEqual(float(zero),0.)

    def test_exact_envelope_gradient_matches_finite_difference(self):
        x=torch.tensor([[0.,0.],[2.,0.]],dtype=torch.float64)
        y=torch.tensor([[.5,1.],[3.,1.],[4.,2.]],dtype=torch.float64,requires_grad=True)
        p=exact_w2(x,y);g=torch.autograd.grad(p.differentiable_distance(x,y),y)[0]
        for row,col in [(0,0),(1,1),(2,0)]:
            a=y.detach().clone();b=a.clone();a[row,col]+=1e-5;b[row,col]-=1e-5
            finite=(exact_w2(x,a).distance-exact_w2(x,b).distance)/2e-5
            self.assertAlmostEqual(float(g[row,col]),finite,places=6)

    def test_glued_distribution_knots_arclength_and_triangle(self):
        z=torch.tensor([[0.,0.],[1.,0.]])
        y=torch.tensor([[3.,0.],[4.,0.],[5.,0.]])
        w=y+torch.tensor([0.,1.])
        p0,p1,E=excess(z,w,y);self.assertGreater(E,0)
        pivot=p0.distance/(p0.distance+p1.distance)
        p=glued_arclength_path(z,w,y,p0,p1,[0.,pivot,1.])
        torch.testing.assert_close(p['q'][0],z[p['source_index']])
        torch.testing.assert_close(p['q'][1],w[p['control_index']])
        torch.testing.assert_close(p['q'][2],y[p['target_index']])
        for key,count in [('source_index',2),('control_index',3),('target_index',3)]:
            np.testing.assert_allclose(torch.bincount(p[key],weights=p['weights']),1/count,atol=1e-10)
        self.assertNotEqual(pivot,.5)
        a,b,E=excess(z,y,y);self.assertAlmostEqual(E,0.)
        end=glued_arclength_path(z,y,y,a,b,[0.,.5,1.]);self.assertTrue(torch.isfinite(end['q']).all())

    def test_basic_projection_has_no_target_ball(self):
        model=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Flatten(0))
        with torch.no_grad():model[0].weight.fill_(1);model[0].bias.zero_()
        result=project_basic(torch.tensor([[-2.],[9.]]),torch.tensor([[1.],[1.]]),model,torch.tensor([-10.]),torch.tensor([10.]))
        self.assertTrue((model(result)>=0).all());self.assertEqual(float(result[1]),9.)

if __name__=='__main__':unittest.main()
