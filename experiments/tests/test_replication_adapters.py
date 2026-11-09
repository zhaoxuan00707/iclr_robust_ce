import unittest
import numpy as np
import pandas as pd
import torch
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from lift_ce.experiments.run_official_proplace import FrozenFrameDataset
from lift_ce.baselines.tabular_replication_adapters import SignedDeltaEvaluator, PositiveSTCE

class ReplicationTests(unittest.TestCase):
    def task(self,hidden=[]):
        model=SimpleNNModel(2,hidden,1,seed=0)
        with torch.no_grad():
            for v in model.model:
                if isinstance(v,torch.nn.Linear): v.weight.fill_(.5);v.bias.fill_(2.)
        data=FrozenFrameDataset(seed=0)
        data.data=pd.DataFrame({'a':[-2.,1.], 'b':[1.,2.], 'target':[0,1]})
        return ClassificationTask(model,data)
    def test_signed_linear_matches_exact(self):
        task=self.task(); evaluator=SignedDeltaEvaluator(task)
        for x in ([-2.,1.],[-4.,-2.],[1.,2.]):
            for delta in (.01,.5,1.):
                exact=.5*sum(x)+2-delta*(sum(abs(v) for v in x)+1)
                self.assertEqual(evaluator.evaluate(pd.Series(x),delta=delta,bias_delta=delta),exact>0)
    def test_relu_zero_radius(self):
        task=self.task([2]); evaluator=SignedDeltaEvaluator(task)
        for x in ([-2.,1.],[-4.,-2.],[1.,2.]):
            self.assertTrue(evaluator.evaluate(pd.Series(x),delta=0,bias_delta=0))
    def test_proplace_signed_output_and_adversarial_weights(self):
        from lift_ce.baselines.tabular_replication_adapters import corrected_proplace
        from gurobipy import Model, GRB
        for hidden in ([],[2]):
            task=self.task(hidden)
            generator=corrected_proplace(task,torch.tensor([[-2.,1.],[1.,2.]]),{})
            generator.x_prime_current=pd.DataFrame([[-2.,1.]])
            generator.delta=.1
            with Model() as m:
                m.Params.OutputFlag=0
                nodes={};nodes[0],m=generator._adv_add_inputs(m)
                nodes,_,m=generator._adv_add_nodes(nodes,{},m)
                m.setObjective(nodes[len(generator.layers)-1][0],GRB.MINIMIZE)
                m.Params.NonConvex=2;m.optimize()
                self.assertEqual(m.Status,GRB.OPTIMAL)
                first=.5*(-2+1)+2-.1*(2+1+1)
                exact=first if not hidden else 2*(.5-.1)*max(0,first)+2-.1
                self.assertAlmostEqual(m.ObjVal,exact,places=5)
                last=len(generator.layers)-1
                reconstructed=sum(m.getVarByName(f'weight_l{last-1}_n{j}_to_l{last}_n0').X*nodes[last-1][j].X for j in range(generator.layers[last-1]))+m.getVarByName(f'bias_into_l{last}_n0').X
                self.assertAlmostEqual(m.ObjVal,reconstructed,places=5)
    def test_projected_milp_matches_explicit_adversary(self):
        from lift_ce.baselines.tabular_replication_adapters import corrected_proplace
        from gurobipy import Model, GRB
        for hidden in ([],[2],[2,2]):
            task=self.task(hidden)
            with torch.no_grad():
                layers=[v for v in task.model.model if isinstance(v,torch.nn.Linear)]
                for i,v in enumerate(layers):
                    v.weight.copy_(torch.tensor(np.random.default_rng(i+42).normal(size=v.weight.shape),dtype=torch.float32))
                    v.bias.fill_(.2)
            generator=corrected_proplace(task,torch.tensor([[-2.,1.],[1.,2.]]),{})
            generator.x_prime_current=pd.DataFrame([[-2.,1.]])
            generator.delta=.1
            for desired in (0,1):
                evaluator=SignedDeltaEvaluator(task)
                evaluator.evaluate(generator.x_prime_current.iloc[0],desired_output=desired,delta=.1,bias_delta=.1,M=1000)
                optimum=evaluator.opt.outputNode.X
                wp,bp=generator._recover_adversarial_weights(evaluator.opt.gurobiModel,optimum)
                self.assertTrue(all(abs(wp[k]-v)<=.100001 for k,v in generator.orig_w.items()))
                with Model() as m:
                    m.Params.OutputFlag=0
                    nodes={};nodes[0],m=generator._adv_add_inputs(m)
                    nodes,_,m=generator._adv_add_nodes(nodes,{},m)
                    m.setObjective(nodes[len(generator.layers)-1][0],GRB.MINIMIZE if desired else GRB.MAXIMIZE)
                    m.Params.NonConvex=2;m.optimize()
                    self.assertEqual(m.Status,GRB.OPTIMAL)
                    self.assertAlmostEqual(m.ObjVal,optimum,places=4)
                evaluator.opt.gurobiModel.dispose()
    def test_trex_positive_candidate(self):
        generator=PositiveSTCE(self.task())
        generator.counterfactual_stability=lambda x: 1.
        value=generator.generate_for_instance(pd.Series([-2.,1.],index=['a','b']))
        np.testing.assert_array_equal(value.to_numpy(),[[1.,2.]])
if __name__=='__main__': unittest.main()
