"""Explicit task103 fixes, leaving vendored implementations unchanged."""
import numpy as np
import pandas as pd
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from robustx.generators.robust_CE_methods.STCE import STCE


class SignedDeltaEvaluator(DeltaRobustnessEvaluator):
    """Repair first-layer interval products for fixed signed inputs."""
    def evaluate(self, instance, desired_output=1, delta=.005, bias_delta=.005,
                 M=10000, epsilon=.0001):
        from gurobipy import Model, GRB
        import torch
        self.opt.gurobiModel = m = Model()
        self.opt.setup(instance, delta=delta, bias_delta=bias_delta, M=M)
        x = np.asarray(instance).reshape(-1)
        layers = [v for v in self.task.model.model if isinstance(v, torch.nn.Linear)]
        layer = layers[0]
        centers = layer.weight.detach().numpy() @ x + layer.bias.detach().numpy()
        radius = delta * np.abs(x).sum() + bias_delta
        m.update()
        if len(layers) == 1:
            for name in ('output_node_C1', 'output_node_C2'):
                m.remove(m.getConstrByName(name))
            m.addConstr(self.opt.outputNode >= float(centers[0]-radius))
            m.addConstr(self.opt.outputNode <= float(centers[0]+radius))
        else:
            for j, center in enumerate(centers):
                node = m.getVarByName(f'v_1_{j}')
                active = m.getVarByName(f'xi_1_{j}')
                for prefix in ('constr3_', 'constr4_'):
                    m.remove(m.getConstrByName(prefix+f'v_1_{j}'))
                m.addConstr(node >= float(center-radius))
                m.addConstr(node <= float(center+radius)+M*active)
        m.setObjective(self.opt.outputNode, GRB.MINIMIZE if desired_output else GRB.MAXIMIZE)
        m.optimize()
        return m.Status == GRB.OPTIMAL and (self.opt.outputNode.X > 0 if desired_output else self.opt.outputNode.X < 0)


class PositiveSTCE(STCE):
    """Same stability score/check, corrected favorable-label candidate pool."""
    def _generation_method(self, instance, column_name='target', neg_value=0,
                           threshold=.4, **kwargs):
        candidates = self.task.training_data.data
        candidates = candidates[candidates[column_name] != neg_value].drop(columns=column_name)
        order = np.argsort(np.linalg.norm(candidates.to_numpy()-instance.to_numpy(), axis=1), kind='stable')
        evaluator = SignedDeltaEvaluator(self.task)
        for j in order:
            value = candidates.iloc[j]
            if evaluator.evaluate(value, desired_output=1-neg_value) and self.counterfactual_stability(value) > threshold:
                return pd.DataFrame(value).T
        return pd.DataFrame(instance).T


class CorrectedPROPLACEAdversary:
    """Link output-layer adversarial weights to the output, including signed LR inputs.

    The vendor output interval assumes nonnegative activations and leaves its
    output weight variables unused. An equality with the existing bounded
    weights repairs both issues without changing the parameter uncertainty set.
    """
    def _adv_add_inputs(self, model):
        if self.x_prime_current is None or self.x_prime_current.empty:
            raise RuntimeError('PROPLACE master problem returned no solution')
        return super()._adv_add_inputs(model)

    def _adv_add_nodes(self, node_vars, aux_vars, model):
        from gurobipy import quicksum
        nodes, aux, model = super()._adv_add_nodes(node_vars, aux_vars, model)
        model.update()
        # The vendored single-output implementation ends with these two linear
        # interval constraints. Replace them with its actual perturbed affine map.
        model.remove(model.getConstrs()[-2:])
        last = len(self.layers)-1
        weights = [model.getVarByName(f'weight_l{last-1}_n{j}_to_l{last}_n0')
                   for j in range(self.layers[last-1])]
        bias = model.getVarByName(f'bias_into_l{last}_n0')
        model.addConstr(nodes[last][0] == quicksum(w*nodes[last-1][j] for j,w in enumerate(weights))+bias)
        model.update()
        return nodes, aux, model

    def _adv_prob(self):
        # With fixed x and independent box-bounded weights, every row can attain
        # any preactivation in [w.h+b-d*(||h||_1+1), w.h+b+d*(||h||_1+1)].
        # Hidden h is nonnegative, making these bounds affine after layer one.
        # Thus the signed interval MILP is an exact projection of the weight
        # formulation (same inherited big-M), not an interval-propagation bound.
        from gurobipy import GRB
        if self.x_prime_current is None or self.x_prime_current.empty:
            raise RuntimeError('PROPLACE master problem returned no solution')
        evaluator=SignedDeltaEvaluator(self.task)
        evaluator.evaluate(self.x_prime_current.iloc[0],desired_output=1-self.neg_value,
                           delta=self.delta,bias_delta=self.delta,M=1000)
        model=evaluator.opt.gurobiModel
        try:
            if model.Status!=GRB.OPTIMAL:
                raise RuntimeError(f'PROPLACE adversary solver status {model.Status}; no robustness certificate')
            bound=evaluator.opt.outputNode.X
            wp,bp=self._recover_adversarial_weights(model,bound)
            if (self.neg_value==0 and bound>=0) or (self.neg_value!=0 and bound<0):
                self.achieved=True
                return
            self.w_primes.append(wp);self.b_primes.append(bp)
        finally:
            model.dispose()

    def _recover_adversarial_weights(self,model,bound):
        previous=self.x_prime_current.iloc[0].to_numpy(float)
        wp={};bp={}
        for layer in range(1,len(self.layers)):
            actual=[]
            for j in range(self.layers[layer]):
                keys=[f'weight_l{layer-1}_n{k}_to_l{layer}_n{j}' for k in range(len(previous))]
                bkey=f'bias_into_l{layer}_n{j}'
                center=sum(self.orig_w[k]*v for k,v in zip(keys,previous))+self.orig_b[bkey]
                radius=self.delta*(np.abs(previous).sum()+1)
                target=bound if layer==len(self.layers)-1 else model.getVarByName(f'v_{layer}_{j}').X
                preactivation=target if target>1e-7 or layer==len(self.layers)-1 else min(center,0.)
                coefficient=(preactivation-center)/radius if radius else 0.
                if abs(coefficient)>1+1e-4:
                    raise RuntimeError('Cannot reconstruct native PROPLACE adversarial weights')
                coefficient=np.clip(coefficient,-1,1)
                for key,value in zip(keys,previous): wp[key]=self.orig_w[key]+self.delta*coefficient*np.sign(value)
                bp[bkey]=self.orig_b[bkey]+self.delta*coefficient
                replay=sum(wp[k]*v for k,v in zip(keys,previous))+bp[bkey]
                if layer<len(self.layers)-1: replay=max(0.,replay)
                if abs(replay-target)>1e-4*max(1.,abs(target)):
                    raise RuntimeError('PROPLACE recovered adversarial model does not replay optimum')
                actual.append(replay)
            previous=np.asarray(actual)
        return wp,bp


def corrected_proplace(task,train_x,groups,hard=False,license_fallback=False):
    from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
    from .mixed_tabular_adapters import mixed_proplace
    parent=type(mixed_proplace(task,train_x,groups,hard)) if groups else PROPLACE
    class CorrectedPROPLACE(CorrectedPROPLACEAdversary,parent):
        pass
    if license_fallback:
        from .proplace_license_fallback import LicenseFallbackMaster
        class FallbackPROPLACE(LicenseFallbackMaster,CorrectedPROPLACE):
            pass
        return FallbackPROPLACE(task)
    return CorrectedPROPLACE(task)
