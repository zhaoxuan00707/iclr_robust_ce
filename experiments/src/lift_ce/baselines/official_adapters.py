"""Thin adapters for official external baselines; no fallback algorithms."""
from __future__ import annotations
import pandas as pd
import torch

def _tabular_wrapper_class(input_scale=None,input_shift=None):
    from ellice.models.wrappers import ModelWrapper
    class FixedAffine(torch.nn.Module):
        def __init__(self,scale,shift):
            super().__init__();self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32));self.register_buffer('shift',torch.as_tensor(shift,dtype=torch.float32))
        def forward(self,x):return x*self.scale+self.shift
    class TabularMLPWrapper(ModelWrapper):
        def __init__(self,model,backend='custom'):
            super().__init__(model,backend);dim=model.features[0].in_features if len(model.features) else model.classifier.in_features
            scale=torch.ones(dim) if input_scale is None else torch.as_tensor(input_scale,dtype=torch.float32)
            shift=torch.zeros(dim) if input_shift is None else torch.as_tensor(input_shift,dtype=torch.float32)
            self.affine=FixedAffine(scale,shift);self.wrapped=torch.nn.Sequential(self.affine,self.model)
        def get_torch_model(self): return self.wrapped
        def split_model(self):
            last=self.model.classifier
            return torch.nn.Sequential(self.affine,self.model.features),torch.cat([last.weight.detach().reshape(-1),last.bias.detach()])
        def predict_proba(self,x):
            # ElliCE may move the wrapped model to CUDA.  Its NumPy-facing
            # callback must follow that device and return host NumPy values.
            device=next(self.wrapped.parameters()).device
            value=torch.as_tensor(x,dtype=torch.float32).clone().to(device)
            with torch.no_grad(): p=torch.sigmoid(self.wrapped(value)).reshape(-1,1)
            return torch.cat([1-p,p],1).cpu().numpy()
    return TabularMLPWrapper

def generate_ellice_cf(factual, base_model, training_x, training_y, feature_names,
                       target_class=1, robustness_epsilon=.01,
                       one_hot_groups=None, device='cpu',model_input_scale=None,
                       model_input_shift=None,permitted_range=None,allowed_values=None,requires="valid"):
    """Call the official ElliCE continuous generator with native defaults."""
    import ellice
    frame=pd.DataFrame(training_x,columns=feature_names);frame['target']=training_y
    data=ellice.Data(dataframe=frame,target_column='target')
    exp=ellice.Explainer(model=base_model,data=data,backend='custom',
        backend_model_class=_tabular_wrapper_class(model_input_scale,model_input_shift),device=device)
    return exp.generate_counterfactuals(
        query_instances=pd.Series(factual,index=feature_names),method='continuous',
        target_class=target_class,robustness_epsilon=robustness_epsilon,
        features_to_vary='all',one_hot_groups=one_hot_groups,
        permitted_range=permitted_range,allowed_values=allowed_values,
        return_probs=True,progress_bar=False,requires=requires)

def require_proplace_license():
    """Fail before PROPLACE construction unless the official solver is usable."""
    import gurobipy as gp
    model=gp.Model('proplace_license_check');x=model.addVar(lb=0);model.setObjective(x,gp.GRB.MAXIMIZE);model.addConstr(x<=1);model.optimize()
    if model.Status!=gp.GRB.OPTIMAL: raise RuntimeError(f'Gurobi status {model.Status}')
    from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
    return PROPLACE
