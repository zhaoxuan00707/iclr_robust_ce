"""Opt-in categorical-domain adaptations; canonical baseline defaults are unchanged.

Relaxed mode matches the standardized simplex used by Task100 GRACE. Hard mode
is a projected-gradient ROAR heuristic / binary-constrained PROPLACE extension.
Neither rounding nor the ROAR projection supplies a robustness certificate.
"""
from ..counterfactuals.mixed_update_aware import MixedDomain


def categorical_projector(train_x, groups, hard=False):
    domain = MixedDomain(train_x, groups)

    def project(value):
        return domain.project(value.unsqueeze(0), hard=hard).squeeze(0)

    return project


def add_categorical_constraints(model, inputs, domain, hard=False):
    """z_j = low_j + scale_j * p_j; p in simplex or one-hot vertices."""
    from gurobipy import GRB, quicksum
    for group_index, ids in enumerate(domain.groups.values()):
        p = [model.addVar(lb=0., ub=1., vtype=GRB.BINARY if hard else GRB.CONTINUOUS,
                         name=f'category_{group_index}_{j}') for j in ids]
        model.addConstr(quicksum(p) == 1., name=f'category_sum_{group_index}')
        for j, var in zip(ids, p):
            model.addConstr(inputs[f'v_0_{j}'] == float(domain.low[j]) + float(domain.scale[j])*var,
                            name=f'category_affine_{group_index}_{j}')
    return model


def mixed_proplace(task, train_x, groups, hard=False):
    """Return a fresh official PROPLACE instance with only input-domain additions."""
    from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
    domain = MixedDomain(train_x, groups)

    class CategoricalPROPLACE(PROPLACE):
        def _master_prob_add_inputs(self, model):
            inputs, model = super()._master_prob_add_inputs(model)
            add_categorical_constraints(model, inputs, domain, hard=hard)
            return inputs, model

    return CategoricalPROPLACE(task)
