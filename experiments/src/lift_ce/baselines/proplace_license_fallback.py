"""Equivalent linear-MIP backend fallback for PROPLACE's growing master."""
import time
import numpy as np
import pandas as pd


def solve_linear_highs(model,time_limit=30.):
    from scipy.optimize import milp,Bounds,LinearConstraint
    from gurobipy import GRB
    model.update()
    if model.NumQConstrs or model.NumQNZs or model.NumGenConstrs or model.NumSOS:
        raise ValueError('Fallback only supports ordinary linear mixed-integer models')
    variables=model.getVars();kinds=[v.VType for v in variables]
    if any(k not in (GRB.CONTINUOUS,GRB.INTEGER,GRB.BINARY) for k in kinds):
        raise ValueError('Unsupported variable type')
    c=np.asarray([v.Obj for v in variables])*model.ModelSense
    lb=np.asarray([v.LB for v in variables]);ub=np.asarray([v.UB for v in variables])
    lb[lb<=-GRB.INFINITY]= -np.inf;ub[ub>=GRB.INFINITY]=np.inf
    constraints=model.getConstrs();rhs=np.asarray([v.RHS for v in constraints]);sense=np.asarray([v.Sense for v in constraints])
    lower=np.where(sense=='<',-np.inf,rhs);upper=np.where(sense=='>',np.inf,rhs)
    matrix=model.getA().tocsc()
    result=milp(c,integrality=np.asarray([int(k!=GRB.CONTINUOUS) for k in kinds]),bounds=Bounds(lb,ub),constraints=LinearConstraint(matrix,lower,upper),options={'time_limit':time_limit,'mip_rel_gap':1e-4})
    if result.status!=0 or result.x is None:
        raise RuntimeError(f'HiGHS master status {result.status}: {result.message}')
    z=result.x;az=matrix@z
    violation=max(float(np.max(np.maximum(lower-az,0),initial=0)),float(np.max(np.maximum(az-upper,0),initial=0)),float(np.max(np.maximum(lb-z,0),initial=0)),float(np.max(np.maximum(z-ub,0),initial=0)))
    integer=np.asarray([k!=GRB.CONTINUOUS for k in kinds])
    if violation>1e-5 or np.max(np.abs(z[integer]-np.round(z[integer])),initial=0)>1e-5:
        raise RuntimeError(f'HiGHS returned infeasible master solution: violation {violation}')
    return z,dict(backend='highs',objective=float(model.ModelSense*result.fun+model.ObjCon),mip_gap=float(result.mip_gap or 0.),variables=len(z),constraints=len(rhs),violation=violation)


class LicenseFallbackMaster:
    """Original master equations; change backend only for license-size error."""
    def _master_prob_add_one_model(self,model,wp,bp,model_idx,input_nodes):
        # Exact interval bounds over the SAME candidate convex hull. Stable ReLUs
        # need no binary variable. Preserve the original big-M=1000 feasible set.
        from gurobipy import GRB,quicksum
        low,high=[x.copy() for x in self._candidate_bounds]
        previous=[input_nodes[f'v_0_{j}'] for j in range(self.layers[0])]
        for layer in range(1,len(self.layers)):
            values=[];newlow=[];newhigh=[]
            for j in range(self.layers[layer]):
                weights=np.asarray([wp[f'weight_l{layer-1}_n{k}_to_l{layer}_n{j}'] for k in range(len(previous))])
                bias=bp[f'bias_into_l{layer}_n{j}']
                lo=float(np.maximum(weights,0)@low+np.minimum(weights,0)@high+bias)-1e-7
                hi=float(np.maximum(weights,0)@high+np.minimum(weights,0)@low+bias)+1e-7
                expr=quicksum(float(w)*v for w,v in zip(weights,previous))+bias
                if layer==len(self.layers)-1:
                    model.addConstr(expr>=.0001 if self.neg_value==0 else expr<=-.0001)
                    continue
                if lo>=0:
                    value=expr
                    if hi>1000:model.addConstr(expr<=1000)
                elif hi<=0:
                    value=0.
                    if lo < -1000:model.addConstr(expr>=-1000)
                else:
                    value=model.addVar(lb=0.,ub=min(1000.,max(0.,hi)),name=f'tight_{model_idx}_{layer}_{j}')
                    inactive=model.addVar(vtype=GRB.BINARY)
                    model.addConstr(value>=expr)
                    model.addConstr(value<=expr+min(1000.,max(0.,-lo))*inactive)
                    model.addConstr(value<=min(1000.,max(0.,hi))*(1-inactive))
                values.append(value);newlow.append(max(0.,lo));newhigh.append(min(1000.,max(0.,hi)))
            previous=values;low=np.asarray(newlow);high=np.asarray(newhigh)
        model.update();return model

    def _master_prob(self,ilist,candidates):
        from gurobipy import Model,GRB,GurobiError,quicksum
        with Model() as model:
            model.Params.OutputFlag=0
            self._candidate_bounds=(np.min(candidates,axis=0),np.max(candidates,axis=0))
            inputs,model=self._master_prob_add_inputs(model)
            model=self._master_prob_add_plausibility(model,candidates,inputs)
            for i,(wp,bp) in enumerate(zip(self.w_primes,self.b_primes)):
                model=self._master_prob_add_one_model(model,wp,bp,i,inputs)
            distances=[]
            for i,value in enumerate(ilist):
                # |x-factual|>=0 already implied by the original two inequalities.
                v=model.addVar(lb=0.,vtype=GRB.CONTINUOUS,name=f'objl1_feat_{i}')
                model.addConstr(v>=value-inputs[f'v_0_{i}']);model.addConstr(v>=inputs[f'v_0_{i}']-value);distances.append(v)
            model.setObjective(quicksum(distances),GRB.MINIMIZE);model.update()
            remaining=max(.001,getattr(self,'master_deadline',time.perf_counter()+30.)-time.perf_counter())
            model.Params.TimeLimit=remaining
            try:
                model.optimize()
                if model.Status!=GRB.OPTIMAL:raise RuntimeError(f'Gurobi master status {model.Status}')
                values=np.asarray([v.X for v in model.getVars()]);info=dict(backend='gurobi',objective=model.ObjVal,variables=model.NumVars,constraints=model.NumConstrs)
            except GurobiError as exc:
                if exc.errno!=10010:raise
                values,info=solve_linear_highs(model,time_limit=max(.001,getattr(self,'master_deadline',time.perf_counter()+30.)-time.perf_counter()))
            if not hasattr(self,'master_solver_records'):self.master_solver_records=[]
            self.master_solver_records.append(info)
            return pd.DataFrame([[values[inputs[f'v_0_{j}'].index] for j in range(len(ilist))]])
