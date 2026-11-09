"""Diagnostic MISOCP bound for fixed German/logistic ElliCE epsilon .05."""
import importlib.util,json
from pathlib import Path
import numpy as np
import gurobipy as gp
s=importlib.util.spec_from_file_location('bound',str(Path(__file__).with_name('099_ellice_feasibility_bound.py')));b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
m=gp.Model('ellice_hard_feasibility');m.Params.TimeLimit=45;m.Params.Threads=1;m.Params.MIPGap=1e-5
cat={j for ids in b.domain.groups.values() for j in ids}
x=[m.addVar(lb=l,ub=u,vtype=gp.GRB.BINARY if j in cat else gp.GRB.CONTINUOUS) for j,(l,u) in enumerate(b.bounds)]
for ids in b.domain.groups.values():m.addConstr(gp.quicksum(x[j] for j in ids)==1)
h=[x[j]*float(b.scale[j])+float(b.low[j]) for j in range(len(x))]+[1.]
A=b.A.numpy();v=[m.addVar(lb=-gp.GRB.INFINITY) for _ in h];t=m.addVar(lb=0.)
for i in range(len(h)):m.addConstr(v[i]==gp.quicksum(float(A[i,j])*h[j] for j in range(len(h))))
m.addQConstr(gp.quicksum(z*z for z in v)<=t*t)
m.setObjective(gp.quicksum(float(b.omega[j])*h[j] for j in range(len(h)))-t,gp.GRB.MAXIMIZE)
m.optimize()
res=dict(status=int(m.Status),solutions=int(m.SolCount),bound=float(m.ObjBound),time=float(m.Runtime))
if m.SolCount:
 point=np.array([z.X for z in x]);res.update(objective=float(m.ObjVal),independent_margin=-b.fg(point)[0])
(b.out/'hard_feasibility.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
