"""Concavity upper bound for ElliCE's logistic robust margin over relaxed domain."""
import importlib.util,json
from pathlib import Path
import numpy as np,torch,pandas as pd
from scipy.optimize import minimize
s=importlib.util.spec_from_file_location('diag',str(Path(__file__).with_name('098_diagnose_ellice.py')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1)
parent=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1/german_logistic');p=torch.load(parent/'prepared.pt',weights_only=False);d=p['data'];names=d['feature_names'];domain=m.MixedDomain(d['train_x'],p['groups']);lo,sc=domain.low,domain.scale
model=m.TabularMLP(len(names),[]);model.load_state_dict(torch.load(parent/'current.pt',weights_only=True));model.eval()
train=(d['train_x']-lo)/sc;frame=pd.DataFrame(train.numpy(),columns=names);frame['target']=d['train_y'].numpy();data=m.Data(dataframe=frame,target_column='target');wrapper=m._tabular_wrapper_class(sc,lo)(model)
gen=m.ContinuousGenerator(model=wrapper,data=data,eps=.05,reg_coef=1e-4,device='cpu')
omega=gen.omega_c.double();A=gen.Q_inv_sqrt.double();low=lo.double();scale=sc.double()
def fg(v):
 x=torch.tensor(v,dtype=torch.float64,requires_grad=True);h=torch.cat([x*scale+low,torch.ones(1,dtype=torch.float64)]);margin=h@omega-torch.linalg.vector_norm(A@h);g=torch.autograd.grad(margin,x)[0];return -float(margin.detach()),-g.numpy()
bounds=list(zip(train.amin(0).numpy().astype(float),train.amax(0).numpy().astype(float)))
mat=np.zeros((len(domain.groups),len(names)))
for row,ids in enumerate(domain.groups.values()):mat[row,ids]=1
constraint=dict(type='eq',fun=lambda z:mat@z-1,jac=lambda z:mat)
opt=minimize(fg,train.mean(0).numpy().astype(float),jac=True,bounds=bounds,constraints=constraint,method='SLSQP',options={'maxiter':2000,'ftol':1e-12})
f,g=fg(opt.x);g=-g;support=0.
for j in domain.numeric:support+=max(g[j]*bounds[j][0],g[j]*bounds[j][1])
for ids in domain.groups.values():support+=max(g[ids])
upper=-f+support-g@opt.x
hard=domain.project((torch.tensor(opt.x,dtype=torch.float32)*sc+lo)[None],True)[0];hard_margin=-fg(((hard-lo)/sc).numpy().astype(float))[0]
result=dict(hard_rounded_margin=hard_margin,dataset='german',classifier='logistic',epsilon=.05,regularization=1e-4,scipy_success=bool(opt.success),message=str(opt.message),relaxed_margin=-f,concavity_upper_bound=float(upper),simplex_error=float(abs(mat@opt.x-1).max()),interpretation='For a linear head and fixed affine representation, margin is concave. Its tangent maximized over numeric box and categorical simplices upper-bounds all hard-category outputs too. Negative upper bound means no robust output exists in the declared feature domain for this fixed ellipsoid; longer optimization cannot fix it.',numerical_not_formal_certificate=True)
out=m.ROOT/'results/101_ellice_convergence';out.mkdir(exist_ok=True);torch.save({'relaxed':torch.tensor(opt.x),'hard':hard},out/'bound_witness.pt');(out/'feasibility_bound.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
