"""Audit native ElliCE robustness of saved calibration outputs without rerunning."""
import importlib.util,json
from pathlib import Path
import pandas as pd,torch
s=importlib.util.spec_from_file_location('diag',str(Path(__file__).with_name('098_diagnose_ellice.py')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);torch.set_num_threads(1)
root=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1');rows=[]
for worker in sorted(root.iterdir()):
 if not worker.is_dir() or not list(worker.glob('cal_ellice_*.pt')):continue
 p=torch.load(worker/'prepared.pt',weights_only=False);d=p['data'];names=d['feature_names'];domain=m.MixedDomain(d['train_x'],p['groups']);lo,sc=domain.low,domain.scale
 model=m.TabularMLP(len(names),p['hidden']);model.load_state_dict(torch.load(worker/'current.pt',weights_only=True));model.eval();frame=pd.DataFrame(((d['train_x']-lo)/sc).numpy(),columns=names);frame['target']=d['train_y'].numpy();data=m.Data(dataframe=frame,target_column='target');wrapper=m._tabular_wrapper_class(sc,lo)(model)
 for file in sorted(worker.glob('cal_ellice_*.pt')):
  eps=float(file.stem.removeprefix('cal_ellice_'));gen=m.ContinuousGenerator(model=wrapper,data=data,eps=eps,reg_coef=1e-4,device='cpu');o=torch.load(file,weights_only=False);z=(o['cf']-lo)/sc
  with torch.no_grad():
   h=gen._get_penult_features(z);h=torch.cat([h,torch.ones(len(h),1)],1);nom=h@gen.omega_c;rob=nom-torch.linalg.vector_norm(gen.Q_inv_sqrt@h.T,dim=0)
  rows.append(dict(worker=worker.name,epsilon=eps,n=len(z),current_valid=int((nom>0).sum()),native_robust=int((rob>0).sum()),min_margin=float(rob.min()),native_failures=sum(o['native_failure'])))
out=m.ROOT/'results/101_ellice_convergence';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'saved_calibration_audit.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
