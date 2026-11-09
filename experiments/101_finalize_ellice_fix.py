"""Explicit calibration-feasibility-gated ElliCE follow-up; old outputs preserved."""
import importlib.util,json,time
from pathlib import Path
import pandas as pd,torch
s=importlib.util.spec_from_file_location('diag',str(Path(__file__).with_name('098_diagnose_ellice.py')));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(1)
parent=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1/german_logistic');out=Path('/home/xu.zhao/robust_ce_results/iclr_runs/ellice_convergence/20260925_feasible_native_v1');out.mkdir(parents=True,exist_ok=False)
screen=pd.read_csv('/home/xu.zhao/robust_ce_results/iclr_runs/ellice_convergence/20260925_v1/raw.csv');baseline=screen[screen.config=='default'];eligible=baseline.groupby('epsilon').ellipsoid_robust.all();eligible=list(eligible[eligible].index)
cal=pd.read_csv(parent/'calibration.csv');cal=cal[(cal.method=='ellice')&cal.param.isin(eligible)].sort_values(['FV','Cost','param'],ascending=[False,True,True]);epsilon=float(cal.iloc[0].param)
(out/'selection.json').write_text(json.dumps(dict(epsilon=epsilon,optimizer='official defaults',eligibility='all62 calibration cases pass native ellipsoid criterion',selection='original calibration FV, then cost; test excluded',eligible_epsilons=eligible,time=time.time()),indent=2))
p=torch.load(parent/'prepared.pt',weights_only=False);d=p['data'];names=d['feature_names'];domain=m.MixedDomain(d['train_x'],p['groups']);lo,sc=domain.low,domain.scale
model=m.TabularMLP(len(names),p['hidden']);model.load_state_dict(torch.load(parent/'current.pt',weights_only=True));model.eval();frame=pd.DataFrame(((d['train_x']-lo)/sc).numpy(),columns=names);frame['target']=d['train_y'].numpy();data=m.Data(dataframe=frame,target_column='target');wrapper=m._tabular_wrapper_class(sc,lo)(model);gen=m.ContinuousGenerator(model=wrapper,data=data,eps=epsilon,reg_coef=1e-4,device='cpu');groups=[[names[j] for j in ids] for ids in p['groups'].values()];rows=[];values=[]
for i,source in enumerate(p['sources']['test']):
 torch.manual_seed(90000+i);m.np.random.seed(90000+i)
 result=gen.generate(pd.Series(((source-lo)/sc).numpy(),index=names),one_hot_groups=groups,target_class=1,progress_bar=False,requires='robust')
 z=torch.tensor(result[names].iloc[0].to_numpy(float),dtype=torch.float32)[None];cf=z*sc+lo
 with torch.no_grad():
  h=gen._get_penult_features(z);h=torch.cat([h,torch.ones(1,1)],1);nom=(h@gen.omega_c).item();rob=nom-torch.linalg.vector_norm(gen.Q_inv_sqrt@h.T).item()
 assert torch.allclose(cf,domain.project(cf,True),atol=1e-5)
 values.append(cf[0]);rows.append(dict(index=i,nominal_logit=nom,robust_logit=rob,native_success=rob>0))
cfs=torch.stack(values);torch.save(cfs,out/'counterfactuals.pt');(out/'frozen.json').write_text(json.dumps({'time':time.time()}))
# Only after selection and CF freeze, access existing final future checkpoints.
bank=torch.load(parent/'future_test.pt',weights_only=False);preds=[]
for params,buffers in bank['states']:
 future=m.TabularMLP(len(names),p['hidden']);future.load_state_dict({**params,**buffers});future.eval()
 with torch.no_grad():preds.append(future(cfs).sigmoid()>=.5)
result=dict(dataset='german',classifier='logistic',epsilon=epsilon,n=len(cfs),native_robust_success=sum(r['native_success'] for r in rows),FV=float(torch.stack(preds).float().mean()),cost=float((cfs-p['sources']['test']).norm(dim=1).mean()),raw=str(out),protocol='Separate calibration-feasibility-gated follow-up; original Table100 selection unchanged')
pd.DataFrame(rows).to_csv(out/'per_case.csv',index=False);(out/'result.json').write_text(json.dumps(result,indent=2));repo=m.ROOT/'results/101_ellice_convergence';(repo/'corrected_followup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
