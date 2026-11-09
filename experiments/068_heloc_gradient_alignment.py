"""Same-Q nominal/sensitivity gradient audit of frozen task013 HELOC outputs."""
import argparse,os,sys,json,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--worker',type=int);a=p.parse_args();out=ROOT/'results/027_heloc_smooth_hinge/gradient_alignment';out.mkdir(exist_ok=True);c=json.loads((a.run/'resolved_config.json').read_text());sel=json.loads((a.run/'selection.json').read_text());beta=sel['beta'];rs=[0,c['b_ref']*.25,sel['r']];jobs=[(split,point) for split in ['cal','test'] for point in ['Base',*rs]]
 if a.worker is None:
  assert not (out/'summary.csv').exists(),'Existing audit protected'
  def execute(i):
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
   with (out/f'worker{i}.log').open('w') as f:subprocess.run([sys.executable,__file__,'--run',str(a.run),'--worker',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(execute,range(8)))
  import pandas as pd
  d=pd.DataFrame([json.loads((out/f'point{i}.json').read_text()) for i in range(8)]);d.to_csv(out/'summary.csv',index=False);print(d[['split','point','cosine','nominal_gradient_norm','sensitivity_gradient_norm','weighted_ratio_r025_bref','weighted_ratio_selected_r']].to_string(index=False));return
 import torch
 sys.path.insert(0,str(ROOT/'experiments/src'))
 from lift_ce.experiments.heloc_smooth_hinge import current,objective,load,sha,key
 torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(.04)
 split,point=jobs[a.worker];pop=load(a.run/'population.pt');m=current(c,'cuda');path=a.run/'population.pt' if point=='Base' else a.run/split/key(point,beta)/'output.pt';q0=pop[split]['base'] if point=='Base' else load(path)['Q'];q=q0.cuda().detach().requires_grad_();d={k:pop[k].cuda() for k in ['x','y']};f,n,s=objective(m,d,q,sel['r'],c);gn=torch.autograd.grad(n,q,retain_graph=True)[0];gs=torch.autograd.grad(s,q,retain_graph=True)[0];gf=torch.autograd.grad(f,q)[0];torch.testing.assert_close(gf,gn+sel['r']*gs,rtol=2e-4,atol=1e-7);nn=float(gn.double().norm());sn=float(gs.double().norm());dot=float((gn.double()*gs.double()).sum());assert torch.isfinite(gn).all() and torch.isfinite(gs).all()
 row={'split':split,'point':point if point=='Base' else f"r/b_ref={point/c['b_ref']:g}",'r_at_saved_solution':None if point=='Base' else point,'beta_at_saved_solution':None if point=='Base' else beta,'nominal':float(n.detach()),'sensitivity':float(s.detach()),'nominal_gradient_norm':nn,'sensitivity_gradient_norm':sn,'gradient_dot_product':dot,'cosine':dot/(nn*sn) if nn*sn else None,'weighted_ratio_r025_bref':.25*c['b_ref']*sn/nn if nn else None,'weighted_ratio_selected_r':sel['r']*sn/nn if nn else None,'selected_r':sel['r'],'b_ref':c['b_ref'],'gradient_decomposition_max_error':float((gf-gn-sel['r']*gs).abs().max()),'Q_source':str(path),'Q_source_sha256':sha(path),'population_sha256':sha(a.run/'population.pt'),'checkpoint_sha256':sha(Path(c['raw_parent'])/'current.pt'),'config_sha256':sha(a.run/'resolved_config.json'),'GPU':int(os.environ['CUDA_VISIBLE_DEVICES'])}
 torch.save({'nominal_gradient':gn.cpu(),'sensitivity_gradient':gs.cpu()},out/f'gradients{a.worker}.pt');(out/f'point{a.worker}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');print(row)
if __name__=='__main__':main()
