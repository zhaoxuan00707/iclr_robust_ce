"""Measure nominal-vs-sensitivity support gradients at identical saved Q."""
import argparse,json,os,sys,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--worker',type=int);a=p.parse_args();arms=['original','p052','p0505','p05001'];jobs=[(arm,split,point) for arm in arms for split in ['cal','test'] for point in (['Base'] if split=='cal' else ['Base','r0','r025'])];out=ROOT/'results/030_mnist_low_margin_base/gradient_alignment';out.mkdir(exist_ok=True)
 if a.worker is None:
  assert not (out/'summary.csv').exists(),'Existing diagnostics are protected'
  def execute(i):
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(i%8),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
   with (out/f'worker{i}.log').open('w') as log:subprocess.run([sys.executable,__file__,'--run',str(a.run),'--worker',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  for start in range(0,len(jobs),8):
   with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(execute,range(start,min(start+8,len(jobs)))))
  import pandas as pd
  d=pd.DataFrame([json.loads((out/f'point{i}.json').read_text()) for i in range(len(jobs))]);d.to_csv(out/'summary.csv',index=False);print(d[['condition','split','point','cosine','nominal_gradient_norm','sensitivity_gradient_norm','weighted_gradient_ratio_r025']].to_string(index=False));return
 import torch
 sys.path.insert(0,str(ROOT/'experiments/src'))
 from lift_ce.experiments.stylemnist_grace_v1 import assets,load,sha
 from lift_ce.counterfactuals.update_aware import surrogate
 torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065)
 arm,split,point=jobs[a.worker];raw=a.run/arm;c=json.loads((raw/'resolved_config.json').read_text());pop=load(raw/'population.pt');m,ae,_=assets(c,'cuda');path=raw/'population.pt' if point=='Base' else raw/split/('r0_beta1.02' if point=='r0' else 'r0.25_beta1.02')/'output.pt';q0=pop[split]['base_z'] if point=='Base' else load(path)['Q'];q=q0.to('cuda').detach().requires_grad_();f,n,s=surrogate(m,pop['x'].to('cuda'),pop['y'].to('cuda'),q,.25,steps=c['update']['steps'],lr=c['update']['lr'],decode=ae.decode,hinge_temperature=c['smooth_hinge']['temperature']);gn=torch.autograd.grad(n,q,retain_graph=True)[0];gs=torch.autograd.grad(s,q,retain_graph=True)[0];gf=torch.autograd.grad(f,q)[0];torch.testing.assert_close(gf,gn+.25*gs,rtol=2e-4,atol=1e-7);nn=float(gn.double().norm());sn=float(gs.double().norm());dot=float((gn.double()*gs.double()).sum());assert torch.isfinite(gn).all() and torch.isfinite(gs).all();cos=dot/(nn*sn) if nn>0 and sn>0 else None
 row={'condition':arm,'split':split,'point':point,'GPU':int(os.environ['CUDA_VISIBLE_DEVICES']),'nominal':float(n.detach()),'sensitivity':float(s.detach()),'nominal_gradient_norm':nn,'sensitivity_gradient_norm':sn,'gradient_dot_product':dot,'cosine':cos,'weighted_gradient_ratio_r025':.25*sn/nn if nn>0 else None,'full_gradient_decomposition_max_error':float((gf-gn-.25*gs).abs().max()),'Q_source':str(path),'Q_source_sha256':sha(path),'config_sha256':sha(raw/'resolved_config.json'),'checkpoint_sha256':sha(Path(c['checkpoint'])),'population_gradient':'flatten all64x64 latent support coordinates; unweighted sensitivity gradient; no geometry terms'}
 torch.save({'nominal_gradient':gn.cpu(),'sensitivity_gradient':gs.cpu()},out/f'gradients{a.worker}.pt');(out/f'point{a.worker}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');print(row)
if __name__=='__main__':main()
