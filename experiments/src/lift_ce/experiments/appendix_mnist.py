"""Task100 fixed-population horizon mismatch and no-geometry ablation."""
import argparse,copy,json,os,shutil,time
from pathlib import Path
import pandas as pd,torch
from .stylemnist_grace_v1 import assets,load,sha,dump,shifted_images
from .mnist_update_aware import update_model
from .mnist_low_margin_base import optimize
ROOT=Path(__file__).resolve().parents[4]
PARENT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/mnist_low_margin_base/20260924_v1/p05001')

def prepare(a,c):
 for f in ['population.pt','base_recovery.json']:shutil.copyfile(PARENT/f,a.output/f)
 (a.output/'unconstrained').mkdir()
 for f in ['population.pt','base_recovery.json']:shutil.copyfile(PARENT/f,a.output/'unconstrained'/f)
 paths=[PARENT/'population.pt',PARENT/'test/r0.05_beta1.02/output.pt',PARENT/'test/r0.05_beta1.02/metrics.json',PARENT/'test_model_manifest.csv',ROOT/'iclr_robust_ce-73.pdf']
 paths += [Path(c['checkpoint']),ROOT/'tasks/100_extra_appendix.md']
 dump(a.output/'input_manifest.json',{str(p):sha(p) for p in paths});dump(a.output/'resolved_config.json',c)

def futures(a,c):
 d=load(a.output/'population.pt');m,ae,_=assets(c,a.device);origin=copy.deepcopy(m.state_dict());images={'Base':d['test']['base'],'GRACE':load(PARENT/'test/r0.05_beta1.02/output.pt')['images']};folder=a.output/'future_models';folder.mkdir(exist_ok=True);rows=[];manifest=pd.read_csv(PARENT/'test_model_manifest.csv');strengths=dict(zip(c['future']['families'],c['future']['strengths']))
 for idx,rec in enumerate(manifest.to_dict('records')):
  if idx%8!=a.worker:continue
  xx=shifted_images(d['x'],rec['family'],strengths[rec['family']]*rec['severity'],rec['seed']).to(a.device);yy=d['y'].to(a.device);u=copy.deepcopy(m);one=copy.deepcopy(c);one['update']['steps']=1
  for step in range(1,41):
   u=update_model(u,xx,yy,one)
   if step not in [3,5,10,20,40]:continue
   path=folder/f'scenario{idx:02d}_T{step}.pt';torch.save(u.state_dict(),path)
   if step==3:
    historical=load(Path(rec['path']));assert all(torch.equal(v.cpu(),historical[k]) for k,v in u.state_dict().items()),'T3 replay differs'
   with torch.no_grad():
    train_loss=float(torch.nn.functional.binary_cross_entropy_with_logits(u(xx),yy));disp=sum(float((p-origin[n]).square().sum()) for n,p in u.named_parameters())**.5
    for method,im in images.items():rows.append({'scenario':idx,'T_eval':step,'T_design':3,'method':method,'FV':float((u(im.to(a.device)).sigmoid()>=.5).float().mean()),'train_loss':train_loss,'parameter_displacement':disp,'proximity':float((im-d['test']['factual']).flatten(1).norm(dim=1).mean()),'family':rec['family'],'severity':rec['severity'],'scenario_seed':rec['seed'],'experiment_seed':0,'checkpoint':str(path),'sha256':sha(path),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
  print('TRAJECTORY',idx,flush=True)
 pd.DataFrame(rows).to_csv(a.output/f'future_worker{a.worker}.csv',index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','futures','unconstrained'],required=True);p.add_argument('--worker',type=int,default=0);a=p.parse_args();a.device='cuda';c=json.loads((ROOT/'experiments/configs/mnist_low_margin_base.json').read_text());torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.08)
 if a.phase=='prepare':prepare(a,c)
 elif a.phase=='futures':futures(a,c)
 else:a.output=a.output/'unconstrained';a.split='test';a.r=.05;a.beta=1.02;a.disable_geometry=True;optimize(a,c)
if __name__=='__main__':main()
