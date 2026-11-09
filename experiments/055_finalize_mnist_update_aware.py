"""Finish saved MNIST scoring without retaining unused baseline metric graphs.

Optimization and saved models are not modified. This evaluator detaches returned
surrogate scalars only after the full sensitivity calculation: no outer derivative
is requested during reporting. It preserves the original optimization source.
"""
import argparse,json,hashlib,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from run import bootstrap
bootstrap()
import torch
from lift_ce.experiments import mnist_update_aware as experiment

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 for name in ['summary.csv','future_per_model.csv','frozen_images.pt','complete.json']:
  if (a.run/name).exists():raise RuntimeError(f'Refusing to overwrite existing {name}')
 torch.set_num_threads(1);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.065)
 original=experiment.surrogate
 def metrics_only(*args,**kwargs):return tuple(t.detach() for t in original(*args,**kwargs))
 experiment.surrogate=metrics_only
 config=json.loads((a.run/'resolved_config.json').read_text())
 code=Path(experiment.__file__)
 (a.run/'evaluation_provenance.json').write_text(json.dumps({'evaluator':str(Path(__file__).resolve()),'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha256':hashlib.sha256(code.read_bytes()).hexdigest(),'change':'Detach computed baseline scalar metrics; no changes to objective values, optimization, images, models or selection','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2)+'\n')
 experiment.evaluate(argparse.Namespace(output=a.run,device='cuda'),config)
if __name__=='__main__':main()
