"""Eight-GPU matched classifier-quality experiment controller."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;os.environ['GENV_GPUS']='8';subprocess.run(['genv','attach','--count','8'],check=True)
 def state(phase,**kw):
  p=ROOT/'state.yaml';s=yaml.safe_load(p.read_text());s.update(active_task='tasks/015_mnist_weaker_classifier.md',status='in_progress');s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'gpu_ids':list(range(8)),**kw};s['next_actions']=['Complete matched weaker-classifier experiment and report'];p.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(label,phase,gpu=1,extra=(),initial=False):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/mnist_weak_mpl','CUBLAS_WORKSPACE_CONFIG':':4096:8'};cmd=[PY,'experiments/run.py','run','mnist_weaker_classifier','--output',str(out)]+([] if initial else ['--resume'])+['--','--phase',phase,*map(str,extra)];log=Path('/tmp/mnist_weak_prepare.log') if initial else out/f'{label}.log'
  with log.open('a') as f:f.write(repr(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',label,flush=True)
 def batch(jobs):
  with concurrent.futures.ThreadPoolExecutor(8) as ex:
   fs=[ex.submit(run,label,phase,gpu,extra) for label,phase,gpu,extra in jobs]
   for f in fs:f.result()
 try:
  state('prepare');run('prepare','prepare',initial=True);state('classifier_training');batch([(f'candidate{i}','train',i,['--worker',i]) for i in range(8)]);run('choose','choose');state('matched_population_and_gradient_audit');run('population','population')
  state('future_models')
  for arm in ['strong','weak']:
   batch([(f'{arm}_future{i}','futures',i,['--arm',arm,'--worker',i]) for i in range(8)]);run(f'{arm}_merge','merge',extra=['--arm',arm])
  for split in ['cal','test']:
   state(split+'_optimization');jobs=[(arm,r) for arm in ['strong','weak'] for r in [0,.05,.1,.25]];batch([(f'{arm}_{split}_r{r:g}','optimize',i,['--arm',arm,'--split',split,'--r',r,'--beta',1.02]) for i,(arm,r) in enumerate(jobs)])
   if split=='cal':
    for arm in ['strong','weak']:run(f'{arm}_selection','select',extra=['--arm',arm])
  state('test_evaluation')
  for arm in ['strong','weak']:run(f'{arm}_evaluation','evaluate',extra=['--arm',arm])
  state('report_pending');print('EXPERIMENT_COMPLETE',flush=True)
 except BaseException as exc:state('execution_error',error=str(exc));raise
if __name__=='__main__':main()
