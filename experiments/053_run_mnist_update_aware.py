"""Isolated eight-GPU controller for revised MNIST method."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--stage',choices=['smoke','full'],default='full');p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.output
 os.environ['GENV_GPUS']='8'
 subprocess.run(['genv','attach','--count','8'],check=True)
 os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
 def state(phase):
  f=ROOT/'state.yaml';s=yaml.safe_load(f.read_text());s['active_task']='tasks/011_mnist_update_aware.md';s['status']='in_progress';s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'authority':'paper/iclr_robust_ce-62.pdf Section 3','controller_pid':os.getpid(),'gpu_ids':list(range(8))};s['next_actions']=['Complete new-method MNIST pilot; save numerical results and figures'];f.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(label,phase,gpu=1,extra=(),initial=False):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mnist_update_aware_mpl',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  cmd=[PY,'experiments/run.py','run','mnist_update_aware','--output',str(out)]+([] if initial else ['--resume'])+['--','--phase',phase,*map(str,extra)]
  log=Path('/tmp/mnist_update_aware_prepare.log') if initial else out/f'{label}.log'
  with log.open('a') as f:
   f.write('COMMAND '+repr(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',label,flush=True)
 state('prepare_and_smoke')
 if not a.resume:
  run('prepare','prepare',initial=True);run('smoke','smoke')
 if a.stage=='smoke':return
 c=json.loads((ROOT/'experiments/configs/mnist_update_aware.json').read_text());state('calibration')
 run('cal_models','futures',extra=['--split','cal'])
 jobs=[(r,b) for r in c['r_grid'] for b in c['beta_grid']];gpus=list(range(8))
 for start in range(0,len(jobs),8):
  with concurrent.futures.ThreadPoolExecutor(8) as ex:
   fs=[ex.submit(run,f'cal_r{r:g}_b{b:g}','optimize',gpus[i],['--split','cal','--r',r,'--beta',b]) for i,(r,b) in enumerate(jobs[start:start+8])]
   for f in fs:f.result()
 run('selection','select');sel=json.loads((out/'selection.json').read_text());state('test_optimization')
 with concurrent.futures.ThreadPoolExecutor(8) as ex:
  fs=[ex.submit(run,f'test_b{b:g}','optimize',gpus[i],['--split','test','--r',sel['r'],'--beta',b]) for i,b in enumerate(c['beta_grid'])]
  for f in fs:f.result()
 state('future_evaluation');run('test_models','futures',extra=['--split','test']);subprocess.run([PY,'experiments/055_finalize_mnist_update_aware.py','--run',str(out)],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/mnist_update_aware_mpl','CUBLAS_WORKSPACE_CONFIG':':4096:8'},check=True);state('report_pending');print('EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
