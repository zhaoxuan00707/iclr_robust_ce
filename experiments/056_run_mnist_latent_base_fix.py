import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output;os.environ['GENV_GPUS']='8';subprocess.run(['genv','attach','--count','8'],check=True)
 def state(phase):
  p=ROOT/'state.yaml';s=yaml.safe_load(p.read_text());s.update(active_task='tasks/012_mnist_latent_base_fix.md',status='in_progress');s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'gpu_ids':list(range(8))};p.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(label,phase,gpu=1,extra=(),initial=False):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/mnist_fix_mpl','CUBLAS_WORKSPACE_CONFIG':':4096:8'};cmd=[PY,'experiments/run.py','run','mnist_latent_base_fix','--output',str(out)]+([] if initial else ['--resume'])+['--','--phase',phase,*map(str,extra)];log=Path('/tmp/mnist_latent_base_fix_prepare.log') if initial else out/f'{label}.log'
  with log.open('a') as f:f.write(repr(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',label,flush=True)
 state('base_recovery');run('prepare','prepare',initial=True);c=json.loads((ROOT/'experiments/configs/mnist_latent_base_fix.json').read_text());jobs=[(r,b) for r in c['r_grid'] for b in c['beta_grid']]
 for split in ['cal','test']:
  state(split+'_optimization')
  for start in range(0,len(jobs),8):
   with concurrent.futures.ThreadPoolExecutor(8) as ex:
    fs=[ex.submit(run,f'{split}_r{r:g}_b{b:g}','optimize',i,['--split',split,'--r',r,'--beta',b]) for i,(r,b) in enumerate(jobs[start:start+8])]
    for f in fs:f.result()
  if split=='cal':run('selection','select')
 state('test_scoring');run('evaluation','evaluate');state('report_pending');print('COMPLETE',flush=True)
if __name__=='__main__':main()
