"""Audit and run frozen-Q comparisons; never train or overwrite parent artifacts."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
 def state(phase,**kw):
  file=ROOT/'state.yaml';s=yaml.safe_load(file.read_text());s['status']='in_progress';s['current_work'].update(phase=phase,run=str(out),controller_pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kw);file.write_text(yaml.safe_dump(s,sort_keys=False))
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/heloc_realization_mpl')
 def run(phase,beta=None,gpu=None):
  cmd=[PY,str(ROOT/'experiments/run.py'),'run','heloc_realization_comparison','--output',str(out)]
  if phase!='audit':cmd+=['--resume']
  cmd+=['--',phase,'--device','cpu' if gpu is None else 'cuda:0']
  if beta is not None:cmd+=['--beta',str(beta)]
  log=Path('/tmp/heloc_realization_audit_20260918.log') if phase=='audit' else out/f'beta{beta}.log'
  e=env.copy()
  if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
  with log.open('a') as f:
   f.write(json.dumps({'command':cmd,'gpu':gpu})+'\n');f.flush();result=subprocess.run(cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{phase} {beta} failed; see {log}')
  print('DONE',phase,beta,flush=True)
 try:
  state('archived_reproduction_and_manifest_audit',new_experiments_started=True);run('audit')
  subprocess.run(['genv','attach','--count','8'],check=True,env=env)
  state('matched_realization_comparison',gpu_ids=[1,2,3,5])
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(run,'compare',b,g) for b,g in zip([100,105,110,120],[1,2,3,5])]
   for f in futures:f.result()
  state('report_pending');print('COMPARISON_COMPLETE',flush=True)
 except BaseException as e:state('execution_error',error=str(e));raise
if __name__=='__main__':main()
