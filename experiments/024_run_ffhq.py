"""Eight-GPU, resume-safe FFHQ execution and state tracking."""
import argparse,json,os,queue,subprocess,threading,time,concurrent.futures
from pathlib import Path
from datetime import datetime,timezone
import yaml
ROOT=Path(__file__).resolve().parent.parent;PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;c=json.loads((ROOT/'experiments/configs/ffhq_grace_v1.json').read_text());lock=threading.Lock()
 def state(phase,status='in_progress',**extra):
  with lock:
   path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s['active_task']='tasks/001_face_baseline.md';s['status']=status;s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':c['gpu_ids'],'completed_jobs':len(list((out/'jobs').glob('*/complete.json'))),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra};path.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(phase,job,gpu,initial=False):
  if (out/'jobs'/job/'complete.json').exists():return
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH='/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/ffhq_mpl')
  cmd=[PY,str(ROOT/'experiments/run.py'),'run','ffhq_grace_v1','--output',str(out)]
  if not initial:cmd+=['--resume']
  cmd+=['--',phase,'--job',job,'--device','cuda:0'];log=Path('/tmp')/f'ffhq_{out.name}_initialize.log' if initial else out/'logs'/f'{job}.log';start=time.time()
  with log.open('a') as f:f.write(json.dumps({'command':cmd,'physical_gpu':gpu})+'\n');f.flush();result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{job}: exit={result.returncode}; log={log}')
  (out/'jobs'/job/'launch_timing.json').write_text(json.dumps({'job':job,'phase':phase,'gpu':gpu,'wall_seconds':time.time()-start,'start_unix':start,'end_unix':time.time(),'command':cmd},indent=2));state(phase,last_completed=job);print('DONE',job,'GPU',gpu,flush=True)
 def parallel(jobs,phase):
  state(phase);q=queue.Queue()
  for item in jobs:q.put(item)
  def worker(gpu):
   while True:
    try:kind,job=q.get_nowait()
    except queue.Empty:return
    run(kind,job,gpu)
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
   futures=[pool.submit(worker,gpu) for gpu in c['gpu_ids']]
   for f in futures:f.result()
 try:
  state('initialize');run('initialize','initialize',0,initial=not out.exists())
  parallel([('sample',f'sample_{split}_{i}') for split,n in [('train',8),('calibration',2),('test',2)] for i in range(n)],'generated_cohorts')
  run('prepare','prepare',0)
  parallel([('base',f'base_{split}') for split in ['calibration','test']],'base_CF_generation')
  parallel([('attack',f'attack_{ri}_{j}') for ri in range(2) for j in range(4)]+[('calmodel',f'calmodel_{i}') for i in range(4)],'bank_and_calibration_models')
  parallel([('optimize',f'opt_calibration_{ri}_{bi}') for ri in range(2) for bi in range(4)],'calibration_refinement')
  parallel([('pool',f'pool_calibration_{ri}') for ri in range(2)],'calibration_candidate_pooling');run('select','select',0)
  ri=json.loads((out/'selected_configuration.json').read_text())['r_index']
  parallel([('optimize',f'opt_test_{ri}_{bi}') for bi in range(4)],'test_refinement');run('pool',f'pool_test_{ri}',0);run('freeze','freeze',0)
  parallel([('future',f'future_{family}_{si}_{rep}') for family in c['future']['families'] for si in range(5) for rep in range(2)],'future_evaluation')
  state('independent_review');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/ffhq_mpl')
  subprocess.run([PY,str(ROOT/'experiments/025_report_ffhq.py'),'--output',str(out)],cwd=ROOT,env=env,check=True)
  state('complete','complete',report='results/014_ffhq_grace_v1/report.md')
  path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s.setdefault('completed_work',{})['ffhq_grace_v1']=s['current_work'].copy();s['next_actions']=[];path.write_text(yaml.safe_dump(s,sort_keys=False));print('COMPLETE',out,flush=True)
 except BaseException as error:state('execution_error',error=str(error));raise
if __name__=='__main__':main()
