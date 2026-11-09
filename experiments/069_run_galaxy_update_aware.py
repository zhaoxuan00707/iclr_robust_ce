"""Eight-GPU controller for task017; original outputs are read-only."""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--audit-only',action='store_true');a=p.parse_args();out=a.output
 os.environ['GENV_GPUS']='8';subprocess.run(['genv','attach','--count','8'],check=True)
 c=json.loads((ROOT/'experiments/configs/galaxy_update_aware.json').read_text())
 def state(phase,**kw):
  path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text())
  if s.get('active_task')!='tasks/017_galaxy_update_aware.md':s['previous_work_before_task017']=s.get('current_work')
  s.update(active_task='tasks/017_galaxy_update_aware.md',status='in_progress',current_work={'task':'tasks/017_galaxy_update_aware.md','phase':phase,'run':str(out),'gpu_ids':list(range(8)),**kw},next_actions=['Complete Galaxy directed-pair pilot and independent report']);path.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(label,phase,gpu,extra=(),initial=False):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
  cmd=[PY,'experiments/run.py','run','galaxy_update_aware','--output',str(out)]+([] if initial else ['--resume'])+['--','--phase',phase,*map(str,extra)]
  log=Path('/tmp/galaxy_update_aware_'+out.name+'_prepare.log') if initial else out/f'{label}.log'
  with log.open('a') as f:f.write(repr(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',label,flush=True)
 def batch(jobs):
  with concurrent.futures.ThreadPoolExecutor(8) as ex:
   fs=[ex.submit(run,*j) for j in jobs]
   for f in fs:f.result()
 try:
  state('prepare');run('prepare','prepare',1,initial=True)
  state('derivative_audit');batch([(f'audit_{pair}','audit',gpu,['--pair',pair]) for pair,gpu in zip(c['targets'],[1,6])])
  if a.audit_only:state('audit_complete_full_grid_pending');return
  state('future_models');batch([(f'futures_{i}','futures',i,['--worker',i]) for i in range(8)])
  # GPU4 has little free memory: used above for inference/update jobs only.
  # Heavy jobs use the other seven GPUs, in bounded batches.
  heavy=[0,1,2,3,5,6,7]
  for split in ['cal','test']:
   state(split+'_optimization');jobs=[(pair,r,b) for pair in c['targets'] for r in c['r_grid'] for b in c['beta_grid']]
   for start in range(0,len(jobs),len(heavy)):
    batch([(f'{pair}_{split}_{r:g}_{b:g}','optimize',gpu,['--pair',pair,'--split',split,'--r',r,'--beta',b]) for gpu,(pair,r,b) in zip(heavy,jobs[start:start+len(heavy)])])
   state(split+'_evaluation');batch([(f'{pair}_{split}_evaluation','evaluate',gpu,['--pair',pair,'--split',split]) for pair,gpu in zip(c['targets'],[1,6])])
  state('report_pending');print('EXPERIMENT_COMPLETE',flush=True)
 except BaseException as exc:state('execution_error',error=str(exc));raise
if __name__=='__main__':main()
