"""Task018 original-CLIP FFHQ controller; eight GPUs, isolated fresh runs."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--audit-only',action='store_true');a=p.parse_args();out=a.output;os.environ['GENV_GPUS']='8';subprocess.run(['genv','attach','--count','8'],check=True);c=json.loads((ROOT/'experiments/configs/ffhq_update_aware.json').read_text())
 def state(phase,**kw):
  path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text())
  if s.get('active_task')!='tasks/018_ffhq_update_aware.md':s['previous_work_before_task018']=s.get('current_work')
  s.update(active_task='tasks/018_ffhq_update_aware.md',status='in_progress',current_work={'task':'tasks/018_ffhq_update_aware.md','phase':phase,'run':str(out),'gpu_ids':list(range(8)),**kw},next_actions=['Finish original-CLIP FFHQ update-aware experiment and heatmaps']);path.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(label,phase,gpu,extra=(),initial=False):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'/tmp/ffhq_editor_deps:/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','MPLCONFIGDIR':'/tmp/ffhq_update_aware_mpl'}
  cmd=[PY,'experiments/run.py','run','ffhq_update_aware','--output',str(out)]+([] if initial else ['--resume'])+['--','--phase',phase,*map(str,extra)]
  log=Path('/tmp/ffhq_update_aware_'+out.name+'_prepare.log') if initial else out/f'{label}.log'
  with log.open('a') as f:f.write(repr(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',label,flush=True)
 def batch(jobs):
  with concurrent.futures.ThreadPoolExecutor(8) as ex:
   for f in [ex.submit(run,*job) for job in jobs]:f.result()
 try:
  state('prepare');run('prepare','prepare',1,initial=True);state('derivative_audit');run('audit','audit',1)
  if a.audit_only:state('audit_complete_full_grid_pending');return
  state('future_models');batch([(f'futures{i}','futures',i,['--worker',i]) for i in range(8)])
  heavy=list(range(8))
  for split in ['cal','test']:
   state(split+'_optimization');jobs=[(r,b) for r in c['r_grid'] for b in c['beta_grid']]
   for start in range(0,len(jobs),len(heavy)):
    batch([(f'{split}_{r:g}_{b:g}','optimize',gpu,['--split',split,'--r',r,'--beta',b]) for gpu,(r,b) in zip(heavy,jobs[start:start+len(heavy)])])
   state(split+'_evaluation');run(f'{split}_evaluation','evaluate',1,['--split',split])
  state('report_pending');print('EXPERIMENT_COMPLETE',flush=True)
 except BaseException as e:state('execution_error',error=str(e));raise
if __name__=='__main__':main()
