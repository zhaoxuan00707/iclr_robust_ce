"""Launch task021 without retraining or retuning any model."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY=sys.executable
OUT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/text_expanded_evaluation/20260925_v1')
def main():
 c=json.loads((ROOT/'experiments/configs/text_expanded_evaluation.json').read_text())
 def state(phase):
  p=ROOT/'state.yaml';s=yaml.safe_load(p.read_text())
  if s.get('active_task')!=c['task']:s['previous_work_before_task021']=s.get('current_work')
  s.update(active_task=c['task'],status='in_progress',current_work={'task':c['task'],'phase':phase,'run':str(OUT),'gpu_ids':c['gpu_ids']},next_actions=['Finish and independently verify expanded 64-sentence evaluation']);p.write_text(yaml.safe_dump(s,sort_keys=False))
 def run(phase,gpu,block=None):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'/tmp/grace_text019_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
  cmd=[PY,'experiments/run.py','run','text_expanded_evaluation','--output',str(OUT)]+([] if phase=='prepare' else ['--resume'])+['--','--phase',phase]+([] if block is None else ['--block',str(block)])
  log=Path('/tmp/text021_prepare.log') if block is None else OUT/f'block{block}.log'
  with log.open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',phase,block,flush=True)
 try:
  state('prepare');run('prepare',0);state('generation_and_optimization')
  with concurrent.futures.ThreadPoolExecutor(8) as ex:
   jobs=[ex.submit(run,'worker',gpu,i) for i,gpu in enumerate(c['gpu_ids'])]
   for job in jobs:job.result()
  state('independent_verification_pending');print('RAW_COMPLETE',flush=True)
 except BaseException:state('execution_error');raise
if __name__=='__main__':main()
