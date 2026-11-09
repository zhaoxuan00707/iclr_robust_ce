"""Coordinate the user-requested Galaxy pairs over disjoint halves of eight GPUs."""
import json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent.parent
PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
TAGS=['merging_round','round_cigar']
RUNS={tag:Path('/home/xu.zhao/robust_ce_results/galaxy_grace_v1')/f'20260916_pairs_{tag}_v1' for tag in TAGS}
def state(status,**extra):
 p=ROOT/'state.yaml';s=yaml.safe_load(p.read_text());s['active_task']='tasks/001_galaxy_task.md';s['status']=status
 s['current_work']={'task':s['active_task'],'phase':'pair_specific_replication','controller_pid':os.getpid(),'gpu_ids':list(range(8)),'runs':{k:str(v) for k,v in RUNS.items()},'completed_jobs':{k:len(list((v/'jobs').glob('*/complete.json'))) for k,v in RUNS.items()},'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
 s['next_actions']=['Finish and verify the two Galaxy pairs.','Read AGENTS.md and tasks/001_face_baseline.md; run and save face results without routine clarification.']
 if status=='complete':s.setdefault('completed_work',{})['galaxy_pairs']=s['current_work'].copy()
 p.write_text(yaml.safe_dump(s,sort_keys=False))
def main():
 state('in_progress');processes=[]
 for tag in TAGS:
  log=open(f'/tmp/galaxy_pairs_{tag}.log','a')
  cmd=[PY,str(ROOT/'experiments/020_run_galaxy.py'),'--output',str(RUNS[tag]),'--config',str(ROOT/f'experiments/configs/galaxy_{tag}.json'),'--no-state']
  processes.append((tag,subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)))
 while any(p.poll() is None for _,p in processes):
  state('in_progress');time.sleep(15)
 failures={tag:p.returncode for tag,p in processes if p.returncode}
 if failures:state('in_progress',failures=failures);raise RuntimeError(failures)
 state('complete',reports=[f'results/012_galaxy_{t}/report.md' for t in TAGS]);print('PAIR RUNS COMPLETE',flush=True)
if __name__=='__main__':main()
