#!/usr/bin/env python3
"""Task004 isolated phase controller, with state and command provenance."""
import json
import os
from pathlib import Path
import subprocess
import time
import yaml
REPO=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/minimal_validity_repair_wro_v1/20260916_minimal_v2')
PYTHON='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
LOGS=RUN/'orchestration';LOGS.mkdir(exist_ok=True)
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'1,2,3,5'}
BASE=[PYTHON,str(REPO/'experiments/run.py'),'run','minimal_validity_repair_wro_v1','--output',str(RUN),'--resume','--']
completed=[]

def state(phase,**extra):
    file=REPO/'state.yaml';s=yaml.safe_load(file.read_text())
    if s['active_task']!='tasks/004_baseline.md':raise RuntimeError('active task changed')
    s['current_work'].update(phase=phase,new_run=str(RUN),controller_pid=os.getpid(),completed_pipeline_phases=completed,**extra)
    tmp=file.with_suffix('.yaml.tmp');tmp.write_text(yaml.safe_dump(s,sort_keys=False));tmp.replace(file)
    (LOGS/'progress.json').write_text(json.dumps({'phase':phase,'completed':completed,**extra},indent=2))

def phase(name,phase_name=None,arguments=None,parallel=False,device='cpu'):
    args=arguments or [];jobs=[];start=time.time()
    for worker in range(4 if parallel else 1):
        command=BASE+[phase_name or name,*args,'--workers','4','--device',f'cuda:{worker}' if parallel else device]
        if parallel:command+=['--worker',str(worker)]
        file=LOGS/f'{name}_{worker}.log';handle=file.open('a')
        with (LOGS/'commands.jsonl').open('a') as manifest:manifest.write(json.dumps({'stage':name,'worker':worker,'command':command,'cwd':str(REPO),'log':str(file),'CUDA_VISIBLE_DEVICES':ENV['CUDA_VISIBLE_DEVICES'],'OMP_NUM_THREADS':1,'MKL_NUM_THREADS':1})+'\n')
        p=subprocess.Popen(command,cwd=REPO,env=ENV,stdout=handle,stderr=subprocess.STDOUT);jobs.append((p,handle))
    state(name,worker_pids=[p.pid for p,_ in jobs])
    while any(p.poll() is None for p,_ in jobs):
        if any(p.poll() not in (None,0) for p,_ in jobs):
            for p,_ in jobs:
                if p.poll() is None:p.terminate()
            raise RuntimeError(f'{name} failed; see phase logs')
        time.sleep(15);state(name,elapsed_phase_seconds=time.time()-start,worker_pids=[p.pid for p,_ in jobs if p.poll() is None])
    for p,h in jobs:
        h.close()
        if p.returncode:raise RuntimeError(f'{name} failed; see phase logs')
    completed.append(name);state(name+'_complete',worker_pids=[])

try:
    phase('smoke',device='cuda:0')
    phase('optimize',parallel=True)
    phase('freeze_raw',device='cuda:0')
    phase('repair')
    phase('paths_train','paths',['--split','train'])
    phase('discover',parallel=True)
    phase('freeze_bank')
    phase('score_train','score',['--split','train'],parallel=True)
    phase('merge_train','merge',['--split','train'])
    phase('select')
    phase('paths_test','paths',['--split','test'])
    phase('score_test','score',['--split','test'],parallel=True)
    phase('merge_test','merge',['--split','test'])
    phase('direct',device='cuda:0')
    phase('report')
    state('evaluation_complete_pending_review',worker_pids=[])
except Exception as error:
    state('pipeline_failed',failure=str(error),worker_pids=[])
    raise
