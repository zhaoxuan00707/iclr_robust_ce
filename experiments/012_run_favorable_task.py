#!/usr/bin/env python3
"""Run the predeclared task-003 phases; persist progress and stop on failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

REPO = Path(__file__).resolve().parents[1]
RUN = Path('/home/xu.zhao/robust_ce_results/iclr_runs/favorable_region_geodesic_wro_v1/20260916_intrinsic_v1')
PYTHON = '/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
LOGS = RUN/'orchestration'
LOGS.mkdir(exist_ok=True)
ENV = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'CUDA_VISIBLE_DEVICES':'1,2,3,5'}
BASE = [PYTHON,str(REPO/'experiments/run.py'),'run','favorable_region_geodesic_wro_v1','--output',str(RUN),'--resume','--']
completed=[]


def state(phase, status='in_progress', **extra):
    file=REPO/'state.yaml';obj=yaml.safe_load(file.read_text())
    if obj['active_task'] != 'tasks/003_baseline.md':
        raise RuntimeError('active task changed; controller stopping')
    obj['status']=status
    obj['current_work'].update(phase=phase,new_run=str(RUN),controller_pid=os.getpid(),
                               completed_pipeline_phases=completed,**extra)
    temp=file.with_suffix('.yaml.tmp');temp.write_text(yaml.safe_dump(obj,sort_keys=False));temp.replace(file)
    (LOGS/'progress.json').write_text(json.dumps({'phase':phase,'status':status,'completed':completed,**extra},indent=2)+'\n')


def phase(name,arguments=None,parallel=False):
    arguments=arguments or []
    jobs=[];start=time.time()
    for worker in range(4 if parallel else 1):
        file=LOGS/f'{name}_{worker}.log'
        cmd=BASE+[name.split('_')[0],*arguments,'--workers','4']
        if parallel:cmd+=['--worker',str(worker),'--device',f'cuda:{worker}']
        elif name.split('_')[0] in ['future']:cmd+=['--device','cuda:0']
        handle=file.open('a');process=subprocess.Popen(cmd,cwd=REPO,env=ENV,stdout=handle,stderr=subprocess.STDOUT)
        jobs.append((process,handle,cmd))
    state(name,worker_pids=[p.pid for p,_,_ in jobs])
    while any(p.poll() is None for p,_,_ in jobs):
        failed=[(p.returncode,cmd) for p,_,cmd in jobs if p.poll() not in (None,0)]
        if failed:
            for p,_,_ in jobs:
                if p.poll() is None:p.terminate()
            raise RuntimeError(f'{name} failed: {failed}')
        time.sleep(15)
        state(name,elapsed_phase_seconds=time.time()-start,worker_pids=[p.pid for p,_,_ in jobs if p.poll() is None])
    for p,h,cmd in jobs:
        h.close()
        if p.returncode:raise RuntimeError(f'{name} failed: {cmd}')
    completed.append(name);state(name+'_complete')


try:
    phase('smoke')
    phase('cache',parallel=True)
    phase('optimize')
    phase('paths_train',['--split','train'])
    phase('paths_cal',['--split','cal'])
    phase('discover',parallel=True)
    # Underscore phase names need explicit dispatch instead of the split alias.
    original=BASE.copy()
    cmd=BASE+['freeze_bank']
    subprocess.run(cmd,cwd=REPO,env=ENV,stdout=(LOGS/'freeze_bank.log').open('a'),stderr=subprocess.STDOUT,check=True)
    completed.append('freeze_bank')
    for split in ['train','cal']:
        phase('score_'+split,['--split',split],parallel=True)
        phase('merge_'+split,['--split',split])
    phase('select')
    phase('paths_test',['--split','test'])
    phase('score_test',['--split','test'],parallel=True)
    phase('merge_test',['--split','test'])
    phase('future')
    phase('report')
    state('evaluation_complete_pending_review',status='in_progress',worker_pids=[])
except Exception as error:
    state('pipeline_failed',failure=str(error),worker_pids=[])
    raise
