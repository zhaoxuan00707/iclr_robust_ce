"""Supervise the active run through its declared feasibility gate and follow-up.

No parameter tuning, source rewriting, or successful-feasibility assumption is
performed. Errors and missing cells remain explicit in state.yaml.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import yaml

REPO=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/path_level_geodesic_wro_v1/20260916_path_v1')
OUT=RUN/'heloc'
ENV={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
LOGS=RUN/'orchestration/followup'


def state_update(phase,status='in_progress',**extra):
    file=REPO/'state.yaml'
    state=yaml.safe_load(file.read_text())
    state['status']=status
    state['current_work'].update(phase=phase,**extra)
    temporary=file.with_suffix('.yaml.tmp')
    temporary.write_text(yaml.safe_dump(state,sort_keys=False));temporary.replace(file)


def command(phase,split='train',worker=0,workers=1,device='cpu'):
    return [sys.executable,str(REPO/'experiments/009_path_followup.py'),phase,'--output',str(RUN),
            '--split',split,'--worker',str(worker),'--workers',str(workers),'--device',device]


def run_phase(phase,split='train',parallel=False):
    workers=4 if parallel else 1
    processes=[];handles=[]
    state_update(f'{phase}_{split}')
    for worker in range(workers):
        file=LOGS/f'{phase}_{split}_{worker}.log'
        handle=file.open('x');handles.append(handle)
        cmd=command(phase,split,worker,workers,f'cuda:{worker}' if parallel else 'cuda:0')
        process=subprocess.Popen(cmd,cwd=REPO,env=ENV,stdout=handle,stderr=subprocess.STDOUT)
        processes.append(process)
    try:
        last=0
        while any(p.poll() is None for p in processes):
            if time.time()-last>55:
                state_update(f'{phase}_{split}',worker_pids=[p.pid for p in processes],worker_exit_codes=[p.poll() for p in processes])
                last=time.time()
            time.sleep(5)
        if any(p.returncode for p in processes):
            raise RuntimeError(f'{phase}/{split} failed: {[p.returncode for p in processes]}; inspect {LOGS}')
    finally:
        for handle in handles:handle.close()


def main():
    LOGS.mkdir(parents=True,exist_ok=True)
    frozen={name:__import__('hashlib').sha256((REPO/'experiments'/name).read_bytes()).hexdigest()
            for name in ['008_summarize_path_run.py','009_path_followup.py','010_complete_path_task.py']}
    (RUN/'completion_controller_manifest.json').write_text(json.dumps({'pid':os.getpid(),'source_sha256':frozen,'python':sys.executable,'run':str(RUN)},indent=2)+'\n')
    try:
        while not (OUT/'completion.json').exists():
            manifest=RUN/'orchestration/manifest.json'
            if manifest.exists() and json.loads(manifest.read_text())['status']=='failed':
                raise RuntimeError('training worker failed; do not infer infeasibility from an execution error')
            subprocess.run([sys.executable,str(REPO/'experiments/007_path_progress.py'),'--brief'],cwd=REPO,env=ENV,check=True,stdout=subprocess.DEVNULL)
            progress=json.loads((REPO/'results/005_path_progress.json').read_text())
            state_update('full_support_training_path_optimization',completed_jobs=progress['completed_jobs'],
                         completed_iterate_checks=progress['completed_steps'],feasible_iterates_observed=progress['feasible_iterates_observed'],controller_pid=os.getpid())
            time.sleep(30)
        completion=json.loads((OUT/'completion.json').read_text())
        if completion['status']=='COMPLETED_CONSTRUCTION_REJECTION':
            subprocess.run([sys.executable,str(REPO/'experiments/008_summarize_path_run.py')],cwd=REPO,env=ENV,check=True)
            state_update('construction_completed_all_paths_rejected',status='completed_with_feasibility_rejection',
                         results='results/005_path_level_task_report.md',
                         missing_or_gated_cells=completion['missing_or_gated_cells'],
                         deployment_feasible=False,task_issue='Declared local optimizer found no feasible training iterate; downstream comparisons are explicitly gated, not completed.')
        elif completion['status']=='CONSTRUCTION_COMPLETE_DOWNSTREAM_REQUIRED':
            run_phase('initialize')
            for split in ['cal','test']:
                run_phase('optimize',split,True);run_phase('collect',split)
            run_phase('discover',parallel=True);run_phase('freeze_eval_bank')
            for split in ['train','cal','test']:
                run_phase('score',split,True)
                # Merger needs the number of shards, but only one process.
                with (LOGS/f'merge_{split}.log').open('x') as handle:
                    subprocess.run(command('merge',split,workers=4),cwd=REPO,env=ENV,check=True,stdout=handle,stderr=subprocess.STDOUT)
            run_phase('select');run_phase('future_models',parallel=True)
            run_phase('evaluate');run_phase('finish')
            completion=json.loads((OUT/'completion.json').read_text())
            state_update('evaluation_complete',status='completed',results='results/005_path_level_task_report.md',
                         completion=completion,task_issue='See reported feasibility outcomes; no unrestricted-global or untouched-confirmation claim.')
        else:raise RuntimeError(f'Unexpected completion state {completion["status"]}')
        print('Task execution finished; see state.yaml and results/005_path_level_task_report.md',flush=True)
    except Exception:
        error=traceback.format_exc()
        (RUN/'controller_error.txt').write_text(error)
        state_update('execution_error',status='failed',task_issue=error,results='results/005_path_progress.json')
        raise

if __name__=='__main__':main()
