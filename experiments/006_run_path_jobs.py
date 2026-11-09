"""Run the declared HELOC construction jobs on currently visible GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
OUTPUT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/path_level_geodesic_wro_v1/20260916_path_v1')


def main():
    import torch
    devices=torch.cuda.device_count()
    if devices < 1:raise RuntimeError('no visible GPU')
    workers=min(4,devices)
    logdir=OUTPUT/'orchestration';logdir.mkdir(exist_ok=True)
    commands=[];processes=[];handles=[]
    prefix=[sys.executable,str(REPO/'experiments/run.py'),'run','path_level_geodesic_wro_v1','--output',str(OUTPUT),'--resume','--']
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for worker in range(workers):
        log=logdir/f'worker{worker}.log'
        handle=log.open('x');handles.append(handle)
        cmd=prefix+['optimize','--split','train','--workers',str(workers),'--worker',str(worker),'--device',f'cuda:{worker}']
        processes.append(subprocess.Popen(cmd,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT))
        commands.append({'worker':worker,'pid':processes[-1].pid,'command':cmd,'log':str(log)})
    manifest={'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'workers':workers,'jobs':commands,'status':'running'}
    (logdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        while any(p.poll() is None for p in processes):time.sleep(5)
        manifest['exit_codes']=[p.returncode for p in processes]
        manifest['status']='optimization_complete' if all(p.returncode==0 for p in processes) else 'failed'
        (logdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if manifest['status']=='failed':raise RuntimeError(manifest['exit_codes'])
        for phase in ['collect','report']:
            subprocess.run(prefix+[phase],cwd=REPO,env=env,check=True)
    finally:
        for h in handles:h.close()

if __name__=='__main__':main()
