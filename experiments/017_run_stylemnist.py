"""Resume-safe phase barriers and one worker per allocated GPU."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
PYTHON='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((ROOT/'experiments/configs/stylemnist_grace_v1.json').read_text());gpus=cfg['gpu_ids']
    def run(phase,job,gpu,initial=False):
        if (a.output/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        argv=[PYTHON,str(ROOT/'experiments/run.py'),'run','stylemnist_grace_v1','--output',str(a.output)]
        if not initial:argv+=['--resume']
        argv+=['--',phase,'--job',job,'--device','cuda:0']
        log=Path('/tmp')/f'stylemnist_{a.output.name}_{job}.log' if initial else a.output/'logs'/f'{job}.log'
        with log.open('a') as f:
            f.write(json.dumps({'argv':argv,'physical_gpu':gpu})+'\n');f.flush()
            r=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{job} failed: {log}')
        print('DONE',job,'GPU',gpu,flush=True)
    def parallel(phase,jobs):
        # Fixed disjoint queues prevent overlapping workers on any allocated GPU.
        queues=[jobs[i::len(gpus)] for i in range(len(gpus))]
        def worker(i):
            for job in queues[i]:run(phase,job,gpus[i])
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            results=[pool.submit(worker,i) for i in range(len(gpus))]
            for f in results:f.result()
    run('prepare','prepare',gpus[0],initial=not a.output.exists())
    parallel('attack',[f'attack_initial_{i}' for i in range(cfg['attack']['initial_restarts'])])
    betas=[f'{round(b*100):03d}' for b in cfg['beta_geo']]
    parallel('optimize',[f'opt_initial_{b}' for b in betas])
    parallel('attack',[f'attack_fresh_{b}_{r}' for b in betas for r in range(cfg['attack']['fresh_restarts_per_beta'])])
    parallel('optimize',[f'opt_final_{b}' for b in betas])
    run('freeze','freeze',gpus[0])
    parallel('future',[f'future_{f}_{s}_{r}' for f in cfg['future']['families'] for s in range(len(cfg['future']['severity_multipliers'])) for r in range(cfg['future']['realizations'])])
    run('report','report',gpus[0])
    print('COMPLETE',a.output,flush=True)

if __name__=='__main__':main()
