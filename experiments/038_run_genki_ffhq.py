"""Eight-GPU controller; every worker uses the repository launcher."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import yaml
ROOT=Path(__file__).resolve().parents[1]
PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
ENV={'PYTHONPATH':'/tmp/ffhq_editor_deps:/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps',
     'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/genki_ffhq_mpl'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    def state(phase,**extra):
        path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s.update(active_task='tasks/006_genki_ffhq_classifier.md',status='in_progress')
        s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':list(range(8)),
                           'updated_utc':datetime.now(timezone.utc).isoformat(),**extra};path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(ENV,CUDA_VISIBLE_DEVICES=str(gpu))
        cmd=[PY,str(ROOT/'experiments/run.py'),'run','genki_ffhq_classifier_v1','--output',str(out)]
        if not initial:cmd+=['--resume']
        cmd+=['--',phase,'--job',job,'--device','cpu' if phase in ['initialize','select'] else 'cuda:0']
        log=Path('/tmp/genki_ffhq_initialize.log') if initial else out/'logs'/f'{job}.log'
        with log.open('a') as f:
            f.write(json.dumps({'command':cmd,'gpu':gpu})+'\n');f.flush();r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{job} failed: {log}')
        print('DONE',job,'GPU',gpu,flush=True)
    def parallel(phase):
        state(phase)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for f in [pool.submit(run,phase,f'{phase}_{i}',i) for i in range(8)]:f.result()
    try:
        subprocess.run(['genv','attach','--count','8'],check=True)
        state('initialize');run('initialize','initialize',0,not out.exists());parallel('extract');parallel('fit')
        state('selection');run('select','select',0);state('evaluation');run('evaluate','evaluate',0)
        state('report_pending');print('TRAINING_AND_EVALUATION_COMPLETE',flush=True)
    except BaseException as e:state('execution_error',error=str(e));raise


if __name__=='__main__':main()
