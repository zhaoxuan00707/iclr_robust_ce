"""Run only new beta searches; use eight GPUs for independent case screening."""
import argparse,concurrent.futures,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
ENV={'PYTHONPATH':'/tmp/ffhq_editor_deps:/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/genki_beta_mpl'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    def state(phase,**extra):
        path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s.update(active_task='tasks/009_genki_preservation_search.md',status='in_progress')
        s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':list(range(8)),
                           'updated_utc':datetime.now(timezone.utc).isoformat(),**extra};path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(ENV,CUDA_VISIBLE_DEVICES=str(gpu));cmd=[PY,str(ROOT/'experiments/run.py'),'run','genki_preservation_search','--output',str(out)]
        if not initial:cmd+=['--resume']
        cmd+=['--',phase,'--job',job,'--device','cuda:0'];log=Path('/tmp/genki_preservation_initialize.log') if initial else out/'logs'/f'{job}.log'
        with log.open('a') as f:
            f.write(json.dumps({'command':cmd,'gpu':gpu})+'\n');f.flush();r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{job} failed: {log}')
        print('DONE',job,gpu,flush=True)
    def parallel(phase,jobs):
        state(phase)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for f in [pool.submit(run,phase,job,i) for i,job in enumerate(jobs)]:f.result()
    try:
        subprocess.run(['genv','attach','--count','8'],check=True);state('initialize');run('initialize','initialize',0,not out.exists())
        parallel('search',[f'search_{i}' for i in range(8)]);state('combine');run('combine','combine',0)
        parallel('quality',[f'quality_{i}' for i in range(8)]);state('freeze');run('freeze','freeze',0)
        state('report_pending');print('PRESERVATION_SEARCH_COMPLETE',flush=True)
    except BaseException as e:state('execution_error',error=str(e));raise


if __name__=='__main__':main()
