"""Eight-GPU GENKI/FFHQ GRACE controller, with frozen phases and per-job outputs."""
import argparse,concurrent.futures,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
ENV={'PYTHONPATH':'/tmp/ffhq_editor_deps:/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/genki_grace_mpl'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    def state(phase,**extra):
        path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s.update(active_task='tasks/007_genki_grace_identity.md',status='in_progress')
        s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':list(range(8)),
                           'completed_jobs':len(list((out/'jobs').glob('*/complete.json'))),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
        path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(ENV,CUDA_VISIBLE_DEVICES=str(gpu));cmd=[PY,str(ROOT/'experiments/run.py'),'run','genki_grace_identity_v1','--output',str(out)]
        if not initial:cmd+=['--resume']
        cmd+=['--',phase,'--job',job,'--device','cuda:0'];log=Path('/tmp/genki_grace_initialize.log') if initial else out/'logs'/f'{job}.log'
        with log.open('a') as f:
            f.write(json.dumps({'command':cmd,'gpu':gpu})+'\n');f.flush();r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{job} exit {r.returncode}: {log}')
        print('DONE',job,'GPU',gpu,flush=True)
    def parallel(phase,jobs):
        state(phase)
        def lane(gpu):
            for job in jobs[gpu::8]:run(phase,job,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for f in [pool.submit(lane,g) for g in range(8)]:f.result()
    try:
        subprocess.run(['genv','attach','--count','8'],check=True)
        state('initialize');run('initialize','initialize',0,not out.exists())
        parallel('base',[f'base_{split}_{i}' for split in ['calibration','test'] for i in range(8)])
        parallel('pack',['pack_calibration','pack_test']);run('clean','clean',0)
        parallel('attack',[f'attack_{ri}_{j}' for ri in range(2) for j in range(4)])
        parallel('calmodel',[f'calmodel_{i}' for i in range(4)])
        parallel('optimize',[f'opt_calibration_{ri}_{bi}' for ri in range(2) for bi in range(4)])
        parallel('pool',[f'pool_calibration_{ri}' for ri in range(2)]);state('select');run('select','select',0)
        ri=json.loads((out/'selected_configuration.json').read_text())['r_index']
        parallel('optimize',[f'opt_test_{ri}_{bi}' for bi in range(4)]);state('pool');run('pool',f'pool_test_{ri}',0)
        state('freeze');run('freeze','freeze',0)
        parallel('future',[f'future_{family}_{si}_{rep}' for family in ['brightness','noise','blur'] for si in range(5) for rep in range(2)])
        state('report_pending');print('GRACE_EVALUATION_COMPLETE',out,flush=True)
    except BaseException as e:state('execution_error',error=str(e));raise


if __name__=='__main__':main()
