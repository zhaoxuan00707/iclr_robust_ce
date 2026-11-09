"""Eight-GPU controller for stronger and independent-channel calibration."""
import argparse,concurrent.futures,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
ENV={'PYTHONPATH':'/tmp/ffhq_editor_deps:/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/ffhq_editor_v2_mpl'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    def state(phase,**extra):
        path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s['active_task']='tasks/004_face_editor_extension.md';s['status']='in_progress';s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':list(range(8)),'completed_jobs':len(list((out/'jobs').glob('*/complete.json'))),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra};path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(ENV,CUDA_VISIBLE_DEVICES=str(gpu));cmd=[PY,str(ROOT/'experiments/run.py'),'run','ffhq_smile_editor_v2','--output',str(out)]
        if not initial:cmd+=['--resume']
        cmd+=['--',phase,'--job',job,'--device','cuda:0'];log=Path('/tmp/ffhq_editor_v2_initialize.log') if initial else out/'logs'/f'{job}.log'
        with log.open('a') as f:f.write(json.dumps({'command':cmd,'gpu':gpu})+'\n');f.flush();result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{job}: exit{result.returncode}; {log}')
        print('DONE',job,gpu,flush=True)
    def parallel(phase,jobs):
        state(phase)
        def lane(gpu):
            for job in jobs[gpu::8]:run(phase,job,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for f in [pool.submit(lane,i) for i in range(8)]:f.result()
    try:
        subprocess.run(['genv','attach','--count','8'],check=True);state('initialize');run('initialize','initialize',0,not out.exists());parallel('scalar',[f'scalar_{i}' for i in range(8)]);run('scalar_gate','scalar_gate',0)
        gate=json.loads((out/'scalar_gate.json').read_text());print('SCALAR_GATE',gate,flush=True)
        if not gate['passed']:parallel('vector',[f'vector_{count}_{i}' for count in [4,8] for i in range(8)])
        state('independent_smile_audit');run('supervised','supervised',0)
        state('report');env=os.environ.copy();env.update(ENV);subprocess.run([PY,str(ROOT/'experiments/036_report_ffhq_smile_editor_v2.py'),'--output',str(out)],cwd=ROOT,env=env,check=True)
        state('qualitative_review',report='results/017_ffhq_smile_editor_v2/report.md');print('CALIBRATION_COMPLETE',out,flush=True)
    except BaseException as e:state('execution_error',error=str(e));raise

if __name__=='__main__':main()
