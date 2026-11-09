"""Eight-GPU controller for the matched frozen-content FFHQ experiment."""
import argparse, concurrent.futures, json, os, subprocess, threading, time
from pathlib import Path
from datetime import datetime, timezone
import yaml
ROOT=Path(__file__).resolve().parents[1]
PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
ENV={'PYTHONPATH':'/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/ffhq_identity_mpl'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    lock=threading.Lock();start=time.time()
    def state(phase,status='in_progress',**extra):
        with lock:
            path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s['active_task']='tasks/002_face_baseline.md';s['status']=status
            s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':list(range(8)),'completed_jobs':len(list((out/'jobs').glob('*/complete.json'))),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
            path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(ENV,CUDA_VISIBLE_DEVICES=str(gpu))
        cmd=[PY,str(ROOT/'experiments/run.py'),'run','ffhq_grace_identity_v1','--output',str(out)]
        if not initial:cmd+=['--resume']
        cmd+=['--',phase,'--job',job,'--device','cuda:0']
        log=Path('/tmp/ffhq_identity_initialize.log') if initial else out/'logs'/f'{job}.log';t=time.time()
        with log.open('a') as f:
            f.write(json.dumps({'command':cmd,'physical_gpu':gpu})+'\n');f.flush();result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{job} exited {result.returncode}: {log}')
        (out/'jobs'/job/'launch_timing.json').write_text(json.dumps({'job':job,'phase':phase,'gpu':gpu,'wall_seconds':time.time()-t,'command':cmd},indent=2))
        print('DONE',job,'GPU',gpu,flush=True)
    def parallel(jobs,phase):
        state(phase)
        # One sequential lane per GPU, so every eight-job stage uses all eight.
        def lane(gpu):
            for kind,job in jobs[gpu::8]:run(kind,job,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(lane,gpu) for gpu in range(8)]:future.result()
    try:
        subprocess.run(['genv','attach','--count','8'],check=True)
        state('initialize');run('initialize','initialize',0,initial=not out.exists())
        parallel([('gradient',f'gradient_{i}') for i in range(8)],'training_gradient_subspace')
        run('basis','basis',0)
        parallel([('reconstruction',f'reconstruction_{i}') for i in range(8)],'calibration_reconstruction')
        parallel([('optimize',f'opt_bottleneck_calibration_{ri}_{bi}') for ri in range(2) for bi in range(4)],'calibration_optimization')
        parallel([('pool',f'pool_bottleneck_calibration_{ri}') for ri in range(2)],'calibration_pool');run('select','select',0)
        spec=json.loads((out/'selected_configuration.json').read_text());ri=spec['r_index'];parent=Path(json.loads((out/'resolved_config.json').read_text())['parent_run'])
        modes=['bottleneck']
        if spec['r_star']!=json.loads((parent/'selected_configuration.json').read_text())['r_star']:modes+=['full']
        parallel([('optimize',f'opt_{mode}_test_{ri}_{bi}') for mode in modes for bi in range(4)],'test_optimization')
        parallel([('pool',f'pool_{mode}_test_{ri}') for mode in modes],'test_pool');run('freeze','freeze',0)
        (out/'construction_clock.json').write_text(json.dumps({'wall_seconds':time.time()-start,'end_utc':datetime.now(timezone.utc).isoformat(),'excludes_reused_parent_construction':True},indent=2))
        parallel([('future',f'future_{i}') for i in range(8)],'matched_future_evaluation')
        parallel([('metrics',f'metrics_{i}') for i in range(8)],'identity_and_perceptual_evaluation')
        state('independent_review');env=os.environ.copy();env.update(ENV)
        subprocess.run([PY,str(ROOT/'experiments/029_report_ffhq_identity.py'),'--output',str(out)],cwd=ROOT,env=env,check=True)
        state('complete','complete',report='results/015_ffhq_identity_v1/report.md')
        path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s.setdefault('completed_work',{})['ffhq_grace_identity_v1']=s['current_work'].copy();path.write_text(yaml.safe_dump(s,sort_keys=False))
        print('COMPLETE',out,flush=True)
    except BaseException as error:
        state('execution_error',error=str(error));raise

if __name__=='__main__':main()
