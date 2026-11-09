"""Resume-safe eight-GPU Galaxy controller, state.yaml and end-to-end timing."""
import argparse,concurrent.futures,json,os,queue,subprocess,sys,threading,time
from datetime import datetime,timezone
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent.parent
PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'experiments/configs/galaxy_grace_v1.json');p.add_argument('--no-state',action='store_true');a=p.parse_args();out=a.output
    c=json.loads(a.config.read_text());lock=threading.Lock()
    def state(phase,status='in_progress',**kw):
        if a.no_state:return
        with lock:
            path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s['active_task']='tasks/001_galaxy_task.md';s['status']=status
            s['current_work']={'task':s['active_task'],'phase':phase,'new_run':str(out),'controller_pid':os.getpid(),'gpu_ids':c['gpu_ids'],'configuration':'experiments/configs/galaxy_grace_v1.json','completed_jobs':len(list((out/'jobs').glob('*/complete.json'))) if out.exists() else 0,'previous_results_preserved':True,'updated_utc':datetime.now(timezone.utc).isoformat(),**kw}
            path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():print('SKIP',job,flush=True);return
        env=os.environ.copy();env.update(GALAXY_CONFIG=str(a.config.resolve()),CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH='/tmp/galaxy_grace_deps',MPLCONFIGDIR='/tmp/galaxy_mpl')
        cmd=[PY,str(ROOT/'experiments/run.py'),'run','galaxy_grace_v1','--output',str(out)]
        if not initial:cmd.append('--resume')
        cmd+=['--',phase,'--job',job,'--device','cuda:0']
        log=Path('/tmp')/f'galaxy_{out.name}_{job}.log' if initial else out/'logs'/f'{job}.log'
        start=time.time()
        with log.open('a') as f:
            f.write(json.dumps({'command':cmd,'physical_gpu':gpu})+'\n');f.flush();r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{job} exit={r.returncode}; {log}')
        timing={'job':job,'phase':phase,'gpu':gpu,'subprocess_start_unix':start,'subprocess_end_unix':time.time(),'wall_seconds':time.time()-start,'command':cmd}
        (out/'jobs'/job/'launch_timing.json').write_text(json.dumps(timing,indent=2));print('DONE',job,'GPU',gpu,flush=True)
        state(phase,last_completed_job=job)
    def parallel(jobs,phase):
        state(phase);q=queue.Queue()
        for item in jobs:q.put(item)
        def worker(gpu):
            while True:
                try:kind,job=q.get_nowait()
                except queue.Empty:return
                run(kind,job,gpu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            fs=[pool.submit(worker,g) for g in c['gpu_ids']]
            for f in fs:f.result()
    try:
        state('prepare');run('prepare','prepare',c['gpu_ids'][0],initial=not out.exists())
        if not (out/'construction_clock.json').exists():(out/'construction_clock.json').write_text(json.dumps({'start_unix':time.time(),'start_utc':datetime.now(timezone.utc).isoformat()},indent=2))
        parallel([('base',f'base_{t}') for t in c['task_targets']],'base_generation')
        jobs=[('attack',f'attack_{ri}_{rep}') for ri in range(len(c['r_grid'])) for rep in range(c['attack']['restarts_per_radius'])]
        jobs += [('calibration_model',f'calmodel_{i}') for i in range(len(c['calibration_scenarios']))]
        parallel(jobs,'attack_bank_and_calibration_models')
        parallel([('optimize',f'opt_calibration_{t}_{ri}') for t in c['task_targets'] for ri in range(len(c['r_grid']))],'calibration_refinement')
        run('select','select',c['gpu_ids'][0])
        chosen=json.loads((out/'selected_configuration.json').read_text())['tasks']
        parallel([('optimize',f'opt_test_{t}_{chosen[str(t)]["r_index"]}') for t in c['task_targets']],'test_refinement')
        run('freeze','freeze',c['gpu_ids'][0])
        clock=json.loads((out/'construction_clock.json').read_text())
        if 'end_unix' not in clock:
            clock.update(end_unix=time.time(),end_utc=datetime.now(timezone.utc).isoformat());clock['wall_seconds']=clock['end_unix']-clock['start_unix'];(out/'construction_clock.json').write_text(json.dumps(clock,indent=2))
        parallel([('future',f'future_{family}_{si}_{rep}') for family in c['future']['families'] for si in range(len(c['future']['severity'])) for rep in range(c['future']['realizations'])],'future_evaluation')
        state('report_and_independent_review')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(c['gpu_ids'][0]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH='/tmp/galaxy_grace_deps',MPLCONFIGDIR='/tmp/galaxy_mpl')
        subprocess.run([PY,str(ROOT/'experiments/021_report_galaxy.py'),'--output',str(out)],cwd=ROOT,env=env,check=True)
        state('complete','complete',report='results/011_galaxy_grace_v1/report.md',completion_review='results/011_galaxy_grace_v1/completion_review.json')
        if a.no_state:
            print('COMPLETE',out,flush=True);return
        p=ROOT/'state.yaml';s=yaml.safe_load(p.read_text());s.setdefault('completed_work',{})['galaxy_grace_v1']=s['current_work'].copy();s['next_actions']=[];p.write_text(yaml.safe_dump(s,sort_keys=False))
        print('COMPLETE',out,flush=True)
    except BaseException as error:
        state('execution_error','in_progress',error=str(error));raise
if __name__=='__main__':main()
