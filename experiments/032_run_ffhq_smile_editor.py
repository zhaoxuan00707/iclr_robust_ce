"""Eight-GPU calibration-only local smile-editor screen."""
import argparse,concurrent.futures,json,os,subprocess,threading,time,sys
from pathlib import Path
from datetime import datetime,timezone
import yaml
ROOT=Path(__file__).resolve().parents[1];PY='/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python'
ENV={'PYTHONPATH':'/tmp/ffhq_editor_deps:/tmp/ffhq_identity_deps:/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/ffhq_editor_mpl'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-discovery-from',type=Path);args=p.parse_args();out=args.output
    if args.reuse_discovery_from and not out.exists():
        sys.path.insert(0,str(ROOT/'experiments'));import run as launcher
        src=args.reuse_discovery_from;path,identity=launcher.validate_output(out,'ffhq_smile_editor_v1',False);path.mkdir(parents=True)
        (path/'repository_run.json').write_text(json.dumps({'identity':identity,'initial_argv':['reuse frozen discovery; reevaluate after legacy-checkpoint loader fix']},indent=2))
        reused=[]
        for item in src.iterdir():
            if item.name in ['repository_run.json','logs','jobs','qualitative_examples','figures']:continue
            (path/item.name).symlink_to(item);reused.append(str(item))
        for name in ['jobs','logs','qualitative_examples','figures']:(path/name).mkdir()
        for job in (src/'jobs').iterdir():
            if (job/'complete.json').exists():
                assert not job.name.startswith('evaluate');(path/'jobs'/job.name).symlink_to(job);reused.append(str(job))
        (path/'discovery_reuse.json').write_text(json.dumps({'source':str(src),'reason':'Legacy FaceNet checkpoint requires mmap=False and full classifier-head load before embedding mode; no previous calibration metrics were produced','reused_read_only':reused,'original_repository_marker':json.loads((src/'repository_run.json').read_text()),'changed_inference':'checkpoint deserialization only; weights strictly loaded without changes'},indent=2))
    def state(phase,status='in_progress',**extra):
        path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s['active_task']='tasks/003_face_editor_screen.md';s['status']=status;s['current_work']={'task':s['active_task'],'phase':phase,'run':str(out),'controller_pid':os.getpid(),'gpu_ids':list(range(8)),'completed_jobs':len(list((out/'jobs').glob('*/complete.json'))),'updated_utc':datetime.now(timezone.utc).isoformat(),**extra};path.write_text(yaml.safe_dump(s,sort_keys=False))
    def run(phase,job,gpu,initial=False):
        if (out/'jobs'/job/'complete.json').exists():return
        env=os.environ.copy();env.update(ENV,CUDA_VISIBLE_DEVICES=str(gpu));cmd=[PY,str(ROOT/'experiments/run.py'),'run','ffhq_smile_editor_v1','--output',str(out)]
        if not initial:cmd+=['--resume']
        cmd+=['--',phase,'--job',job,'--device','cuda:0'];log=Path('/tmp/ffhq_editor_initialize.log') if initial else out/'logs'/f'{job}.log'
        with log.open('a') as f:f.write(json.dumps({'command':cmd,'gpu':gpu})+'\n');f.flush();result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{job}: exit{result.returncode}; {log}')
        print('DONE',job,gpu,flush=True)
    def parallel(phase):
        state(phase)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for f in [pool.submit(run,phase,f'{phase}_{i}',i) for i in range(8)]:f.result()
    try:
        subprocess.run(['genv','attach','--count','8'],check=True);state('initialize');run('initialize','initialize',0,initial=not out.exists());parallel('discover');run('controls','controls',0);parallel('evaluate');state('report')
        env=os.environ.copy();env.update(ENV);subprocess.run([PY,str(ROOT/'experiments/033_report_ffhq_smile_editor.py'),'--output',str(out)],cwd=ROOT,env=env,check=True)
        gate=json.loads((out/'editor_gate.json').read_text());state('calibration_review',numerical_gate_passed=gate['numerical_gate_passed'],report='results/016_ffhq_smile_editor_v1/report.md');print('SCREEN_COMPLETE',gate,flush=True)
    except BaseException as e:state('execution_error',error=str(e));raise

if __name__=='__main__':main()
