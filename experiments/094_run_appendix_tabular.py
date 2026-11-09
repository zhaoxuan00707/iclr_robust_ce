"""Eight independent dataset/classifier workers; no old outputs overwritten."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',action='store_true');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a fresh run root')
 a.output.mkdir(parents=True)
 state=yaml.safe_load((ROOT/'state.yaml').read_text());state['current_work'].update(phase='tabular_preflight' if a.audit else 'tabular_running',tabular_run=str(a.output),controller_pid=os.getpid(),active_experiment_processes=8,experiment_3='preflight' if a.audit else 'running',experiment_4='pending_comparisons');(ROOT/'state.yaml').write_text(yaml.safe_dump(state,sort_keys=False))
 cfg=json.loads((ROOT/'experiments/configs/appendix_tabular.json').read_text());(a.output/'config.json').write_text(json.dumps(cfg,indent=2))
 jobs=[(ds,model) for ds in cfg['datasets'] for model in cfg['models']]
 def run(gpu,ds,model):
  cmd=[sys.executable,'experiments/run.py','run','appendix_tabular','--output',str(a.output/f'{ds}_{model}'),'--','--dataset',ds,'--model',model]+(['--audit'] if a.audit else [])
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':'/tmp/task100_deps','PYTHONUNBUFFERED':'1'}
  with (a.output/f'{ds}_{model}.log').open('w') as f: result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(ds,model,'EXIT',result.returncode,flush=True);return dict(dataset=ds,model=model,gpu=gpu,command=cmd,exitcode=result.returncode)
 with concurrent.futures.ThreadPoolExecutor(8) as pool:
  results=list(pool.map(lambda job:run(*job),[(gpu,*job) for gpu,job in enumerate(jobs)]))
 (a.output/'jobs.json').write_text(json.dumps(results,indent=2))
 state=yaml.safe_load((ROOT/'state.yaml').read_text());state['current_work'].update(phase='tabular_preflight_finished' if a.audit else 'tabular_raw_finished',active_experiment_processes=0);(ROOT/'state.yaml').write_text(yaml.safe_dump(state,sort_keys=False))
 if any(r['exitcode'] for r in results):raise SystemExit(1)
if __name__=='__main__':main()
