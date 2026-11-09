"""New seeds, frozen opt-in adapters, isolated outputs; bounded GPU scheduling."""
import argparse, concurrent.futures, json, os, subprocess, sys, time, queue, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',default='1,2,3,5');p.add_argument('--audit',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 jobs=[(seed,ds,model) for seed in (1,2) for ds in ('diabetes','german','heloc','fico') for model in ('logistic','mlp')]
 gpus=a.gpus.split(',')
 (a.output/'plan.json').write_text(json.dumps({'jobs':jobs,'gpus':gpus,'started':time.time()},indent=2))
 for seed in (1,2):
  config=ROOT/f'experiments/configs/tabular_replication_seed{seed}.json'
  (a.output/config.name).write_bytes(config.read_bytes())
 pending=queue.Queue()
 for job in jobs: pending.put(job)
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for base in ('experiments/src','experiments/vendor') for p in (ROOT/base).rglob('*.py')}
 (a.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
 def lane(gpu):
  records=[]
  while True:
   try: seed,ds,model=pending.get_nowait()
   except queue.Empty: break
   name=f'seed{seed}_{ds}_{model}'
   cmd=[sys.executable,'experiments/run.py','run','appendix_tabular','--output',str(a.output/name),'--','--dataset',ds,'--model',model,'--config',str(a.output/f'tabular_replication_seed{seed}.json')]+(['--audit'] if a.audit else [])
   env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':'/tmp/task100_deps','MPLCONFIGDIR':'/tmp/task103_mpl','PYTHONUNBUFFERED':'1'}
   begin=time.time()
   with (a.output/f'{name}.log').open('w') as log: r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   record=dict(seed=seed,dataset=ds,classifier=model,gpu=gpu,command=cmd,exitcode=r.returncode,seconds=time.time()-begin)
   (a.output/f'{name}_exit.json').write_text(json.dumps(record,indent=2));records.append(record)
   print(name,'exit',r.returncode,flush=True)
  return records
 with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
  futures=[pool.submit(lane,gpu) for gpu in gpus]
  records=sum([f.result() for f in futures],[])
 (a.output/'jobs.json').write_text(json.dumps(records,indent=2))
 if not a.audit:
  report=subprocess.run([sys.executable,'experiments/109_report_tabular_replication.py','--input',str(a.output),'--output',str(ROOT/'results/103_tabular_replication'/a.output.name)],cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/task103_mpl'})
  (a.output/'report_exit.json').write_text(json.dumps({'exitcode':report.returncode}))
 if any(r['exitcode'] for r in records): raise SystemExit(1)
if __name__=='__main__':main()
