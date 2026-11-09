"""Schedule new paired ablation jobs; never overwrite or rerun a completed job."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['profile','debug','main'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    cfg=json.loads((ROOT/'experiments/configs/population_individual_adversary.json').read_text())
    (a.output/'configuration.json').write_text(json.dumps(cfg,indent=2))
    jobs=[]
    datasets=['diabetes'] if a.stage in ['profile','debug'] else cfg['datasets']
    for ds in datasets:
      for seed in ([3] if ds=='diabetes' else cfg['seeds']):
       for setting in cfg['settings'][ds]:
        jobs.append((ds,seed,10 if ds=='diabetes' else cfg['population_size'],setting))
    # Assign paired modes sequentially to one GPU; no competing ablation jobs there.
    gpus=cfg['gpu_ids'] if a.stage=='main' else [3]
    def worker(gpu,subset):
      records=[]
      for ds,seed,n,setting in subset:
       for mode in ['population','individual']:
        name=f"{ds}_seed{seed}_n{n}_r{setting['r_ratio']:g}_{mode}"
        cmd=[sys.executable,'experiments/run.py','run','population_individual_adversary','--output',str(a.output/name),'--','--dataset',ds,'--seed',str(seed),'--n',str(n),'--r-ratio',str(setting['r_ratio']),'--beta',str(setting['beta']),'--adversary-mode',mode,'--sensitivity-chunk-size',str(cfg['sensitivity_chunk_size'])]
        if a.stage=='profile':cmd+=['--profile-steps','3']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/task102_mpl'}
        start=time.time()
        with (a.output/(name+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        record=dict(name=name,gpu=gpu,command=cmd,start=start,end=time.time(),exitcode=r.returncode,stage=a.stage)
        (a.output/(name+'_job.json')).write_text(json.dumps(record,indent=2));records.append(record)
        print(name,'EXIT',r.returncode,flush=True)
        if r.returncode:raise RuntimeError(name)
      return records
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
      futures=[pool.submit(worker,gpu,jobs[i::len(gpus)]) for i,gpu in enumerate(gpus)]
      records=[r for f in futures for r in f.result()]
    (a.output/'jobs.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
