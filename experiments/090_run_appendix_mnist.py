"""Task100 image experiments; eight GPUs, single inherited seed0."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];OUT=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_mnist/20260925_v1')
def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT);args=parser.parse_args();OUT=args.output
 s=yaml.safe_load((ROOT/'state.yaml').read_text());s['previous_work_before_task100']=s.get('current_work');s.update(active_task='tasks/100_extra_appendix.md',status='in_progress',current_work={'task':'tasks/100_extra_appendix.md','phase':'mnist_experiments','run':str(OUT),'seed':0,'gpu_ids':list(range(8))},next_actions=['Complete MNIST experiments and Table1 variants']);(ROOT/'state.yaml').write_text(yaml.safe_dump(s,sort_keys=False))
 def run(phase,gpu,initial=False):
  cmd=[sys.executable,'experiments/run.py','run','appendix_mnist','--output',str(OUT)]+([] if initial else ['--resume'])+['--','--phase',phase,'--worker',str(gpu)]
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};log=Path('/tmp/appendix100_prepare.log') if initial else OUT/f'{phase}_{gpu}.log'
  with log.open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  print('DONE',phase,gpu,flush=True)
 run('prepare',0,True)
 with concurrent.futures.ThreadPoolExecutor(8) as ex:
  for future in [ex.submit(run,'futures',i) for i in range(8)]:future.result()
 run('unconstrained',0);print('RAW_COMPLETE',flush=True)
if __name__=='__main__':main()
