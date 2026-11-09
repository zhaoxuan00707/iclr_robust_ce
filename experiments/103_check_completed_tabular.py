"""Run independent checks on completed workers while the other jobs continue."""
import importlib.util,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='3'
import torch
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('report',str(ROOT/'experiments/095_report_appendix_tabular.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
root=Path('/home/xu.zhao/robust_ce_results/iclr_runs/appendix_tabular/20260925_v1');rows=[]
for p in sorted(root.glob('*/complete.json')):
 print('CHECK',p.parent.name,flush=True)
 f,r,c=m.check_worker(p.parent);rows.extend(c);print('PASSED',len(c),flush=True)
 (ROOT/'results/100_extra_appendix/completed_worker_checks.json').write_text(json.dumps(rows,indent=2))
print('TOTAL',len(rows))
