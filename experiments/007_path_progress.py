"""Summarize live optimizer traces without modifying any experiment artifacts."""
import csv
import json
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/path_level_geodesic_wro_v1/20260916_path_v1')


def main():
    rows=[]
    for path in sorted((RUN/'heloc/optimization/train').glob('*.csv')):
        with path.open() as f:
            trace=[r for r in csv.DictReader(f) if r.get('seconds')]
        if not trace:continue
        last=trace[-1]
        rows.append({'job':path.stem,'last_step':int(last['step']),
                     'complete':path.with_suffix('.pt').exists(),
                     'feasible_iterates':sum(r['feasible']=='True' for r in trace),
                     'last_risk':float(last['true_path_risk']),'last_slack':float(last['slack']),
                     'last_invalid_mass':float(last['maximum_grid_invalid_mass']),
                     'seconds':sum(float(r['seconds']) for r in trace)})
    report={'run':str(RUN),'completed_jobs':sum(r['complete'] for r in rows),
            'expected_jobs':16,'completed_steps':sum(r['last_step']+1 for r in rows),
            'expected_steps_including_final_checks':16*121,
            'feasible_iterates_observed':sum(r['feasible_iterates'] for r in rows),
            'jobs':rows,'status':'running_unless_orchestration_says_otherwise'}
    manifest=RUN/'orchestration/manifest.json'
    if manifest.exists():report['orchestration']=json.loads(manifest.read_text())['status']
    (REPO/'results/005_path_progress.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ["completed_jobs", "completed_steps", "feasible_iterates_observed", "orchestration"]} if "--brief" in sys.argv else report,indent=2))

if __name__=='__main__':main()
