"""Protocol-aware seed summary; do not pool corrected baselines with historical code."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
old=pd.read_csv(ROOT/'results/exp3_table1_single_run/combined_summary.csv');old['seed']=0
new=pd.read_csv(ROOT/'results/103_tabular_replication/20261109_full_v1/per_seed_metrics.csv')
old=old[old.realization!='hard'];new=new[new.realization!='hard']
compatible=old.method.isin(['wachter','grace_wachter','dice','grace_dice']) | ((old.method=='roar')&old.dataset.isin(['diabetes','heloc']))
used=pd.concat([old[compatible],new],ignore_index=True)
out=ROOT/'results/103_tabular_replication/three_seed_summary'
used.to_csv(out/'per_seed_included.csv',index=False)
rows=[]
methods=['wachter','grace_wachter','dice','grace_dice','trex','roar','proplace','ellice']
names=['Wachter','GRACE-Wachter','DiCE','GRACE-DiCE','T-Rex','ROAR','PROPLACE','ElliCE']
for (ds,clf,method),g in used.groupby(['dataset','classifier','method']):
 failed=(g.status!='executed').any()
 rows.append(dict(dataset=ds,classifier=clf,method=method,n_seeds=g.seed.nunique(),seeds=','.join(map(str,sorted(g.seed.unique()))),FV_mean=100*g.FV.mean() if not failed else float('nan'),FV_sd=100*g.FV.std(ddof=1) if not failed else float('nan'),proximity_mean=g.Cost.mean() if not failed else float('nan'),proximity_sd=g.Cost.std(ddof=1) if not failed else float('nan'),native_failures=int(g.native_failures.sum()),infrastructure_failure=failed))
a=pd.DataFrame(rows);a.to_csv(out/'summary.csv',index=False)
lines=['Mean ± sample SD across training/split seeds. Cells: FV (%) / proximity. Superscript ² denotes two seeds (1,2); otherwise three seeds (0,1,2). Proximity is the original standardized feature-space L2. Mixed-data GRACE/ROAR/PROPLACE use relaxed outputs, not hard-category results. FV includes all returned outputs, including failed constructions; see native_failures in CSV. — denotes infrastructure failures, not zero performance.','']
for clf in ['mlp','logistic']:
 lines+=['**'+('MLP' if clf=='mlp' else 'Logistic regression')+'**','','| Method | HELOC | Diabetes | German | FICO |','|---|---|---|---|---|']
 for method,label in zip(methods,names):
  vals=[]
  for ds in ['heloc','diabetes','german','fico']:
   r=a[(a.dataset==ds)&(a.classifier==clf)&(a.method==method)].iloc[0]
   vals.append('—' if r.infrastructure_failure else f'{r.FV_mean:.2f} ± {r.FV_sd:.2f} / {r.proximity_mean:.3f} ± {r.proximity_sd:.3f}'+(' ²' if r.n_seeds==2 else ''))
  lines.append('| '+label+' | '+' | '.join(vals)+' |')
 lines.append('')
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
