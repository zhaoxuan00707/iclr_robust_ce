import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import torch,numpy as np,pandas as pd
from lift_ce.counterfactuals.sentence_codebook import w2,geometry
ROOT=Path(__file__).resolve().parents[1];RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_update_aware/20260924_v1');RES=ROOT/'results/034_text_toxicity_followup';RES.mkdir(exist_ok=True)
def lex(s):return re.findall(r'\w+',s.lower())
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
 torch.set_num_threads(1);data=load(RUN/'data.pt');pops=load(RUN/'population.pt');rows=[];shifts=[]
 for split,p in pops.items():
  base=p['codes'][p['base_ids']];D=w2(base,p['ref'])[0];f=p['f'];ids=np.array(p['base_ids'])
  for k,old in enumerate(ids):
   candidates=[]
   for j,t in enumerate(p['texts']):
    if lex(t)==lex(p['texts'][old]):continue
    prop=ids.copy();prop[k]=j;b=geometry(base,p['codes'][prop],p['ref'])/D;candidates.append((b,float(f[j]-f[old]),j))
   improving=[x for x in candidates if x[1]<-1e-10];rows.append({'split':split,'index':k,'minimum_beta_lexical_change':min(v[0] for v in candidates),'minimum_beta_nominal_improving_lexical_change':min((v[0] for v in improving),default=None),'feasible_lexical_changes_beta102':sum(v[0]<=1.02+1e-8 for v in candidates),'feasible_improving_lexical_changes_beta102':sum(v[0]<=1.02+1e-8 for v in improving)})
  now=(base@data['w']+data['b']).sigmoid()
  for rec in pd.read_csv(RUN/f'{split}_future_manifest.csv').to_dict('records'):
   model=load(rec['path']);p1=(base@model['w']+model['b']).sigmoid();shifts.append({'split':split,'job':rec['job'],'head_L2_change':float((model['w']-data['w']).norm()),'max_base_probability_change':float((p1-now).abs().max()),'mean_base_probability_change':float((p1-now).abs().mean())})
 pd.DataFrame(rows).to_csv(RES/'parent_geometry_diagnosis.csv',index=False);pd.DataFrame(shifts).to_csv(RES/'parent_update_diagnosis.csv',index=False);print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(shifts).groupby('split').agg({'head_L2_change':'max','max_base_probability_change':'max','mean_base_probability_change':'mean'}))
if __name__=='__main__':main()
