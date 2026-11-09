"""Independently verify and report all64 task021 factuals, including failures."""
import os,sys,json,math,importlib.util,html,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/tmp/grace_text019_deps')
os.environ.setdefault('MPLCONFIGDIR','/tmp/text021_report_mpl')
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,pandas as pd,torch,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
spec=importlib.util.spec_from_file_location('independent020',ROOT/'experiments/084_report_text_followup.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
load,dump,sha,update,loss,dist,edit=audit.load,audit.dump,audit.sha,audit.update,audit.loss,audit.dist,audit.edit
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/text_expanded_evaluation/20260925_v1');RES=ROOT/'results/035_text_expanded_evaluation';FIG=ROOT/'figures/text_expanded_evaluation'
def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 RES.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);assert not (RES/'report.md').exists()
 c=json.loads((RUN/'resolved_config.json').read_text());d=load(RUN/'data.pt');parent=Path(c['parent']);old=load(parent/'data.pt');original=load(Path(c['original'])/'data.pt')
 for p,h in json.loads((RUN/'input_manifest.json').read_text()).items():assert sha(p)==h,p
 norm=lambda t:' '.join(t.lower().split());used={norm(t) for dd in [old,original] for rows in dd['parts'].values() for r in rows for t in r.values()};fresh=[norm(t) for r in d['rows'] for t in r.values()];assert not set(fresh)&used;assert len(fresh)==len(set(fresh))==128
 selected=json.loads((parent/'selection.json').read_text());assert selected['r']==0 and selected['beta']==1.5
 from transformers import AutoTokenizer,AutoModel
 path=Path(c['assets'])/'all-MiniLM-L6-v2';tok=AutoTokenizer.from_pretrained(path,local_files_only=True);enc=AutoModel.from_pretrained(path,local_files_only=True).cuda().eval()
 def encode(texts):
  result=[]
  with torch.no_grad():
   for start in range(0,len(texts),32):
    batch=tok(texts[start:start+32],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda');h=enc(**batch).last_hidden_state;mask=batch.attention_mask.unsqueeze(-1);result.append(torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=1).cpu().double())
  return torch.cat(result)
 models=[];futurechecks=[]
 for p in sorted((parent/'futures/test').glob('*.pt')):
  m=load(p);idx=m['indices'];w,b=update(old['features']['test_future'][idx],old['labels']['test_future'][idx],d['w'],d['b']);torch.testing.assert_close(w,m['w'],atol=1e-10,rtol=0);torch.testing.assert_close(b,m['b'],atol=1e-10,rtol=0);models.append(m);futurechecks.append({'path':str(p),'sha256':sha(p),'full30step_reconstruction':True})
 assert len(models)==24
 summary=[];examples=[];individual=[];checks=[]
 for block in range(8):
  folder=RUN/f'block{block}';freeze=json.loads((folder/'freeze.json').read_text())
  for p,h in freeze['hashes'].items():assert sha(p)==h
  pop=load(folder/'population.pt');codes=pop['codes'];reencoded=encode(pop['texts']);torch.testing.assert_close(reencoded,codes,atol=2e-6,rtol=0)
  facts=codes[pop['factual_ids']];base=codes[pop['base_ids']];D,_=dist(base,d['ref']);strings={};ivs={}
  # Verify first-four-beam Base policy independently, including retained failures.
  generated=json.loads((folder/'generated.json').read_text())
  for i,row in enumerate(generated):
   assert row['factual']==d['rows'][block*8+i]['toxic']
   first=[t.strip() for t in row['beams'][:4] if t.strip()] or [row['factual'].strip()];zz=encode(first);pr=(zz@d['w']+d['b']).sigmoid();valid=[j for j,v in enumerate(pr) if v>=.5];j=min(valid,key=lambda j:float((zz[j]-facts[i]).norm())) if valid else int(pr.argmax());assert pop['texts'][pop['base_ids'][i]]==first[j];assert pop['base_failures'][i]==(not bool(valid))
  for name,r in [('Base',0),('Selected',0),('r0.25',.25)]:
   metrics=None
   if name=='Base':ids=pop['base_ids'];q=base;supports=ids
   else:
    payload=load(folder/f'r{r:g}.pt');ids=payload['deployed_ids'];supports=payload['support_ids'];q=codes[ids];metrics=json.loads((folder/f'r{r:g}_metrics.json').read_text());assert sorted(ids)==sorted(supports);assert payload['texts']==[pop['texts'][i] for i in ids];torch.testing.assert_close(q,payload['codes'],atol=0,rtol=0)
    d0,_=dist(base,codes[supports]);d1,_=dist(codes[supports],d['ref']);assert d0+d1<=1.5*D+1e-8;assert abs(d0+d1-metrics['geometry'])<1e-10;assert abs(float((q-base).square().sum(1).mean().sqrt())-d0)<1e-10
    trace=pd.read_csv(folder/f'r{r:g}_trace.csv');assert (trace.surrogate.diff().dropna()<=1e-12).all()
    if metrics['stopped']:
     assert not bool(trace.iloc[-1].changed)
     f=pop['f'].numpy();adj=pop['adj'].numpy();total=adj[supports].sum(0);fsum=f[supports].sum()
     for k in range(8):
      values=(fsum-f[supports[k]]+f)/8+r*math.sqrt(128)*np.linalg.norm(total[None]-adj[supports[k]][None]+adj,axis=1)/8
      for j in np.flatnonzero(values<metrics['surrogate']-1e-10):
       proposed=supports.copy();proposed[k]=j;assert dist(base,codes[proposed])[0]+dist(codes[proposed],d['ref'])[0]>1.5*D+1e-8
   x=d['x'].clone().requires_grad_();w,b=update(x,d['y'],d['w'],d['b']);nom=loss(q,w,b);g=torch.autograd.grad(nom,x)[0];sens=math.sqrt(128)*g.norm();cached=math.sqrt(128)*pop['adj'][supports].mean(0).norm();assert abs(float(sens)-float(cached))<1e-9
   if metrics:
    assert abs(float(nom)-metrics['nominal'])<1e-10;assert abs(float(sens)-metrics['sensitivity'])<1e-9;assert abs(float(nom+r*sens)-metrics['surrogate'])<1e-9
   texts=[pop['texts'][i] for i in ids];strings[name]=texts;current=(q@d['w']+d['b']).sigmoid();probs=torch.stack([(q@m['w']+m['b']).sigmoid() for m in models]);fv=(probs>=.5).double().mean(0);ivs[name]=fv.tolist();cost=(q-facts).norm(dim=1);cos=(q*facts).sum(1)
   for i in range(8):
    individual.append({'index':block*8+i,'block':block,'method':name,'FV':float(fv[i]),'current_probability':float(current[i]),'cost':float(cost[i]),'cosine':float(cos[i]),'word_edit':edit(d['rows'][block*8+i]['toxic'],texts[i])})
   np.save(RES/f'block{block}_{name}_probabilities.npy',probs.numpy())
   summary.append({'block':block,'method':name,'FV':float(fv.mean()),'current_validity':float((current>=.5).double().mean()),'cost':float(cost.mean()),'cosine':float(cos.mean()),'word_edit':np.mean([edit(d['rows'][block*8+i]['toxic'],t) for i,t in enumerate(texts)]),'nominal':float(nom.detach()),'sensitivity':float(sens),'unique_outputs':len(set(texts)),'base_failures':sum(pop['base_failures'])})
   checks.append({'block':block,'method':name,'reencoding':True,'full_update_sensitivity':True,'geometry_assignment':True,'stopping_checked':None if metrics is None else metrics['stopped']})
  for i in range(8):examples.append({'index':block*8+i,'block':block,'factual':d['rows'][block*8+i]['toxic'],**{k:v[i] for k,v in strings.items()},'human_reference':d['rows'][block*8+i]['neutral'],**{k+'_FV':v[i] for k,v in ivs.items()}})
  print('VERIFIED',block,flush=True)
 frame=pd.DataFrame(summary);agg=frame.groupby('method')[['FV','current_validity','cost','cosine','word_edit','nominal','sensitivity']].mean().reindex(['Base','Selected','r0.25']);agg.to_csv(RES/'summary.csv');frame.to_csv(RES/'population_summary.csv',index=False);pd.DataFrame(individual).to_csv(RES/'individual.csv',index=False);pd.DataFrame(examples).to_csv(RES/'qualitative_examples.csv',index=False);pd.DataFrame(checks).to_csv(RES/'independent_checks.csv',index=False);pd.DataFrame(futurechecks).to_csv(RES/'future_checks.csv',index=False)
 changed=sum(e['Base']!=e['Selected'] for e in examples);robchanged=sum(e['Selected']!=e['r0.25'] for e in examples)
 chunks=['<!doctype html><meta charset="utf-8"><title>64-sentence evaluation</title><style>body{font:17px sans-serif;max-width:1100px;margin:30px auto}article{padding:18px;border-bottom:1px solid #aaa}th,td{text-align:left;vertical-align:top;padding:7px}th{width:150px}</style><h1>All 64 new factuals</h1><p>Selected: r=0, beta=1.5. Classifier validity is not a human judgment of meaning preservation.</p>']
 with PdfPages(FIG/'qualitative_all64.pdf') as pdf:
  for start in range(0,64,2):
   fig=plt.figure(figsize=(12,10));ax=fig.add_axes([0,0,1,1]);ax.axis('off')
   for offset,e in enumerate(examples[start:start+2]):
    top=.965-offset*.48;ax.text(.035,top,f"Case {e['index']} | Base FV {100*e['Base_FV']:.1f}% | selected FV {100*e['Selected_FV']:.1f}%",fontsize=12,va='top');y=top-.05;chunks.append(f"<article><h2>Case {e['index']}</h2><table>")
    for key in ['factual','Base','Selected','r0.25','human_reference']:
     ax.text(.035,y,key,fontsize=10,va='top',weight='bold');text='\n'.join(textwrap.wrap(e[key],95));ax.text(.19,y,text,fontsize=10,va='top');y-=max(1,len(text.splitlines()))*.019+.024;chunks.append(f'<tr><th>{key}</th><td>{html.escape(e[key])}</td></tr>')
    chunks.append('</table></article>')
   ax.text(.035,.018,'All cases included. Embedding geometry does not guarantee grammar or meaning preservation.',fontsize=9);pdf.savefig(fig);plt.close(fig)
 (FIG/'qualitative_all64.html').write_text('\n'.join(chunks))
 table=agg.copy();table['FV']*=100;table['current_validity']*=100
 report=f'''# Expanded frozen-settings ParaDetox evaluation

64 new factuals, eight populations of eight. Raw run: `{RUN}`.
Same trained models, nominal128 examples, favorable references and selected r0/beta1.5 as task020; diagnostic r.25/beta1.5 fixed in advance. No retuning or new training.

{table.to_markdown(floatfmt='.5f')}

FV/current validity are percentages. Cost is normalized MiniLM embedding L2, cosine is an automatic similarity proxy, word edit is normalized word Levenshtein. Nominal/sensitivity are averaged over eight population objectives, not one64-support objective.
Selected changes {changed}/64 strings versus Base; r.25 changes {robchanged}/64 versus selected r0. The report includes all64 cases without filtering failed Base generation or bad rewrites. Base failed for {int(frame[frame.method=='Base'].base_failures.sum())}/64 cases. No semantic gate or validity repair was added.

## Scope and checks
All texts in all task019/020 partitions excluded from new selection. Seed2101; first64 classifier-negative5–25-word factuals in shuffled remaining corpus, eight consecutive blocks. Test human rewrites withheld until reporting.8 beam+64 sampled candidates, exact same Base policy. T30 full-batch eta.5, smooth temperature.05, ten coordinate sweeps, geometry tolerance1e-8. All eight GPUs used, one block each. Five existing focused tests passed.

Independent actual-string re-encoding, Base construction, population geometry, exact assignment, full30-step autograd sensitivity, monotonic traces and claimed single-coordinate stopping checked for all16 solves. All24 existing future heads reconstructed from original shifted training examples and matched. Original input hashes unchanged; new output freeze hashes verified.

The24 future classifiers are reused from task020, already examined previously. This is an extension to unseen factual sentences under a fixed previously examined shift bank, NOT an independent new-shift evaluation. No bootstrap confidence intervals or human semantic labels are claimed. Fixed embedding/classifier proxies can reward unrelated or ungrammatical rewrites. Examine the complete qualitative PDF/HTML and exact CSV, not only favorable examples.

Artifacts: `summary.csv`, `population_summary.csv`, `individual.csv`, `qualitative_examples.csv`, `independent_checks.csv`, `future_checks.csv`; PDF/HTML in `figures/text_expanded_evaluation/`.
'''
 (RES/'report.md').write_text(report);dump(RES/'provenance.json',{'raw_run':str(RUN),'config':c,'input_manifest':json.loads((RUN/'input_manifest.json').read_text()),'new_artifact_hashes':{str(p):sha(p) for p in RUN.rglob('*') if p.is_file()},'selected_changed':changed,'r025_changed':robchanged,'checks':len(checks),'future_checks':len(futurechecks)})
 print(table.to_string(),flush=True)
if __name__=='__main__':main()
