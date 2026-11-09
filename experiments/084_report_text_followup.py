"""Independent task020 saved-output verification and text qualitative report."""
import os,sys,json,hashlib,math,difflib,textwrap,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/tmp/grace_text019_deps')
os.environ.setdefault('MPLCONFIGDIR','/tmp/text019_report_mpl')
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,pandas as pd,torch,yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_followup/20260924_v2')
RES=ROOT/'results/034_text_toxicity_followup';FIG=ROOT/'figures/text_toxicity_followup'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def update(x,y,w,b):
 w=w.detach().clone().requires_grad_();b=b.detach().clone().requires_grad_()
 for _ in range(30):
  loss=torch.nn.functional.binary_cross_entropy_with_logits(x@w+b,y)
  dw,db=torch.autograd.grad(loss,(w,b),create_graph=True);w,b=w-.5*dw,b-.5*db
 return w,b
def loss(q,w,b):return .05*torch.nn.functional.softplus((.5-(q@w+b).sigmoid())/.05).mean()
def dist(x,y):
 c=cdist(np.asarray(x),np.asarray(y),'sqeuclidean');i,j=linear_sum_assignment(c);return float(np.sqrt(c[i,j].mean())),j
def edit(a,b):
 a=a.split();b=b.split();row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  rr=[i]
  for j,y in enumerate(b,1):rr.append(min(row[j]+1,rr[-1]+1,row[j-1]+int(x!=y)))
  row=rr
 return row[-1]/max(len(a),len(b),1)
def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;FIG.mkdir(exist_ok=True,parents=True);assert not (RES/'independent_checks.csv').exists(),'Do not overwrite completed verification'
 inventory={'artifacts':[{'path':str(p),'sha256':sha(p)} for p in sorted(RUN.rglob('*')) if p.is_file()]}
 dump(RES/'verification_inventory.json',inventory)
 for name,h in json.loads((RUN/'input_manifest.json').read_text()).items():assert sha(name)==h
 data=load(RUN/'data.pt');population=load(RUN/'population.pt');cfg=json.loads((RUN/'resolved_config.json').read_text());selection=json.loads((RUN/'selection.json').read_text());freeze=json.loads((RUN/'test_freeze.json').read_text())
 for name,h in freeze['hashes'].items():assert sha(name)==h
 sets={k:set(' '.join(t.lower().split()) for r in rows for t in r.values()) for k,rows in data['parts'].items()}
 for a in sets:
  for b in sets:
   if a!=b:assert not sets[a]&sets[b],(a,b)
 parent=load(Path(cfg['parent'])/'data.pt')
 prior={ ' '.join(t.lower().split()) for part in parent['parts'].values() for row in part for t in row.values() }
 for split in cfg['splits']:assert not sets[split]&prior
 assert len(pd.read_csv(Path(cfg['parent'])/'generator_train.csv'))==300
 from transformers import AutoTokenizer,AutoModel
 mp=Path(cfg['assets'])/'all-MiniLM-L6-v2';tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);enc=AutoModel.from_pretrained(mp,local_files_only=True).cuda().eval()
 def encode(texts):
  result=[]
  with torch.no_grad():
   for start in range(0,len(texts),32):
    batch=tok(texts[start:start+32],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda');hidden=enc(**batch).last_hidden_state;mask=batch.attention_mask.unsqueeze(-1);result.append(torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),dim=1).cpu().double())
  return torch.cat(result)
 checks=[];future_checks=[];example_rows=[];equality=[];tables={};test_ims={};test_probs={}
 for split in ['cal','test']:
  pop=population[split];codes=pop['codes'];reencoded=encode(pop['texts']);torch.testing.assert_close(reencoded,codes,atol=2e-6,rtol=0)
  base=codes[pop['base_ids']];facts=codes[pop['factual_ids']];ref=pop['ref'];D,_=dist(base,ref);summary=pd.read_csv(RUN/f'{split}_summary.csv').set_index('method');individual=pd.read_csv(RUN/f'{split}_individual.csv');per=pd.read_csv(RUN/f'{split}_per_model.csv');manifest=pd.read_csv(RUN/f'{split}_future_manifest.csv');models=[]
  for rec in manifest.to_dict('records'):
   assert sha(rec['path'])==rec['sha256'];model=load(rec['path']);idx=model['indices'];pool=split+'_future';x=data['features'][pool][idx];y=data['labels'][pool][idx];w,b=update(x,y,data['w'],data['b']);torch.testing.assert_close(w,model['w'],atol=1e-10,rtol=0);torch.testing.assert_close(b,model['b'],atol=1e-10,rtol=0)
   assert y[:64].eq(0).all() and y[64:].eq(1).all();distance=math.sqrt((dist(data['x'][:64],x[:64])[0]**2+dist(data['x'][64:],x[64:])[0]**2)/2);assert abs(distance-rec['label_preserving_W2'])<1e-10
   if split=='test':assert Path(rec['path']).stat().st_mtime>freeze['time']
   models.append(model);future_checks.append({'split':split,'job':rec['job'],'weights_SGD_match':True,'displacement_match':True,'sha256_match':True})
  strings={}
  for method,row in summary.iterrows():
   if method=='Base':ids=np.asarray(pop['base_ids']);q=base;r=0;payload=None
   else:
    path=RUN/split/method/'output.pt';payload=load(path);supports=payload['support_ids'];ids=payload['deployed_ids'];q=codes[ids];r=row.r;assert sorted(supports)==sorted(ids);torch.testing.assert_close(q,payload['codes'],atol=0,rtol=0);assert payload['texts']==[pop['texts'][i] for i in ids]
    d0,assignment=dist(base,codes[supports]);d1,_=dist(codes[supports],ref);assert abs(d0-row.W2_Base_Q)<1e-10 and abs(d1-row.W2_Q_P1)<1e-10;assert abs(D-row.D)<1e-10 and abs(row.budget-row.beta*D)<1e-10;assert d0+d1<=row.budget+1e-8;assert abs(float((q-base).square().sum(1).mean().sqrt())-d0)<1e-10
    trace=pd.read_csv(path.parent/'trace.csv');assert (trace.surrogate.diff().dropna()<=1e-12).all();assert (not bool(trace.iloc[-1].changed)) if row.coordinate_stopped else len(trace)==10
    # Independently certify the claimed finite one-coordinate stopping condition.
    f=pop['f'].numpy();adj=pop['adj'].numpy();total=adj[supports].sum(0);fsum=f[supports].sum();best=float(row.surrogate)
    for k in (range(8) if row.coordinate_stopped else []):
     candidates=(fsum-f[supports[k]]+f)/8+math.sqrt(128)*r*np.linalg.norm(total[None]-adj[supports[k]][None]+adj,axis=1)/8
     for j in np.flatnonzero(candidates<best-1e-10):
      proposed=supports.copy();proposed[k]=j;assert dist(base,codes[proposed])[0]+dist(codes[proposed],ref)[0]>row.budget+1e-8
    if split=='test':assert selection['time']<path.stat().st_mtime<freeze['time']
   texts=[pop['texts'][i] for i in ids];strings[method]=texts;x=data['x'].clone().requires_grad_();w,b=update(x,data['y'],data['w'],data['b']);nom=loss(q,w,b);g=torch.autograd.grad(nom,x)[0];sens=math.sqrt(len(x))*g.norm();assert abs(float(nom)-row.nominal)<1e-10 and abs(float(sens)-row.sensitivity)<1e-10;assert abs(float(nom+r*sens)-row.surrogate)<1e-10
   cost=float((q-facts).norm(dim=1).mean());assert abs(cost-row.semantic_cost)<1e-10;assert abs(float((q*facts).sum(1).mean())-row.cosine_similarity)<1e-10;assert abs(np.mean([edit(pop['rows'][i]['factual'],t) for i,t in enumerate(texts)])-row.word_edit)<1e-10;assert len(set(texts))==row.unique_outputs
   current=(reencoded[ids]@data['w']+data['b']).sigmoid();assert abs(float((current>=.5).double().mean())-row.current_validity)<1e-12
   fvs=[];probabilities=[]
   for rec,m in zip(manifest.to_dict('records'),models):
    prob=(reencoded[ids]@m['w']+m['b']).sigmoid();old=individual[(individual.method==method)&(individual.job==rec['job'])].sort_values('index');assert len(old)==8;np.testing.assert_allclose(prob.numpy(),old.probability,atol=3e-6,rtol=0);fv=float((prob>=.5).double().mean());oldfv=per[(per.method==method)&(per.job==rec['job'])].FV.iloc[0];assert abs(fv-oldfv)<1e-12;fvs.append(fv);probabilities.append(prob)
   assert abs(np.mean(fvs)-row.FV)<1e-12
   if payload is not None:checks.append({'split':split,'method':method,'reencoding':True,'geometry':True,'assignment':True,'full_update_sensitivity':True,'cost':True,'FV':True,'coordinate_stopping_checked':bool(row.coordinate_stopped)})
   if split=='test':
    test_ims[method]=texts;test_probs[method]={'current':current.tolist(),'FV':(torch.stack(probabilities)>=.5).double().mean(0).tolist()}
  for beta in cfg['beta_grid']:
   control=f'r0_beta{beta:g}'
   for r in cfg['r_grid']:equality.append({'split':split,'beta':beta,'r':r,'identical_to_r0_sentences':sum(a==b for a,b in zip(strings[control],strings[f'r{r:g}_beta{beta:g}'])),'n':8})
  if split=='cal':
   best=summary.drop(index='Base').reset_index().sort_values(['FV','semantic_cost','r','beta'],ascending=[False,True,True,True]).iloc[0];assert best.method==selection['method']
  tables[split]=summary;print('VERIFIED',split,flush=True)
 # Human paired references are descriptive only, evaluated after all selection.
 reference_rows=[]
 for i,idx in enumerate(data['cases']['test']):
  reference=data['parts']['test'][idx]['neutral'];reference_rows.append({'index':i,'human_reference':reference});example_rows.append({'index':i,'factual':population['test']['rows'][i]['factual'],**{k:v[i] for k,v in test_ims.items()},'human_reference':reference})
 pd.DataFrame(example_rows).to_csv(RES/'qualitative_examples.csv',index=False);pd.DataFrame(checks).to_csv(RES/'independent_checks.csv',index=False);pd.DataFrame(future_checks).to_csv(RES/'future_checks.csv',index=False);pd.DataFrame(equality).to_csv(RES/'radius_sentence_equality.csv',index=False)
 selected=selection['method'];nomtag=f'r0_beta{selection["beta"]:g}';robtag=f'r0.25_beta{selection["beta"]:g}'
 def tokens(a,b):
  aa=a.split();bb=b.split();result=[]
  for op,i,j,k,l in difflib.SequenceMatcher(a=aa,b=bb,autojunk=False).get_opcodes():
   if op=='equal':result.extend((t,'#333333') for t in bb[k:l])
   else:
    if op in ['delete','replace']:result.extend(('[-'+t+'-]','#aa2222') for t in aa[i:j])
    if op in ['insert','replace']:result.extend(('[+'+t+'+]','#167a31') for t in bb[k:l])
  return result
 def drawdiff(ax,x,y,a,b,width=54):
  if a==b:ax.text(x,y,'No word changes',fontsize=10,color='#555555',va='top');return
  col=0;line=0
  for word,color in tokens(a,b):
   if col+len(word)+1>width:line+=1;col=0
   ax.text(x+col*.0081,y-line*.023,word,fontsize=8.7,fontfamily='monospace',color=color,va='top');col+=len(word)+1
 with PdfPages(FIG/'qualitative_all8.pdf') as pdf:
  for i in range(8):
   fig=plt.figure(figsize=(12,9));ax=fig.add_axes([0,0,1,1]);ax.axis('off');ax.text(.045,.965,f'Toxic → non-toxic · case {i} · selected r={selection["r"]:g}, β={selection["beta"]:g}',fontsize=16,va='top')
   labels=['Factual','Base','Nominal (r=0)','r=0.25','Selected'];texts=[example_rows[i]['factual'],test_ims['Base'][i],test_ims[nomtag][i],test_ims[robtag][i],test_ims[selected][i]];tags=[None,'Base',nomtag,robtag,selected]
   for row,(label,t,tag) in enumerate(zip(labels,texts,tags)):
    y=.89-row*.115;ax.text(.045,y,label,fontsize=11,fontweight='bold',va='top');ax.text(.205,y,textwrap.fill(t,94),fontsize=11,va='top',linespacing=1.4)
    if tag:ax.text(.205,y-.069,f'Current p(non-toxic)={test_probs[tag]["current"][i]:.3f}   Future validity={test_probs[tag]["FV"][i]:.0%}',fontsize=9,color='#555555')
   ax.text(.045,.285,'r=0 versus Base',fontweight='bold',fontsize=11);ax.text(.525,.285,'r=0.25 versus r=0',fontweight='bold',fontsize=11);drawdiff(ax,.045,.257,texts[1],texts[2]);drawdiff(ax,.525,.257,texts[2],texts[3]);ax.text(.045,.06,'[-removed-]   [+added+]\nClassifier-relative validity; unchanged text is shown explicitly. All eight test cases are included.',fontsize=9,color='#555555');pdf.savefig(fig)
   if i in [0,3]:fig.savefig(FIG/f'qualitative_case{i}.png',dpi=140)
   plt.close(fig)
 # Compact all-case browser view with readable native text and the same differences.
 chunks=['<!doctype html><meta charset="utf-8"><title>Text toxicity qualitative results</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}article{border-bottom:1px solid #aaa;padding:20px 0}th,td{text-align:left;vertical-align:top;padding:8px}th{width:140px}.diff{padding:10px;background:#f5f5f5}</style><h1>All eight text cases</h1>']
 for i in range(8):
  chunks.append(f'<article><h2>Case {i}</h2><table>')
  for label,t in [('Factual',example_rows[i]['factual']),('Base',test_ims['Base'][i]),('r=0',test_ims[nomtag][i]),('r=0.25',test_ims[robtag][i]),('Selected',test_ims[selected][i])]:chunks.append(f'<tr><th>{label}</th><td>{html.escape(t)}</td></tr>')
  chunks.append('</table>')
  for label,a,b in [('r=0 versus Base',test_ims['Base'][i],test_ims[nomtag][i]),('r=0.25 versus r=0',test_ims[nomtag][i],test_ims[robtag][i])]:chunks.append('<div class="diff"><b>'+label+': </b>'+('No word changes' if a==b else ' '.join(f'<span style="color:{color}">{html.escape(t)}</span>' for t,color in tokens(a,b)))+'</div>')
  chunks.append('</article>')
 (FIG/'qualitative_all8.html').write_text('\n'.join(chunks))
 test=tables['test'];head={'selected_r':selection['r'],'selected_beta':selection['beta'],'base_test_FV':float(test.loc['Base'].FV),'selected_test_FV':float(test.loc[selected].FV),'selected_vs_base_changed_sentences':sum(a!=b for a,b in zip(test_ims['Base'],test_ims[selected])),'r025_vs_r0_changed_sentences':sum(a!=b for a,b in zip(test_ims[nomtag],test_ims[robtag])),'base_cost':float(test.loc['Base'].semantic_cost),'selected_cost':float(test.loc[selected].semantic_cost),'base_sensitivity':float(test.loc['Base'].sensitivity),'selected_sensitivity':float(test.loc[selected].sensitivity)};dump(RES/'headline.json',head)
 methods=['Base',nomtag,f'r1_beta{selection["beta"]:g}',robtag,f'r4_beta{selection["beta"]:g}'];table=test.loc[methods,['FV','nominal','sensitivity','semantic_cost','word_edit','cosine_similarity']].copy();table.FV*=100;classifier=json.loads((Path(cfg['parent'])/'classifier_metrics.json').read_text());duration=(json.loads((RUN/'complete.json').read_text())['time']-(RUN/'repository_run.json').stat().st_mtime)/60
 import re
 lex=lambda text:re.findall(r'\w+',text.lower())
 lexical_changes=sum(lex(a)!=lex(b) for a,b in zip(test_ims['Base'],test_ims[selected]));robust_lexical=sum(lex(a)!=lex(b) for a,b in zip(test_ims[nomtag],test_ims[robtag]));head.update(selected_vs_base_lexical_changes=lexical_changes,r025_vs_r0_lexical_changes=robust_lexical);dump(RES/'headline.json',head)
 for fname in ['cal_summary.csv','test_summary.csv','cal_future_manifest.csv','test_future_manifest.csv','gradient_audit.json','lexical_opportunities.csv','resolved_config.json','split_audit.json']:(RES/fname).write_bytes((RUN/fname).read_bytes())
 opp=pd.read_csv(RUN/'lexical_opportunities.csv');opptable=opp.groupby(['split','beta']).feasible_word_changing_candidates.agg(['min','median','max']);allmetrics=pd.concat([t.reset_index() for t in tables.values()]);stopped=int(allmetrics.coordinate_stopped.fillna(False).astype(bool).sum())
 report=f'''# Text toxicity follow-up: candidate coverage and geometry repair

Raw run: `{RUN}`. Task020; controller083; independent reporter084.
Parent task019 results are unchanged. New cohort/protocol: do not interpret this
as a matched causal ablation of any single change.

## Results

Calibration selected r={selection['r']:g}, beta={selection['beta']:g},
FV={selection['FV']:.2%}. All settings were frozen before final test scoring.

{table.to_markdown(floatfmt='.6f')}

FV is percent. Selected versus Base changes {lexical_changes}/8 word sequences;
r.25 versus matched-beta r0 changes {robust_lexical}/8. Selected FV gain over Base:
{100*(test.loc[selected].FV-test.loc['Base'].FV):.2f} percentage points; selected
gain over nominal r0: {100*(test.loc[selected].FV-test.loc[nomtag].FV):.2f} points.
These are classifier-relative outcomes, not human-rated semantic detoxification.
Cost is mean factual-to-output normalized MiniLM embedding L2; word-edit cost
and cosine similarity are shown separately. Preserve failures and all8 examples.

## Why the previous experiment stagnated

Read-only parent diagnostic082 found zero feasible single-support word-changing
replacements at beta1.02 for all16 old cases. Changes of capitalization/apostrophes
could fit, but the codebook's next word-changing candidates were too far away.
This is a finite-neighborhood finding, not a proof that all simultaneous support
moves are impossible. Previous test Base probability changes averaged only.00246;
Base already scored100% FV. Increasing r alone cannot create missing feasible
sentences or increase a saturated validity metric.

## Follow-up design

Original fitted logistic head, MiniLM and fitting-only FLAN-T5-small are reused.
No intentional degradation of Base margins. Same first-four-beam nearest-valid
Base construction with unchanged failed-output fallback. Fresh ParaDetox pairs
exclude every text in every prior partition; nominal/cal/test/future pools are
also mutually disjoint. Seed2001;8 factuals per split, no Base-success filtering.
Human neutral counterparts remain excluded from generation and optimization.

8 beam plus64 sampled rewrites per case (32 each at temperature.8 and1.1).
Alternatives duplicating lowercase word sequences are excluded so capitalization
and punctuation variants do not masquerade as improvement; Base/reference/factual
anchor strings are preserved. No current-validity or semantic gate is added.

The prescribed update is30 full-batch SGD steps at lr.5 on128 sentence embeddings,
64/class. All30 steps are differentiated. Smooth hinge temperature.05, nominal
population mean plus r*sqrt(128)*norm(mean feature adjoint), exact two-anchor W2
constraint and one-to-one deployment remain. Q supports are actual sentence codes;
no arbitrary-vector decoder or straight-through token derivative is used.
The sensitivity neighborhood is continuous feature space, not discrete token edits.
An exact gradient-Gram factor accelerates CPU enumeration; direct full-adjoint
values are independently verified. Eight GPUs handle generation/update scoring;
CPU NumPy/SciPy performs the discrete coordinate search.

Grid r={{0,.25,1,4}}, beta={{1.02,1.1,1.25,1.5}}:32 solves. Up to10 coordinate sweeps,
strict improvement1e-10, exact geometry tolerance1e-8. {stopped}/32 saved runs reached
the no-improving-single-coordinate condition; no global optimum is claimed.
Future sampling families length/error/topic, severities1/2/4/8, two repeats;
24 independent cal and24 test models. Seeds211000+/221000+. All new heads use the
same complete30-step operator. These are stress scenarios, not certified inner
maximizers or guaranteed members of each radius ball. Calibration selection uses
FV, then embedding cost, r, beta. Test heads are created after outputs are frozen.

## Feasible lexical coverage

Counts of word-changing one-support replacements available from Base:

{opptable.to_markdown()}

These counts distinguish removal of the search bottleneck from demonstrated FV
improvement. Wider geometry can also admit semantic drift or unrelated sentences.
The pool is discrete/non-geodesic; continuous geometric theorems do not automatically
apply. The parent interpolation/snapping failure remains documented in results033.

## Verification and interpretation

{len(checks)} outputs independently checked: actual-string re-encoding, full30-step
nominal/sensitivity recomputation, geometry, assignment, costs, all per-example
future probabilities and FV. All48 future heads reconstruct under autograd SGD and
match saved weights/displacements. Old-partition exclusion, selection/freeze order,
input hashes and raw inventory hashes checked. Four inherited objective/geometry
tests plus one new gradient-factor cancellation test passed. Finite-difference
checks are in gradient_audit.json. Experiment time{duration:.1f} minutes.

Source corpus labels and classifier acceptance can disagree with human judgments;
fluency and meaning can deteriorate. No independent toxicity model or human study
was added. Compare all examples rather than treating changed text as evidence of
improvement. The test cohort remains only8 cases. No manuscript claims changed.

## Files

All-case PDF/HTML: `figures/text_toxicity_followup/qualitative_all8.pdf` and
`qualitative_all8.html`. Full grid, exact strings, radius equality, parent
stagnation diagnostics and independent checks are in results034. Original raw
runs/checkpoints preserved; no completed prior experiment was rerun.
'''
 for item in inventory['artifacts']:assert sha(item['path'])==item['sha256']
 (RES/'report.md').write_text(report);dump(RES/'completion_review.json',{'independent_outputs':len(checks),'future_models':len(future_checks),'raw_artifact_hashes':len(inventory['artifacts']),'unit_tests':5,'pdf_pages':8,'figures_visually_checked':False,'raw_outputs_unchanged':True,'headline':head});print(json.dumps(head,indent=2))
if __name__=='__main__':main()
