"""Independent task019 saved-output verification and text qualitative report."""
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
RUN=Path('/home/xu.zhao/robust_ce_results/iclr_runs/text_toxicity_update_aware/20260924_v1')
RES=ROOT/'results/033_text_toxicity_update_aware';FIG=ROOT/'figures/text_toxicity_update_aware'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def update(x,y,w,b):
 w=w.detach().clone().requires_grad_();b=b.detach().clone().requires_grad_()
 for _ in range(3):
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
 inventory=json.loads((RES/'pause_manifest.json').read_text())
 for item in inventory['artifacts']:assert sha(item['path'])==item['sha256'],item['path']
 for name,h in inventory['source_sha256'].items():assert sha(ROOT/'experiments/snapshots/text019_v1'/name)==h
 for name,h in json.loads((RUN/'input_manifest.json').read_text()).items():assert sha(name)==h
 data=load(RUN/'data.pt');population=load(RUN/'population.pt');cfg=json.loads((RUN/'resolved_config.json').read_text());selection=json.loads((RUN/'selection.json').read_text());freeze=json.loads((RUN/'test_freeze.json').read_text())
 for name,h in freeze['hashes'].items():assert sha(name)==h
 sets={k:set(' '.join(t.lower().split()) for r in rows for t in r.values()) for k,rows in data['parts'].items()}
 for a in sets:
  for b in sets:
   if a!=b:assert not sets[a]&sets[b],(a,b)
 assert len(pd.read_csv(RUN/'generator_train.csv'))==300
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
    trace=pd.read_csv(path.parent/'trace.csv');assert not bool(trace.iloc[-1].changed);assert (trace.surrogate.diff().dropna()<=1e-12).all()
    # Independently certify the claimed finite one-coordinate stopping condition.
    f=pop['f'].numpy();adj=pop['adj'].numpy();total=adj[supports].sum(0);fsum=f[supports].sum();best=float(row.surrogate)
    for k in range(8):
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
   if payload is not None:checks.append({'split':split,'method':method,'reencoding':True,'geometry':True,'assignment':True,'full_update_sensitivity':True,'cost':True,'FV':True,'coordinate_stopping':True})
   if split=='test':
    test_ims[method]=texts;test_probs[method]={'current':current.tolist(),'FV':(torch.stack(probabilities)>=.5).double().mean(0).tolist()}
  for beta in cfg['beta_grid']:
   control=f'r0_beta{beta:g}'
   for r in cfg['r_grid']:equality.append({'split':split,'beta':beta,'r':r,'identical_to_r0_sentences':sum(a==b for a,b in zip(strings[control],strings[f'r{r:g}_beta{beta:g}'])),'n':8})
  if split=='cal':
   best=summary.drop(index='Base').reset_index().sort_values(['FV','semantic_cost','r','beta'],ascending=[False,True,True,True]).iloc[0];assert best.method==selection['method']
  tables[split]=summary;print('VERIFIED',split,flush=True)
 # Recompute the continuous-geodesic/snap diagnostic separately.
 snap=pd.read_csv(RUN/'latent_snapping_audit.csv')
 for row in snap.itertuples():
  p=population[row.split];base=p['codes'][p['base_ids']].numpy();ref=p['ref'].numpy();D,j=dist(base,ref);q=(1-row.alpha)*base+row.alpha*ref[j];idx=cdist(q,p['codes']).argmin(1);snapped=p['codes'][idx];a=dist(base,q)[0]+dist(q,ref)[0];b=dist(base,snapped)[0]+dist(snapped,ref)[0];assert abs(a-row.continuous_geometry)<1e-10 and abs(b-row.snapped_geometry)<1e-10
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
 methods=['Base',nomtag,f'r0.05_beta{selection["beta"]:g}',robtag,f'r1_beta{selection["beta"]:g}'];table=test.loc[methods,['FV','nominal','sensitivity','semantic_cost','word_edit','cosine_similarity']].copy();table.FV*=100;classifier=json.loads((RUN/'classifier_metrics.json').read_text());duration=(json.loads((RUN/'complete.json').read_text())['time']-(RUN/'repository_run.json').stat().st_mtime)/60
 report=f'''# Text toxicity: discrete sentence-codebook pilot

Raw run: `{RUN}`. Task019; controller076; independent reporter081.
Frozen source and exact configs: `experiments/snapshots/text019_v1/`.

## Result

Calibration selected r={selection['r']:g}, beta={selection['beta']:g}. Test Base and
all eight refined settings have100% FV on8 texts ×24 shared future models.
There is **no demonstrated future-validity improvement** in this pilot.
At selected beta1.02, all four radii return exactly the same eight sentences;
only1/8 differs from Base, mainly capitalization, apostrophes and punctuation.
Sensitivity decreases, but this does not establish improved semantic detoxification.

{table.to_markdown(floatfmt='.6f')}

FV is percent; cost is mean Euclidean distance in normalized MiniLM384 embedding
space; word_edit is normalized word-level Levenshtein distance from the factual.
Do not compare this cost or sensitivity numerically with image experiments.
All methods have8 distinct test outputs. Calibration Base and refined FV are87.5%;
one failed calibration Base remains in the population. No validity repair.

## Method

ParaDetox toxic/neutral pairs, exact shared-text exclusion before seeded splitting.
2000 fitting pairs,256 validation pairs,128 nominal-update pairs,256 pairs each
for calibration future models, test future models, calibration factuals and test
factuals. Seed1901. Pair partitions were independently checked disjoint. The
MiniLM encoder is frozen. A logistic head is fitted with C10/LBFGS; validation
accuracy is{classifier['validation_accuracy']:.2%}, held-out test-pair accuracy
{classifier['heldout_test_pair_accuracy']:.2%}. These accuracies are distinct from
future CF validity. Pretrained encoder/LM corpus overlap cannot be excluded.

FLAN-T5-small is fine-tuned only on fitting pairs for300 AdamW steps, batch16,
lr1e-4. It is a small language model pilot, not a multi-billion-parameter LLM.
Each factual gets8 beam and16 sampled rewrites. Base is the nearest currently
valid first-four beam rewrite, or highest score if none is valid. Pool sizes are
132 calibration/112 test sentences, including factuals and fitting references.
Human neutral test counterparts are withheld from generation/selection; they are
saved only as descriptive references in qualitative_examples.csv.

Each future update is all3 full-batch SGD steps, lr.5, on128 frozen sentence
embeddings with64 examples per class. All head weights and bias update; the
encoder and generator remain fixed. Mean smooth hinge (.05 temperature) plus
r times sqrt(128)||dF/dX|| is computed through all3 steps. The gradient is averaged
across population supports before taking its norm. Data uncertainty is in the
continuous sentence-representation space, not categorical token IDs.

Q has8 equal-mass supports, restricted to actual sentence codes. Search enumerates
single-support replacements for up to10 sweeps under exact Euclidean W2
Base→Q plus Q→reference <= beta times Base→reference. Final deployment uses an
exact one-to-one assignment. All16 runs stop with no improving feasible single
replacement; this is not a global optimum or continuous Algorithm1 convergence.
Finite-codebook restriction is an explicit numerical adaptation.

Future models use disjoint calibration/test sentence pools and within-class
length/error/topic-biased sampling;4 severities ×2 seeds ×3 families per split.
Exact model-update, displacement and random-seed records are in the raw run.
These scenarios are not certified inner maximizers or guaranteed within each
r-ball. Test model checkpoints were created after output freezing. Calibration
selection was independently checked and predates all test output saves.

## Latent-space finding

Sentence-code lookup makes deployed text exact, but arbitrary embedding vectors
lack an inverse. Both alpha=.5 interpolations obey the ambient geometry before
snapping and violate beta1.02 after nearest-sentence snapping. Test geometry rises
from1.270212 to1.728938 (budget1.295616); mean round-trip embedding error0.634123.
The finite sentence set is not geodesic. This directly illustrates why an image-
style continuous latent solver followed by text decoding cannot be assumed valid.
These diagnostics concern this candidate pool, not every possible text decoder.

## Verification and limitations

All192 pause-inventory hashes, source-snapshot hashes, input hashes and test-freeze
hashes pass. All16 outputs were independently re-encoded, checked for exact
assignment/geometry/cost, full-update nominal loss and training-feature sensitivity,
per-example future probabilities and FV. All48 future heads were reconstructed
using autograd SGD gradients and matched their saved weights/displacements.
Four focused unit tests passed before execution. Both original end-to-end
finite-difference checks had relative error below3e-9. Experiment compute took
{duration:.1f} minutes. Eight GPUs handled parallel generation/model jobs; finite
coordinate enumeration itself uses CPU NumPy/SciPy, with GPU objective checks.

Classifier acceptance is not human-rated non-toxicity, fluency or meaning
preservation. Rewrites can remove meaning: e.g. case7 removes the name/token
'dick' from a sentence, and case5 drops its trailing target. Some accusations
and hostile content remain accepted. No independent toxicity model or human
annotation was performed. This small saturated pilot is evidence of a working
discrete realization, not evidence that GRACE improves text robustness.

## Artifacts

- `figures/text_toxicity_update_aware/qualitative_all8.pdf`: eight pages, every case,
  factual/Base/r0/r.25/selected, and two word-difference panels.
- `figures/text_toxicity_update_aware/qualitative_all8.html`: readable browser view.
- `qualitative_examples.csv`: exact saved strings and separate human references.
- `independent_checks.csv`, `future_checks.csv`, `radius_sentence_equality.csv`.
- `latent_snapping_audit.csv`, `headline.json`, cal/test summary CSVs.

A future experiment should first expand and audit semantically faithful candidate
coverage and use a predeclared harder update protocol. Do not tune current settings
on the saturated test results or claim a positive-radius gain from this run.
'''
 (RES/'report.md').write_text(report);dump(RES/'completion_review.json',{'independent_outputs':len(checks),'future_models':len(future_checks),'raw_artifact_hashes':len(inventory['artifacts']),'unit_tests':4,'pdf_pages':8,'figures_visually_checked':False,'raw_outputs_unchanged':True,'headline':head});print(json.dumps(head,indent=2))
if __name__=='__main__':main()
