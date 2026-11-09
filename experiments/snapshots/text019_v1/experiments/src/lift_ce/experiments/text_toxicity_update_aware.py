"""Task019: realizable discrete sentence supports under the update-aware objective."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from lift_ce.counterfactuals.sentence_codebook import update,losses,cache,objective,solve,w2,geometry
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/text_toxicity_update_aware.json'
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tag(r,b):return f'r{r:g}_beta{b:g}'
def normtext(t):return ' '.join(str(t).lower().split())
def encoder(c):
 from transformers import AutoTokenizer,AutoModel
 p=Path(c['assets'])/'all-MiniLM-L6-v2';return AutoTokenizer.from_pretrained(p,local_files_only=True),AutoModel.from_pretrained(p,local_files_only=True).to('cuda').eval()
def encode(texts,tok,model):
 result=[]
 with torch.no_grad():
  for start in range(0,len(texts),64):
   x=tok(texts[start:start+64],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda');h=model(**x).last_hidden_state;mask=x.attention_mask.unsqueeze(-1);v=(h*mask).sum(1)/mask.sum(1);result.append(torch.nn.functional.normalize(v,dim=1).cpu().double())
 return torch.cat(result)
def prepare(out,c):
 dump(out/'resolved_config.json',c);manifest={str(p):sha(p) for p in [CONFIG,ROOT/c['task'],*Path(c['assets']).rglob('*')] if p.is_file()};dump(out/'input_manifest.json',manifest)
 frame=pd.read_csv(Path(c['assets'])/'paradetox/train.tsv',sep='\t');seen=set();pairs=[]
 for row in frame.to_dict('records'):
  toxic=str(row['en_toxic_comment']);neutral=str(row['en_neutral_comment']);a,b=normtext(toxic),normtext(neutral)
  if not a or not b or a==b or a in seen or b in seen:continue
  seen.update([a,b]);pairs.append({'toxic':toxic,'neutral':neutral})
 rng=np.random.default_rng(c['seed']);rng.shuffle(pairs);assert len(pairs)>=sum(c['splits'].values());parts={};offset=0
 for split,n in c['splits'].items():parts[split]=pairs[offset:offset+n];offset+=n
 tok,enc=encoder(c);data={'parts':parts,'features':{},'labels':{}}
 for split,rows in parts.items():
  texts=[r['toxic'] for r in rows]+[r['neutral'] for r in rows];data['features'][split]=encode(texts,tok,enc);data['labels'][split]=torch.cat([torch.zeros(len(rows)),torch.ones(len(rows))]).double();print('ENCODED',split,len(texts),flush=True)
 clf=LogisticRegression(C=10,max_iter=1000,random_state=c['seed']).fit(data['features']['fit'],data['labels']['fit']);data['w']=torch.tensor(clf.coef_[0],dtype=torch.double);data['b']=torch.tensor(float(clf.intercept_[0]),dtype=torch.double)
 metrics={'retained_disjoint_pairs':len(pairs),'partitions':c['splits'],'fitting_accuracy':clf.score(data['features']['fit'],data['labels']['fit']),'validation_accuracy':clf.score(data['features']['validation'],data['labels']['validation']),'heldout_test_pair_accuracy':clf.score(data['features']['test'],data['labels']['test'])}
 data['cases']={}
 for split in ['cal','test']:
  prob=(data['features'][split][:len(parts[split])]@data['w']+data['b']).sigmoid();ids=[i for i,p in enumerate(prob) if p<.5 and 5<=len(parts[split][i]['toxic'].split())<=25][:c['n']];assert len(ids)==c['n'];data['cases'][split]=ids
 # Fixed fitting-only favorable references and topic anchor.
 nf=len(parts['fit']);p=(data['features']['fit'][nf:]@data['w']+data['b']).sigmoid();data['ref_ids']=torch.nonzero(p>=.5).flatten()[:c['n']]+nf;assert len(data['ref_ids'])==c['n'];data['ref']=data['features']['fit'][data['ref_ids']];data['topic']=data['features']['fit'][0]
 data['x']=torch.cat([data['features']['nominal'][:64],data['features']['nominal'][len(parts['nominal']):len(parts['nominal'])+64]]);data['y']=torch.cat([torch.zeros(64),torch.ones(64)]).double()
 torch.save(data,out/'data.pt');dump(out/'classifier_metrics.json',metrics);print(metrics,flush=True)
def train_generator(out,c):
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 p=Path(c['assets'])/'flan-t5-small';tok=AutoTokenizer.from_pretrained(p,local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(p,local_files_only=True).to('cuda').train();model.config.use_cache=False
 rows=load(out/'data.pt')['parts']['fit'];gen=c['generator'];x=tok(['detoxify: '+r['toxic'] for r in rows],padding='max_length',truncation=True,max_length=gen['max_length'],return_tensors='pt');y=tok([r['neutral'] for r in rows],padding='max_length',truncation=True,max_length=gen['max_length'],return_tensors='pt').input_ids;y[y==tok.pad_token_id]=-100
 optim=torch.optim.AdamW(model.parameters(),lr=gen['lr']);rng=np.random.default_rng(c['seed']);trace=[]
 for step in range(gen['steps']):
  ids=rng.choice(len(rows),gen['batch'],replace=False);batch={k:v[ids].to('cuda') for k,v in x.items()};optim.zero_grad();loss=model(**batch,labels=y[ids].to('cuda')).loss;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();trace.append({'step':step+1,'loss':float(loss)})
  if (step+1)%25==0:print('GENERATOR',trace[-1],flush=True)
 model.config.use_cache=True;model.save_pretrained(out/'generator',safe_serialization=True);tok.save_pretrained(out/'generator');pd.DataFrame(trace).to_csv(out/'generator_train.csv',index=False)
def generate(out,c,worker):
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 tok=AutoTokenizer.from_pretrained(out/'generator',local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(out/'generator',local_files_only=True).to('cuda').eval();data=load(out/'data.pt');records=[]
 for split in ['cal','test']:
  i=worker;idx=data['cases'][split][i];text=data['parts'][split][idx]['toxic'];x=tok('detoxify: '+text,return_tensors='pt',truncation=True,max_length=96).to('cuda');torch.manual_seed(190100+(1000 if split=='test' else 0)+i)
  with torch.no_grad():
   beam=model.generate(**x,max_new_tokens=64,num_beams=8,num_return_sequences=8,do_sample=False);samples=model.generate(**x,max_new_tokens=64,num_return_sequences=16,do_sample=True,temperature=.8,top_p=.95)
  records.append({'split':split,'index':i,'source_index':idx,'factual':text,'beams':tok.batch_decode(beam,skip_special_tokens=True),'samples':tok.batch_decode(samples,skip_special_tokens=True)})
  print('GENERATED',split,i,flush=True)
 dump(out/f'generated_{worker}.json',records)
def assemble(out,c):
 data=load(out/'data.pt');tok,enc=encoder(c);records=sum([json.loads((out/f'generated_{i}.json').read_text()) for i in range(8)],[]);population={};audits=[]
 for split in ['cal','test']:
  rows=sorted([r for r in records if r['split']==split],key=lambda r:r['index']);texts=[]
  for row in rows:texts.extend(row['beams']+row['samples']+[row['factual']])
  nf=len(data['parts']['fit']);reftexts=[data['parts']['fit'][int(i)-nf]['neutral'] for i in data['ref_ids']];texts.extend(reftexts);texts=list(dict.fromkeys(t.strip() for t in texts if t.strip()));codes=encode(texts,tok,enc);lookup={t:i for i,t in enumerate(texts)};base_ids=[];factual_ids=[];failures=[]
  for row in rows:
   fact=lookup[row['factual'].strip()];first=[lookup[t.strip()] for t in row['beams'][:4] if t.strip()]
   if not first:first=[fact]
   prob=(codes[first]@data['w']+data['b']).sigmoid();valid=[j for j,p in zip(first,prob) if p>=.5];chosen=min(valid,key=lambda j:float((codes[j]-codes[fact]).norm())) if valid else first[int(prob.argmax())];base_ids.append(chosen);factual_ids.append(fact);failures.append(not bool(valid))
  f,adj=cache(data['x'].cuda(),data['y'].cuda(),codes.cuda(),data['w'].cuda(),data['b'].cuda(),**c['update_params']);f=f.cpu();adj=adj.cpu();ids=torch.tensor(base_ids);x=data['x'].clone().requires_grad_();nw,nb=update(x,data['y'],data['w'],data['b'],**c['update_params']);nom=losses(codes[ids],nw,nb).mean();grad=torch.autograd.grad(nom,x)[0];co,cn,cs=objective(ids,f,adj,len(x),.25);torch.testing.assert_close(cn,nom,atol=1e-10,rtol=1e-7);torch.testing.assert_close(cs,math.sqrt(len(x))*grad.norm(),atol=1e-10,rtol=1e-7)
  direction=torch.randn_like(x);direction/=direction.norm();eps=1e-4
  def fun(xx):w,b=update(xx,data['y'],data['w'],data['b'],**c['update_params']);return losses(codes[ids],w,b).mean()
  fd=float((fun(x+eps*direction)-fun(x-eps*direction))/(2*eps));ad=float((grad*direction).sum());err=abs(fd-ad)/max(abs(fd),abs(ad),1e-12);assert err<1e-3
  audits.append({'split':split,'cache_nominal_error':abs(float(cn-nom)),'cache_sensitivity_error':abs(float(cs)-math.sqrt(len(x))*float(grad.norm())),'finite_difference_relative_error':err,'candidate_count':len(texts),'base_failures':sum(failures)})
  population[split]={'texts':texts,'codes':codes,'base_ids':base_ids,'factual_ids':factual_ids,'rows':rows,'ref':data['ref'],'f':f,'adj':adj,'base_failures':failures}
 torch.save(population,out/'population.pt');dump(out/'gradient_audit.json',audits)
 # Demonstrate why continuous midpoint optimization followed by decoding changes geometry.
 snap=[]
 for split,p in population.items():
  base=p['codes'][p['base_ids']].numpy();ref=p['ref'].numpy();D,assignment=w2(base,ref)
  for alpha in [.25,.5,.75]:
   q=(1-alpha)*base+alpha*ref[assignment];dist=((q[:,None]-p['codes'].numpy()[None])**2).sum(2);ix=dist.argmin(1);decoded=p['codes'].numpy()[ix];snap.append({'split':split,'alpha':alpha,'D':D,'continuous_geometry':geometry(base,q,ref),'snapped_geometry':geometry(base,decoded,ref),'mean_roundtrip_L2':float(np.linalg.norm(decoded-q,axis=1).mean()),'snapped_beta102_feasible':geometry(base,decoded,ref)<=1.02*D+1e-8})
 pd.DataFrame(snap).to_csv(out/'latent_snapping_audit.csv',index=False)
def futures(out,c,split,worker):
 data=load(out/'data.pt');pool=split+'_future';features=data['features'][pool];labels=data['labels'][pool];rows=data['parts'][pool];texts=[r['toxic'] for r in rows]+[r['neutral'] for r in rows];stats={'length':np.array([len(t.split()) for t in texts],float),'error':torch.nn.functional.binary_cross_entropy_with_logits(features@data['w']+data['b'],labels,reduction='none').numpy(),'topic':(features@data['topic']).numpy()};jobs=[(family,sev,rep) for family in c['future']['families'] for sev in c['future']['severities'] for rep in range(c['future']['reps'])];records=[];(out/'futures'/split).mkdir(parents=True,exist_ok=True)
 for job,(family,sev,rep) in enumerate(jobs):
  if job%8!=worker:continue
  seed=(191000 if split=='cal' else 201000)+job;rng=np.random.default_rng(seed);ids=[]
  for label in [0,1]:
   candidates=np.where(labels.numpy()==label)[0];v=stats[family][candidates];z=np.clip((v-v.mean())/max(v.std(),1e-12),-3,3);prob=np.exp(sev*z);prob/=prob.sum();ids.extend(rng.choice(candidates,64,replace=True,p=prob).tolist())
  xx=features[ids];yy=labels[ids];nw,nb=update(xx.cuda(),yy.cuda(),data['w'].cuda(),data['b'].cuda(),**c['update_params']);p=out/'futures'/split/f'{job:02d}.pt';torch.save({'w':nw.cpu(),'b':nb.cpu(),'indices':ids,'seed':seed},p);displacement=math.sqrt((w2(data['x'][:64],xx[:64])[0]**2+w2(data['x'][64:],xx[64:])[0]**2)/2);records.append({'job':job,'family':family,'severity':sev,'rep':rep,'seed':seed,'label_preserving_W2':displacement,'path':str(p),'sha256':sha(p),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
 pd.DataFrame(records).to_csv(out/f'future_{split}_{worker}.csv',index=False)
def optimize(out,c,split,r,beta):
 start=time.time();p=load(out/'population.pt')[split]
 # Candidate adjoints computed through full updates on GPU; norms for current population rechecked on assigned GPU.
 ids,deployed,trace,stopped=solve(p['codes'],p['base_ids'],p['ref'],p['f'],p['adj'],c['update']['n'],r,beta,c['solver']['sweeps'],c['solver']['geometry_tolerance']);v,n,s=objective(torch.tensor(ids,device='cuda'),p['f'].cuda(),p['adj'].cuda(),c['update']['n'],r);base=p['codes'][p['base_ids']];q=p['codes'][ids];D=w2(base,p['ref'])[0];d0=w2(base,q)[0];d1=w2(q,p['ref'])[0];assert d0+d1<=beta*D+1e-8;folder=out/split/tag(r,beta);folder.mkdir(parents=True);torch.save({'support_ids':ids,'deployed_ids':deployed,'codes':p['codes'][deployed],'texts':[p['texts'][i] for i in deployed]},folder/'output.pt');pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);dump(folder/'metrics.json',{'method':tag(r,beta),'split':split,'r':r,'beta':beta,'nominal':float(n),'sensitivity':float(s),'surrogate':float(v),'D':D,'W2_Base_Q':d0,'W2_Q_P1':d1,'budget':beta*D,'feasible':True,'coordinate_stopped':stopped,'sweeps':len(trace),'runtime_seconds':time.time()-start,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')});print('SOLVED',split,tag(r,beta),flush=True)
def word_distance(a,b):
 a=a.split();b=b.split();row=list(range(len(b)+1))
 for i,u in enumerate(a,1):
  nxt=[i]
  for j,v in enumerate(b,1):nxt.append(min(row[j]+1,nxt[-1]+1,row[j-1]+(u!=v)))
  row=nxt
 return row[-1]/max(1,len(a),len(b))
def freeze(out,c):
 manifest={str(out/'selection.json'):sha(out/'selection.json')}
 for r in c['r_grid']:
  for b in c['beta_grid']:
   path=out/'test'/tag(r,b)/'output.pt';manifest[str(path)]=sha(path)
 dump(out/'test_freeze.json',{'time':time.time(),'hashes':manifest})
def evaluate(out,c,split):
 data=load(out/'data.pt');p=load(out/'population.pt')[split];manifest=pd.concat([pd.read_csv(out/f'future_{split}_{i}.csv') for i in range(8)],ignore_index=True);manifest.to_csv(out/f'{split}_future_manifest.csv',index=False);models=[load(path) for path in manifest.path];methods={'Base':{'deployed_ids':p['base_ids']}};metrics={}
 for r in c['r_grid']:
  for b in c['beta_grid']:
   name=tag(r,b);methods[name]=load(out/split/name/'output.pt');metrics[name]=json.loads((out/split/name/'metrics.json').read_text())
 per=[];individual=[];summary=[];facts=p['codes'][p['factual_ids']]
 for name,output in methods.items():
  ids=output['deployed_ids'];q=p['codes'][ids];texts=[p['texts'][int(i)] for i in ids];current=(q@data['w']+data['b']).sigmoid();values=[]
  for rec,m in zip(manifest.to_dict('records'),models):
   prob=(q@m['w']+m['b']).sigmoid();fv=float((prob>=.5).double().mean());values.append(fv);per.append({'method':name,**rec,'FV':fv})
   for i,v in enumerate(prob):individual.append({'method':name,'job':rec['job'],'index':i,'probability':float(v)})
  v,n,s=objective(torch.tensor(ids),p['f'],p['adj'],c['update']['n'],metrics.get(name,{}).get('r',0));row={**metrics.get(name,{}),'method':name,'FV':np.mean(values),'FV_min':min(values),'current_validity':float((current>=.5).double().mean()),'semantic_cost':float((q-facts).norm(dim=1).mean()),'cosine_similarity':float((q*facts).sum(1).mean()),'word_edit':np.mean([word_distance(p['rows'][i]['factual'],t) for i,t in enumerate(texts)]),'unchanged_rate':np.mean([p['rows'][i]['factual'].strip()==t for i,t in enumerate(texts)]),'unique_outputs':len(set(texts)),'nominal':float(n),'sensitivity':float(s),'surrogate':float(v)};summary.append(row)
 pd.DataFrame(summary).to_csv(out/f'{split}_summary.csv',index=False);pd.DataFrame(per).to_csv(out/f'{split}_per_model.csv',index=False);pd.DataFrame(individual).to_csv(out/f'{split}_individual.csv',index=False)
 if split=='cal':
  table=pd.DataFrame(summary);best=table[table.method!='Base'].sort_values(['FV','semantic_cost','r','beta'],ascending=[False,True,True,True]).iloc[0];dump(out/'selection.json',{'method':best['method'],'r':float(best.r),'beta':float(best.beta),'FV':float(best.FV),'semantic_cost':float(best.semantic_cost),'time':time.time()})
 else:dump(out/'complete.json',{'time':time.time()})
 print('EVALUATED',split,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1.02);a=p.parse_args();c=json.loads(CONFIG.read_text());c['update_params']={'steps':c['update']['steps'],'lr':c['update']['lr']};torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.phase=='prepare':prepare(a.output,c)
 elif a.phase=='train':train_generator(a.output,c)
 elif a.phase=='generate':generate(a.output,c,a.worker)
 elif a.phase=='assemble':assemble(a.output,c)
 elif a.phase=='futures':futures(a.output,c,a.split,a.worker)
 elif a.phase=='optimize':optimize(a.output,c,a.split,a.r,a.beta)
 elif a.phase=='freeze':freeze(a.output,c)
 elif a.phase=='evaluate':evaluate(a.output,c,a.split)
 else:raise ValueError(a.phase)
if __name__=='__main__':main()
