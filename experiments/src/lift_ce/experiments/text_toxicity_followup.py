"""Task020: fresh-cohort discrete-text coverage/budget/update follow-up."""
import argparse,json,math,os,re,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from lift_ce.experiments import text_toxicity_update_aware as old
from lift_ce.counterfactuals.sentence_codebook import update,losses,cache,objective,solve,w2,geometry
from lift_ce.counterfactuals.adjoint_factor import factor_adjoint
ROOT=Path(__file__).resolve().parents[4];CONFIG=ROOT/'experiments/configs/text_toxicity_followup.json'
load,dump,sha,tag,encode,encoder=old.load,old.dump,old.sha,old.tag,old.encode,old.encoder
lex=lambda s:tuple(re.findall(r'\w+',s.lower()))
def prepare(out,c):
 parent=Path(c['parent']);d=load(parent/'data.pt');used={old.normtext(t) for part in d['parts'].values() for row in part for t in row.values()};frame=pd.read_csv(Path(c['assets'])/'paradetox/train.tsv',sep='\t');seen=set();fresh=[]
 for row in frame.to_dict('records'):
  a,b=str(row['en_toxic_comment']),str(row['en_neutral_comment']);aa,bb=old.normtext(a),old.normtext(b)
  if not aa or not bb or aa==bb or aa in seen or bb in seen:continue
  seen.update([aa,bb])
  if aa in used or bb in used:continue
  fresh.append({'toxic':a,'neutral':b})
 rng=np.random.default_rng(c['seed']);rng.shuffle(fresh);assert len(fresh)>=sum(c['splits'].values());tok,enc=encoder(c);offset=0
 for split,n in c['splits'].items():
  rows=fresh[offset:offset+n];offset+=n;d['parts'][split]=rows;texts=[r['toxic'] for r in rows]+[r['neutral'] for r in rows];d['features'][split]=encode(texts,tok,enc);d['labels'][split]=torch.cat([torch.zeros(n),torch.ones(n)]).double();print('FRESH',split,n,flush=True)
 d['cases']={}
 for split in ['cal','test']:
  prob=(d['features'][split][:len(d['parts'][split])]@d['w']+d['b']).sigmoid();ids=[i for i,p in enumerate(prob) if p<.5 and 5<=len(d['parts'][split][i]['toxic'].split())<=25][:8];assert len(ids)==8;d['cases'][split]=ids
 n=len(d['parts']['nominal']);d['x']=torch.cat([d['features']['nominal'][:64],d['features']['nominal'][n:n+64]]);d['y']=torch.cat([torch.zeros(64),torch.ones(64)]).double();torch.save(d,out/'data.pt');dump(out/'resolved_config.json',c)
 files=[CONFIG,ROOT/c['task'],parent/'data.pt',*list((parent/'generator').glob('*')),*list(Path(c['assets']).rglob('*'))];dump(out/'input_manifest.json',{str(p):sha(p) for p in files if p.is_file()});dump(out/'split_audit.json',{'unused_pairs_available':len(fresh),'new_partitions':c['splits'],'old_partition_text_exclusion':True})
def generate(out,c,worker):
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 path=Path(c['parent'])/'generator';tok=AutoTokenizer.from_pretrained(path,local_files_only=True);m=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True).cuda().eval();d=load(out/'data.pt');records=[]
 for split in ['cal','test']:
  idx=d['cases'][split][worker];text=d['parts'][split][idx]['toxic'];x=tok('detoxify: '+text,return_tensors='pt',truncation=True,max_length=96).to('cuda');torch.manual_seed(200100+(1000 if split=='test' else 0)+worker)
  with torch.no_grad():
   beam=m.generate(**x,max_new_tokens=64,num_beams=8,num_return_sequences=8,do_sample=False);samples=[]
   for temp in [.8,1.1]:
    for _ in range(2):samples.extend(tok.batch_decode(m.generate(**x,max_new_tokens=64,num_return_sequences=16,do_sample=True,temperature=temp,top_p=.95),skip_special_tokens=True))
  records.append({'split':split,'index':worker,'source_index':idx,'factual':text,'beams':tok.batch_decode(beam,skip_special_tokens=True),'samples':samples});print('GENERATED',split,worker,flush=True)
 dump(out/f'generated_{worker}.json',records)
def assemble(out,c):
 d=load(out/'data.pt');tok,enc=encoder(c);allrows=sum([json.loads((out/f'generated_{i}.json').read_text()) for i in range(8)],[]);pops={};audits=[];opportunities=[]
 for split in ['cal','test']:
  rows=sorted([r for r in allrows if r['split']==split],key=lambda r:r['index']);nf=len(d['parts']['fit']);reftexts=[d['parts']['fit'][int(i)-nf]['neutral'] for i in d['ref_ids']];raw=list(dict.fromkeys(t.strip() for row in rows for t in row['beams']+row['samples']+[row['factual']] if t.strip()));raw+= [t for t in reftexts if t not in raw];z=encode(raw,tok,enc);lookup={t:i for i,t in enumerate(raw)};basetext=[];fails=[]
  for row in rows:
   fact=lookup[row['factual'].strip()];first=[lookup[t.strip()] for t in row['beams'][:4] if t.strip()] or [fact];p=(z[first]@d['w']+d['b']).sigmoid();valid=[j for j,prob in zip(first,p) if prob>=.5];j=min(valid,key=lambda j:float((z[j]-z[fact]).norm())) if valid else first[int(p.argmax())];basetext.append(raw[j]);fails.append(not bool(valid))
  # Anchor strings are immutable; additional candidates are lexically deduplicated.
  texts=list(dict.fromkeys(basetext+reftexts+[row['factual'].strip() for row in rows]));keys={lex(t) for t in texts}
  for t in raw:
   if lex(t) and lex(t) not in keys:texts.append(t);keys.add(lex(t))
  codes=z[[lookup[t] for t in texts]];lookup={t:i for i,t in enumerate(texts)};ids=[lookup[t] for t in basetext];fids=[lookup[row['factual'].strip()] for row in rows]
  f,adj=cache(d['x'].cuda(),d['y'].cuda(),codes.cuda(),d['w'].cuda(),d['b'].cuda(),**c['update_params']);factor=factor_adjoint(adj);ix=torch.tensor(ids,device='cuda');torch.testing.assert_close(adj[ix].mean(0).norm(),factor[ix].mean(0).norm(),rtol=1e-7,atol=1e-10)
  f,adj,factor=f.cpu(),adj.cpu(),factor.cpu();x=d['x'].clone().requires_grad_();w,b=update(x,d['y'],d['w'],d['b'],**c['update_params']);nom=losses(codes[ids],w,b).mean();g=torch.autograd.grad(nom,x)[0];v,n,s=objective(torch.tensor(ids),f,adj,128,.25);torch.testing.assert_close(n,nom,atol=1e-10,rtol=1e-7);torch.testing.assert_close(s,math.sqrt(128)*g.norm(),atol=1e-10,rtol=1e-7);direction=torch.randn_like(x);direction/=direction.norm();eps=1e-4
  def fun(xx):ww,bb=update(xx,d['y'],d['w'],d['b'],**c['update_params']);return losses(codes[ids],ww,bb).mean()
  fd=float((fun(x+eps*direction)-fun(x-eps*direction))/(2*eps));ad=float((g*direction).sum());err=abs(fd-ad)/max(abs(fd),abs(ad),1e-12);assert err<1e-3
  pops[split]={'texts':texts,'codes':codes,'base_ids':ids,'factual_ids':fids,'rows':rows,'ref':d['ref'],'f':f,'adj':adj,'factor':factor,'base_failures':fails};audits.append({'split':split,'raw_distinct_candidates':len(raw),'lexical_candidates':len(texts),'base_failures':sum(fails),'finite_difference_relative_error':err})
  base=codes[ids];D=w2(base,d['ref'])[0]
  for k,idx in enumerate(ids):
   ratios=[]
   for j,t in enumerate(texts):
    if lex(t)==lex(texts[idx]):continue
    proposal=ids.copy();proposal[k]=j;ratios.append(geometry(base,codes[proposal],d['ref'])/D)
   for beta in c['beta_grid']:opportunities.append({'split':split,'index':k,'beta':beta,'feasible_word_changing_candidates':sum(v<=beta+1e-8 for v in ratios)})
  print('AUDITED',audits[-1],flush=True)
 torch.save(pops,out/'population.pt');dump(out/'gradient_audit.json',audits);pd.DataFrame(opportunities).to_csv(out/'lexical_opportunities.csv',index=False)
def optimize(out,c,split,r,beta):
 start=time.time();p=load(out/'population.pt')[split];ids,deployed,trace,stopped=solve(p['codes'],p['base_ids'],p['ref'],p['f'],p['factor'],128,r,beta,c['solver']['sweeps'],c['solver']['geometry_tolerance']);v,n,s=objective(torch.tensor(ids,device='cuda'),p['f'].cuda(),p['adj'].cuda(),128,r);base=p['codes'][p['base_ids']];q=p['codes'][ids];D=w2(base,p['ref'])[0];d0=w2(base,q)[0];d1=w2(q,p['ref'])[0];assert d0+d1<=beta*D+1e-8;assert abs(float(v)-trace[-1]['surrogate'])<1e-9;folder=out/split/tag(r,beta);folder.mkdir(parents=True);torch.save({'support_ids':ids,'deployed_ids':deployed,'codes':p['codes'][deployed],'texts':[p['texts'][i] for i in deployed]},folder/'output.pt');pd.DataFrame(trace).to_csv(folder/'trace.csv',index=False);dump(folder/'metrics.json',{'method':tag(r,beta),'split':split,'r':r,'beta':beta,'nominal':float(n),'sensitivity':float(s),'surrogate':float(v),'D':D,'W2_Base_Q':d0,'W2_Q_P1':d1,'budget':beta*D,'feasible':True,'coordinate_stopped':stopped,'sweeps':len(trace),'runtime_seconds':time.time()-start,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')});print('SOLVED',split,tag(r,beta),flush=True)

def futures(out,c,split,worker):
 data=load(out/'data.pt');pool=split+'_future';features=data['features'][pool];labels=data['labels'][pool];rows=data['parts'][pool];texts=[r['toxic'] for r in rows]+[r['neutral'] for r in rows];stats={'length':np.array([len(t.split()) for t in texts],float),'error':torch.nn.functional.binary_cross_entropy_with_logits(features@data['w']+data['b'],labels,reduction='none').numpy(),'topic':(features@data['topic']).numpy()};jobs=[(family,sev,rep) for family in c['future']['families'] for sev in c['future']['severities'] for rep in range(c['future']['reps'])];records=[];(out/'futures'/split).mkdir(parents=True,exist_ok=True)
 for job,(family,sev,rep) in enumerate(jobs):
  if job%8!=worker:continue
  seed=(211000 if split=='cal' else 221000)+job;rng=np.random.default_rng(seed);ids=[]
  for label in [0,1]:
   candidates=np.where(labels.numpy()==label)[0];v=stats[family][candidates];z=np.clip((v-v.mean())/max(v.std(),1e-12),-3,3);prob=np.exp(sev*z);prob/=prob.sum();ids.extend(rng.choice(candidates,64,replace=True,p=prob).tolist())
  xx=features[ids];yy=labels[ids];nw,nb=update(xx.cuda(),yy.cuda(),data['w'].cuda(),data['b'].cuda(),**c['update_params']);p=out/'futures'/split/f'{job:02d}.pt';torch.save({'w':nw.cpu(),'b':nb.cpu(),'indices':ids,'seed':seed},p);displacement=math.sqrt((w2(data['x'][:64],xx[:64])[0]**2+w2(data['x'][64:],xx[64:])[0]**2)/2);records.append({'job':job,'family':family,'severity':sev,'rep':rep,'seed':seed,'label_preserving_W2':displacement,'path':str(p),'sha256':sha(p),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
 pd.DataFrame(records).to_csv(out/f'future_{split}_{worker}.csv',index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--split',default='cal');p.add_argument('--r',type=float,default=0);p.add_argument('--beta',type=float,default=1.02);a=p.parse_args();c=json.loads(CONFIG.read_text());c['update_params']={'steps':c['update']['steps'],'lr':c['update']['lr']};torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.phase=='prepare':prepare(a.output,c)
 elif a.phase=='generate':generate(a.output,c,a.worker)
 elif a.phase=='assemble':assemble(a.output,c)
 elif a.phase=='futures':futures(a.output,c,a.split,a.worker)
 elif a.phase=='optimize':optimize(a.output,c,a.split,a.r,a.beta)
 elif a.phase=='freeze':old.freeze(a.output,c)
 elif a.phase=='evaluate':old.evaluate(a.output,c,a.split)
 else:raise ValueError(a.phase)
if __name__=='__main__':main()
