"""Frozen-settings task021 evaluation on eight new eight-sentence populations."""
import argparse,json,math,os,re,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from lift_ce.experiments import text_toxicity_update_aware as old
from lift_ce.counterfactuals.sentence_codebook import cache,objective,solve,w2,geometry
from lift_ce.counterfactuals.adjoint_factor import factor_adjoint
ROOT=Path(__file__).resolve().parents[4]
CONFIG=ROOT/'experiments/configs/text_expanded_evaluation.json'
load,dump,sha,encode,encoder=old.load,old.dump,old.sha,old.encode,old.encoder
lex=lambda s:tuple(re.findall(r'\w+',s.lower()))

def prepare(out,c):
 d=load(Path(c['parent'])/'data.pt');prior=load(Path(c['original'])/'data.pt')
 used={old.normtext(t) for data in [d,prior] for rows in data['parts'].values() for row in rows for t in row.values()}
 frame=pd.read_csv(Path(c['assets'])/'paradetox/train.tsv',sep='\t');seen=set();fresh=[]
 for row in frame.to_dict('records'):
  a,b=str(row['en_toxic_comment']),str(row['en_neutral_comment']);aa,bb=old.normtext(a),old.normtext(b)
  if not aa or not bb or aa==bb or aa in seen or bb in seen:continue
  seen.update([aa,bb])
  if aa not in used and bb not in used and 5<=len(a.split())<=25:fresh.append({'toxic':a,'neutral':b})
 np.random.default_rng(c['seed']).shuffle(fresh);tok,enc=encoder(c);chosen=[]
 for start in range(0,len(fresh),128):
  rows=fresh[start:start+128];z=encode([r['toxic'] for r in rows],tok,enc);p=(z@d['w']+d['b']).sigmoid()
  chosen.extend(r for r,v in zip(rows,p) if v<.5)
  if len(chosen)>=c['n']:break
 assert len(chosen)>=c['n'];chosen=chosen[:c['n']]
 assert all(old.normtext(t) not in used for r in chosen for t in r.values())
 torch.save({'rows':chosen,'w':d['w'],'b':d['b'],'x':d['x'],'y':d['y'],'ref':d['ref'],'reftexts':[d['parts']['fit'][int(i)-len(d['parts']['fit'])]['neutral'] for i in d['ref_ids']]},out/'data.pt')
 files=[CONFIG,ROOT/c['task'],Path(c['parent'])/'data.pt',Path(c['original'])/'data.pt',Path(c['parent'])/'selection.json',*list((Path(c['original'])/'generator').glob('*')),*list((Path(c['parent'])/'futures/test').glob('*.pt'))]
 dump(out/'input_manifest.json',{str(p):sha(p) for p in files if p.is_file()});dump(out/'resolved_config.json',c);dump(out/'split_audit.json',{'excluded_prior_texts':len(used),'eligible_unused_pairs':len(fresh),'selected':len(chosen),'seed':c['seed'],'all_prior_partitions_excluded':True});print('PREPARED',len(chosen),flush=True)

def worker(out,c,block):
 from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
 folder=out/f'block{block}';folder.mkdir();d=load(out/'data.pt');rows=d['rows'][block*8:(block+1)*8];path=Path(c['original'])/'generator';tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True).cuda().eval();generated=[]
 for i,row in enumerate(rows):
  torch.manual_seed(210100+8*block+i);x=tok('detoxify: '+row['toxic'],return_tensors='pt',truncation=True,max_length=96).to('cuda')
  with torch.no_grad():
   beams=tok.batch_decode(model.generate(**x,max_new_tokens=64,num_beams=8,num_return_sequences=8,do_sample=False),skip_special_tokens=True);samples=[]
   for temp in [.8,1.1]:
    for _ in range(2):samples.extend(tok.batch_decode(model.generate(**x,max_new_tokens=64,num_return_sequences=16,do_sample=True,temperature=temp,top_p=.95),skip_special_tokens=True))
  generated.append({'factual':row['toxic'],'beams':beams,'samples':samples})
 dump(folder/'generated.json',generated);del model;torch.cuda.empty_cache();tok,enc=encoder(c)
 raw=list(dict.fromkeys(t.strip() for row in generated for t in row['beams']+row['samples']+[row['factual']] if t.strip()));raw += [t for t in d['reftexts'] if t not in raw];z=encode(raw,tok,enc);lookup={t:i for i,t in enumerate(raw)};bases=[];fails=[]
 for row in generated:
  fid=lookup[row['factual'].strip()];first=[lookup[t.strip()] for t in row['beams'][:4] if t.strip()] or [fid];p=(z[first]@d['w']+d['b']).sigmoid();valid=[j for j,v in zip(first,p) if v>=.5];j=min(valid,key=lambda j:float((z[j]-z[fid]).norm())) if valid else first[int(p.argmax())];bases.append(raw[j]);fails.append(not bool(valid))
 texts=list(dict.fromkeys(bases+d['reftexts']+[r['factual'].strip() for r in generated]));keys={lex(t) for t in texts}
 for t in raw:
  if lex(t) and lex(t) not in keys:texts.append(t);keys.add(lex(t))
 codes=z[[lookup[t] for t in texts]];lookup={t:i for i,t in enumerate(texts)};baseids=[lookup[t] for t in bases];fids=[lookup[r['factual'].strip()] for r in generated]
 f,adj=cache(d['x'].cuda(),d['y'].cuda(),codes.cuda(),d['w'].cuda(),d['b'].cuda(),**c['update']);factor=factor_adjoint(adj);torch.testing.assert_close(adj[baseids].mean(0).norm(),factor[baseids].mean(0).norm(),atol=1e-10,rtol=1e-7);f,adj,factor=f.cpu(),adj.cpu(),factor.cpu()
 torch.save({'texts':texts,'codes':codes,'base_ids':baseids,'factual_ids':fids,'base_failures':fails,'f':f,'adj':adj,'factor':factor},folder/'population.pt')
 for r in c['r_grid']:
  ids,deployed,trace,stopped=solve(codes,baseids,d['ref'],f,factor,128,r,c['beta'],10,1e-8);v,n,s=objective(torch.tensor(ids),f,adj,128,r);D=w2(codes[baseids],d['ref'])[0];g=geometry(codes[baseids],codes[ids],d['ref']);assert g<=c['beta']*D+1e-8;assert abs(float(v)-trace[-1]['surrogate'])<1e-9
  torch.save({'support_ids':ids,'deployed_ids':deployed,'codes':codes[deployed],'texts':[texts[i] for i in deployed]},folder/f'r{r:g}.pt');dump(folder/f'r{r:g}_metrics.json',{'r':r,'beta':c['beta'],'nominal':float(n),'sensitivity':float(s),'surrogate':float(v),'D':D,'geometry':g,'feasible':True,'stopped':stopped,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')});pd.DataFrame(trace).to_csv(folder/f'r{r:g}_trace.csv',index=False)
 dump(folder/'freeze.json',{'time':time.time(),'hashes':{str(p):sha(p) for p in folder.glob('*.pt')}});print('FROZEN',block,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['prepare','worker'],required=True);p.add_argument('--block',type=int,default=0);a=p.parse_args();c=json.loads(CONFIG.read_text());torch.set_num_threads(1);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.phase=='prepare':prepare(a.output,c)
 else:worker(a.output,c,a.block)
if __name__=='__main__':main()
