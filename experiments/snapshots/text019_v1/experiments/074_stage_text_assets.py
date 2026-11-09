"""Download public pinned text assets to a fresh external directory."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path('/home/xu.zhao/robust_ce_data/text019_assets_v1')
def get(url):
 with urllib.request.urlopen(url,timeout=180) as r:return r.read()
def main():
 ROOT.mkdir(parents=True,exist_ok=False);manifest=[]
 specs=[('dataset','s-nlp/paradetox',['.csv','README.md','LICENSE']),('model','google/flan-t5-small',['config.json','generation_config.json','model.safetensors','spiece.model','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md']),('model','sentence-transformers/all-MiniLM-L6-v2',['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','README.md'])]
 for typ,repo,wanted in specs:
  meta=json.loads(get('https://huggingface.co/api/'+('datasets/' if typ=='dataset' else 'models/')+repo));rev=meta['sha'];folder=ROOT/repo.split('/')[-1];folder.mkdir()
  names=[v['rfilename'] for v in meta['siblings'] if v['rfilename'] in wanted or (typ=='dataset' and v['rfilename'].endswith('.csv'))]
  for name in names:
   url='https://huggingface.co/'+('datasets/' if typ=='dataset' else '')+repo+'/resolve/'+rev+'/'+name;p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);payload=get(url);p.write_bytes(payload);manifest.append({'repo':repo,'revision':rev,'file':str(p),'url':url,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)});print(repo,name,len(payload),flush=True)
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
