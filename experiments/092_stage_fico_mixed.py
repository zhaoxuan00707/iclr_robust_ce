"""Download pinned original FICO/HELOC CSV and data dictionary; no training."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/xu.zhao/robust_ce_data/fico_mixed_task100_v1')
COMMIT='056ca9505910db184c10d20457551ab48925e317'
def main():
 OUT.mkdir(parents=True,exist_ok=False);records=[]
 for name in ['heloc_dataset_v1.csv','heloc_data_dictionary-2.xlsx']:
  url=f'https://raw.githubusercontent.com/5teffen/FICO-xML-Challenge/{COMMIT}/xML%20Challenge%20Dataset%20and%20Data%20Dictionary/{name}'
  with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
  (OUT/name).write_bytes(data);records.append({'path':str(OUT/name),'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
 record={'repository':'https://github.com/5teffen/FICO-xML-Challenge','commit':COMMIT,'files':records,'original_provider':'FICO Explainable Machine Learning Challenge','mirror':'challenge participant repository; not a separate FICO dataset'}
 (OUT/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');res=ROOT/'results/100_extra_appendix/fico_data_audit';res.mkdir(parents=True,exist_ok=True);(res/'download_provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
