"""Verify the downloaded original codebook and existing mixed-feature loader."""
import sys,json,hashlib,zipfile,xml.etree.ElementTree as E
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/src'))
import pandas as pd,numpy as np,torch
from lift_ce.data.tabular import load_tabular_three_way

def main():
 cfg=json.loads((ROOT/'experiments/configs/fico_mixed_task100.json').read_text());raw=Path(cfg['path']).parent;out=ROOT/'results/100_extra_appendix/fico_data_audit';out.mkdir(parents=True,exist_ok=True)
 for rec in json.loads((raw/'provenance.json').read_text())['files']:assert hashlib.sha256(Path(rec['path']).read_bytes()).hexdigest()==rec['sha256']
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};sheets={}
 with zipfile.ZipFile(raw/'heloc_data_dictionary-2.xlsx') as z:
  strings=[''.join(e.itertext()) for e in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  for name in z.namelist():
   if not(name.startswith('xl/worksheets/sheet') and name.endswith('.xml')):continue
   rows=[]
   for row in E.fromstring(z.read(name)).findall('.//s:row',ns):
    values=[]
    for cell in row.findall('s:c',ns):
     v=cell.find('s:v',ns)
     if v is not None:values.append(strings[int(v.text)] if cell.attrib.get('t')=='s' else v.text)
    if values:rows.append(values)
   sheets[name]=rows
 (out/'dictionary_extracted.json').write_text(json.dumps(sheets,indent=2)+'\n')
 frame=pd.read_csv(cfg['path']);assert frame.shape==(10459,24);assert set(frame.RiskPerformance)=={'Good','Bad'}
 d=load_tabular_three_way(cfg,str(raw),.6,.2,0);splits=['train','cal','test'];checks={}
 for split in splits:assert torch.isfinite(d[split+'_x']).all()
 for i,a in enumerate(splits):
  for b in splits[i+1:]:assert not set(d[a+'_ids'])&set(d[b+'_ids'])
 for col in cfg['categorical_columns']:
  ids=[i for i,n in enumerate(d['feature_names']) if n.startswith(col+'_')];assert len(ids)>1
  vertices=torch.unique(d['train_x'][:,ids],dim=0)
  for split in splits:
   errors=(d[split+'_x'][:,None,ids]-vertices[None]).abs().amax(2).min(1).values;assert errors.max()<1e-6,(col,split,float(errors.max()))
  checks[col]={'encoded_indices':ids,'encoded_names':[d['feature_names'][i] for i in ids],'training_vertices':len(vertices),'raw_value_counts':{str(k):int(v) for k,v in frame[col].value_counts().sort_index().items()}}
 summary={'rows':len(frame),'predictors':23,'numeric_predictors':21,'categorical_predictors':2,'seed':0,'split_sizes':{s:len(d[s+'_ids']) for s in splits},'encoded_dimension':len(d['feature_names']),'categorical_blocks':checks,'finite':True,'disjoint_splits':True,'legal_encoded_vertices':True,'all_missing_predictor_rows':int((frame.drop(columns='RiskPerformance')==-9).all(axis=1).sum()),'source_hashes_verified':True}
 (out/'schema_audit.json').write_text(json.dumps(summary,indent=2)+'\n');pd.DataFrame([{'split':s,'original_row_id':int(i)} for s in splits for i in d[s+'_ids']]).to_csv(out/'split_ids_seed0.csv',index=False);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
