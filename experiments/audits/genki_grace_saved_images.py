"""Uniform saved-image evaluation of already frozen CFs; no selection or training."""
import argparse,json,hashlib
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments/src'),'/tmp/ffhq_grace_deps']
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from lift_ce.ffhq.genki_classifier import GenkiSmileClassifier


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(4)
    frozen=json.loads((a.run/'frozen.json').read_text());timestamp=datetime.fromisoformat(frozen['utc']).timestamp()
    clf=GenkiSmileClassifier(a.run/'checkpoints/f0.pt','cpu');quality=pd.read_csv(a.run/'image_quality.csv')
    cache={};images=[];feature_rows=[]
    for r in quality.itertuples():
        path=a.run/'qualitative_examples'/f'test_{r.factual_index}_{r.method}.png'
        assert path.stat().st_mtime<=timestamp+.01
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in cache:
            native=TF.to_tensor(Image.open(path).convert('RGB'))[None]*2-1
            with torch.no_grad():cache[digest]=clf.features(clf.canonical(native))
        f=cache[digest];feature_rows.append(f)
        with torch.no_grad():p=float(clf.head(f).softmax(1)[0,0])
        images.append(dict(method=r.method,factual_index=r.factual_index,image_path=str(path),sha256=digest,nominal_smile_probability=p))
    d=pd.DataFrame(images);x=torch.cat(feature_rows);rows=[];bank=[]
    def probs(state):return (x@(state['weight'][0]-state['weight'][1])+state['bias'][0]-state['bias'][1]).sigmoid()
    for job in sorted((a.run/'jobs').glob('future_*/model.json')):
        meta=json.loads(job.read_text());p=probs(torch.load(meta['checkpoint'],weights_only=True,map_location='cpu'))
        rows += [dict(method=r.method,factual_index=r.factual_index,**meta['scenario'],smile_probability=float(value),valid=bool(value>=.5)) for r,value in zip(d.itertuples(),p)]
    assert len(rows)==1200
    records=pd.read_csv(a.run/'attack_bank_manifest.csv');spec=json.loads((a.run/'selected_configuration.json').read_text())
    records=records[records.paired_W_data_upper_bound<=spec['r_star']+1e-6]
    for r in records.itertuples():
        p=probs(torch.load(r.checkpoint,weights_only=True,map_location='cpu'))
        bank += [dict(candidate=r.candidate,method=q.method,factual_index=q.factual_index,smile_probability=float(v)) for q,v in zip(d.itertuples(),p)]
    d.to_csv(a.output/'image_predictions.csv',index=False);pd.DataFrame(rows).to_csv(a.output/'future_per_case.csv',index=False);pd.DataFrame(bank).to_csv(a.output/'bank_per_case.csv',index=False)
    for i in range(8):
        a0=d[(d.method=='Base')&(d.factual_index==i)].iloc[0];b0=d[(d.method=='GRACE_beta1.00')&(d.factual_index==i)].iloc[0]
        assert a0.sha256==b0.sha256 and a0.nominal_smile_probability==b0.nominal_smile_probability
    checks={'reason':'Identical saved Base and beta1 images had different cached features due to numerical realization paths; evaluate all saved images uniformly.',
            'saved_images':len(d),'unique_saved_images':len(cache),'saved_images_preceded_freeze':True,'all_evaluated_on_same_cpu_float32_path':True,
            'counterfactuals_or_configuration_changed':False,'heads_retrained':False,'identical_base_beta1_scores_verified':True,
            'timing':'post-hoc consistency audit; all outputs and selected configuration remain frozen','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(pd.DataFrame(rows).groupby('method').valid.mean().to_string(),flush=True)


if __name__=='__main__':main()
