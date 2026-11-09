"""Diagnose failed float32 CUDA finite differences without retraining or overwriting."""
import argparse
import json
from pathlib import Path
import sys
import types
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments/src'),'/tmp/ffhq_grace_deps']
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from lift_ce.ffhq.genki_classifier import GenkiSmileClassifier
from lift_ce.experiments.genki_ffhq_classifier_v1 import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    checkpoint=a.run/'checkpoints/selected.pt';model=GenkiSmileClassifier(checkpoint,'cpu').double()
    # CLIP's custom LayerNorm forces float32 for half-precision inference. In this
    # precision-only audit, use the identical layer-normalization formula in fp64.
    for layer in model.modules():
        if isinstance(layer,torch.nn.LayerNorm):
            layer.forward=types.MethodType(lambda self,x:torch.nn.functional.layer_norm(x,self.normalized_shape,self.weight,self.bias,self.eps),layer)
    table=pd.read_csv(a.run/'transfer_predictions.csv');rows=[];scores=[]
    for r in table.itertuples():
        native=(TF.to_tensor(Image.open(r.image_path).convert('RGB'))[None]*2-1).double()
        with torch.no_grad():p=float(model(native).softmax(1)[0,0])
        scores.append(dict(index=r.index,kind=r.kind,cpu_float64_probability=p,
                           cuda_float32_probability=r.genki_smile_probability,absolute_difference=abs(p-r.genki_smile_probability)))
        if r.kind!='factual':continue
        x=model.canonical(native).detach().requires_grad_(True)
        def score(x):return model.logits_features(model.features(x)).softmax(1)[0,0]
        g,=torch.autograd.grad(score(x),x);direction=g/g.norm();auto=float(g.norm())
        for eps in [.01,.001,.0001]:
            with torch.no_grad():fd=float((score(x+eps*direction)-score(x-eps*direction))/(2*eps))
            error=abs(fd-auto)/max(auto,1e-8)
            rows.append(dict(index=r.index,epsilon=eps,autograd=auto,finite_difference=fd,relative_error=error,passed=bool(auto>0 and error<.03)))
        print('AUDITED',r.index,flush=True)
    d=pd.DataFrame(rows);s=pd.DataFrame(scores);d.to_csv(a.output/'gradient_checks.csv',index=False);s.to_csv(a.output/'precision_scores.csv',index=False)
    primary=d[d.epsilon==.001];small=d[d.epsilon==.0001]
    checks={'diagnostic_reason':'Three original CUDA float32 finite differences exceeded 3% tolerance; preserve original checks.',
            'precision':'CPU float64, same frozen selected model and canonical-image derivative',
            'layer_norm_audit_override':'Identical layer_norm formula in float64; disable CLIP custom fp32 cast only in this audit instance',
            'original_epsilon':.001,'tolerance_unchanged':.03,'primary_pass_count':int(primary.passed.sum()),
            'smaller_epsilon_pass_count':int(small.passed.sum()),'primary_max_relative_error':float(primary.relative_error.max()),
            'maximum_cpu64_cuda32_probability_difference':float(s.absolute_difference.max()),
            'probability_agreement_tolerance':.001,'checkpoint_sha256':sha(checkpoint),
            'test_metrics_recomputed':False,'head_retrained':False,'script_sha256':sha(__file__)}
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert len(primary)==8 and primary.passed.all() and small.passed.all()
    assert s.absolute_difference.max()<.001
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
