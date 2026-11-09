"""GENKI-4K supervised linear probe, with independent held-out evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
import torch
from torch import nn
from torchvision.transforms import functional as TF
from ..ffhq.genki_classifier import CLASS_ORDER, GenkiSmileClassifier, canonical, image_features, load_visual

ROOT = Path(__file__).resolve().parents[4]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def split_groups(table, seed):
    """Keep exact decoded-image duplicates in the same split; labels must agree."""
    assert table.groupby('pixel_sha256').smile.nunique().max() == 1
    groups = table.drop_duplicates('pixel_sha256')
    train, rest = train_test_split(groups, test_size=.3, random_state=seed, stratify=groups.smile)
    val, test = train_test_split(rest, test_size=.5, random_state=seed, stratify=rest.smile)
    mapping = {h: name for name, part in [('train', train), ('validation', val), ('test', test)] for h in part.pixel_sha256}
    result = table.copy(); result['split'] = result.pixel_sha256.map(mapping)
    assert result.groupby('pixel_sha256').split.nunique().max() == 1
    return result


def metrics(y, p):
    return dict(count=len(y), accuracy=float(accuracy_score(y, p>=.5)),
                balanced_accuracy=float(balanced_accuracy_score(y, p>=.5)),
                log_loss=float(log_loss(y, np.c_[1-p,p], labels=[0,1])),
                auroc=float(roc_auc_score(y,p)),
                confusion_matrix_labels_0_1=confusion_matrix(y,p>=.5,labels=[0,1]).tolist())


def initialize(a,c):
    out=a.output
    for name in ['dataset','jobs','logs','features','checkpoints','figures']:(out/name).mkdir(exist_ok=True)
    archive=Path(c['dataset_archive'])
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            dest=(out/'dataset'/m.name).resolve()
            if not dest.is_relative_to((out/'dataset').resolve()) or not (m.isfile() or m.isdir()):
                raise ValueError('Unsafe archive member '+m.name)
        tar.extractall(out/'dataset', filter='data')
    values=np.loadtxt(out/'dataset/labels.txt');assert values.shape==(4000,4)
    records=[]
    for i,row in enumerate(values,1):
        p=out/'dataset/files'/f'file{i:04d}.jpg';im=Image.open(p).convert('RGB');assert row[0] in (0,1)
        pixhash=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        records.append(dict(index=i,image_path=str(p),smile=int(row[0]),yaw=row[1],pitch=row[2],roll=row[3],
                            sha256=sha(p),pixel_sha256=pixhash,width=im.width,height=im.height))
    table=split_groups(pd.DataFrame(records),c['seed']);table.to_csv(out/'dataset_manifest.csv',index=False)
    transfer=pd.read_csv(c['transfer_manifest']);transfer=transfer[transfer.domain=='generated_calibration'].copy()
    assert len(transfer)==16;transfer.to_csv(out/'transfer_inputs.csv',index=False)
    inputs={str(archive):sha(archive),c['clip_checkpoint']:sha(c['clip_checkpoint']),c['transfer_manifest']:sha(c['transfer_manifest'])}
    inputs.update({p:sha(p) for p in transfer.image_path})
    dump(out/'input_manifest.json',inputs);dump(out/'resolved_config.json',c)
    dump(out/'split_summary.json',{'counts':table.groupby(['split','smile']).size().unstack(fill_value=0).to_dict('index'),
                                 'duplicate_images':int(table.pixel_sha256.duplicated().sum()),
                                 'identity_disjoint':False,'reason':'GENKI-4K release does not provide subject IDs',
                                 'no_exact_decoded_image_cross_split_overlap':True})


def extract(a,c):
    lane=int(a.job.split('_')[-1]);table=pd.read_csv(a.output/'dataset_manifest.csv')
    table=table.iloc[lane::8];visual=load_visual(c['clip_checkpoint'],a.device)
    feats=[]
    for start in range(0,len(table),c['batch_size']):
        paths=table.image_path.iloc[start:start+c['batch_size']]
        x=torch.cat([canonical(TF.to_tensor(Image.open(p).convert('RGB'))[None]*2-1) for p in paths]).to(a.device)
        with torch.no_grad():feats.append(image_features(visual,x).cpu())
    torch.save({'index':torch.tensor(table['index'].tolist()),'features':torch.cat(feats)},a.output/'features'/f'{lane}.pt')


def features(out):
    values=[torch.load(out/'features'/f'{i}.pt',weights_only=True,map_location='cpu') for i in range(8)]
    ids=torch.cat([v['index'] for v in values]);x=torch.cat([v['features'] for v in values]);order=ids.argsort()
    assert ids[order].tolist()==list(range(1,4001));return x[order]


def fit(a,c):
    index=int(a.job.split('_')[-1]);penalty=c['l2_penalties'][index]
    table=pd.read_csv(a.output/'dataset_manifest.csv');x=features(a.output).to(a.device,dtype=torch.float64)
    y=torch.tensor(1-table.smile.to_numpy(),device=a.device)
    tr=torch.tensor((table.split=='train').to_numpy(),device=a.device);va=torch.tensor((table.split=='validation').to_numpy(),device=a.device)
    head=nn.Linear(c['feature_dimension'],2,dtype=torch.float64,device=a.device)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=300,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad();loss=nn.functional.cross_entropy(head(x[tr]),y[tr])+penalty*.5*head.weight.square().sum();loss.backward();return loss
    optimizer.step(closure)
    with torch.no_grad():p=head(x[va]).softmax(1)[:,0].cpu().numpy()
    m=metrics(table.loc[table.split=='validation','smile'].to_numpy(),p);m.update(candidate=index,l2=penalty)
    dump(a.output/'jobs'/a.job/'validation.json',m)
    torch.save({'head_state':{k:v.detach().float().cpu() for k,v in head.state_dict().items()},
                'class_order':CLASS_ORDER,'feature_dimension':c['feature_dimension'],'clip_checkpoint':c['clip_checkpoint'],
                'clip_sha256':sha(c['clip_checkpoint']),'l2':penalty,'seed':c['seed']},a.output/'checkpoints'/f'candidate_{index}.pt')


def select(a,c):
    rows=[json.loads((a.output/'jobs'/f'fit_{i}'/'validation.json').read_text()) for i in range(8)]
    ranked=sorted(rows,key=lambda r:(-r['balanced_accuracy'],r['log_loss'],-r['l2']))
    best=ranked[0];source=a.output/'checkpoints'/f'candidate_{best["candidate"]}.pt'
    shutil.copyfile(source,a.output/'checkpoints/selected.pt')
    pd.DataFrame(rows).to_csv(a.output/'validation_candidates.csv',index=False)
    dump(a.output/'selection_frozen.json',{'selected':best,'checkpoint_sha256':sha(source),
                                        'utc':datetime.now(timezone.utc).isoformat(),
                                        'test_metrics_accessed':False,'FFHQ_predictions_accessed_for_selection':False})


def evaluate(a,c):
    out=a.output;frozen=json.loads((out/'selection_frozen.json').read_text());checkpoint=out/'checkpoints/selected.pt'
    assert sha(checkpoint)==frozen['checkpoint_sha256']
    data=torch.load(checkpoint,weights_only=True,map_location='cpu');head=nn.Linear(512,2);head.load_state_dict(data['head_state']);head.eval()
    table=pd.read_csv(out/'dataset_manifest.csv');mask=table.split=='test'
    with torch.no_grad():p=head(features(out)[mask.to_numpy()]).softmax(1)[:,0].numpy()
    test=table.loc[mask,['index','smile']].copy();test['smile_probability']=p;test.to_csv(out/'test_predictions.csv',index=False)
    dump(out/'test_metrics.json',metrics(test.smile.to_numpy(),p))
    adapter=GenkiSmileClassifier(checkpoint,a.device);transfer=pd.read_csv(out/'transfer_inputs.csv');probs=[];checks=[]
    for r in transfer.itertuples():
        native=(TF.to_tensor(Image.open(r.image_path).convert('RGB'))[None]*2-1).to(a.device)
        with torch.no_grad():probs.append(float(adapter(native).softmax(1)[0,0]))
        if r.kind=='factual':
            x=adapter.canonical(native).detach().requires_grad_(True)
            def score(v):return adapter.logits_features(adapter.features(v)).softmax(1)[0,0]
            g,=torch.autograd.grad(score(x),x);direction=g/g.norm()
            with torch.no_grad():fd=float((score(x+.001*direction)-score(x-.001*direction))/.002)
            auto=float(g.norm());error=abs(fd-auto)/max(auto,1e-8)
            # Verify native interface returns the same logits as canonical feature path.
            with torch.no_grad():interface_error=float((adapter(native)-adapter.logits_features(adapter.features(adapter.canonical(native)))).abs().max())
            checks.append(dict(index=r.index,autograd=auto,finite_difference=fd,relative_error=error,
                               interface_error=interface_error,passed=bool(auto>0 and error<.03 and interface_error<1e-6)))
    transfer['genki_smile_probability']=probs;transfer['genki_smile_label']=(np.array(probs)>=.5).astype(int)
    transfer.to_csv(out/'transfer_predictions.csv',index=False);pd.DataFrame(checks).to_csv(out/'adapter_checks.csv',index=False)
    assert all(r['passed'] for r in checks),checks
    assert all(sha(p)==digest for p,digest in json.loads((out/'input_manifest.json').read_text()).items())
    dump(out/'evaluation_checks.json',{'eight_adapter_gradient_checks_passed':True,'source_hashes_unchanged':True,
                                     'human_generated_face_labels_available':False,'FFHQ_test_accessed':False,
                                     'new_counterfactuals_generated':False,'old_flow_or_bank_reused_with_new_classifier':False})


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['initialize','extract','fit','select','evaluate'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(20260917)
    cfg=a.output/'resolved_config.json';c=json.loads((cfg if cfg.exists() else ROOT/'experiments/configs/genki_ffhq_classifier_v1.json').read_text())
    if a.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(c['gpu_memory_fraction']);torch.backends.cudnn.benchmark=False
    job=a.output/'jobs'/a.job;job.mkdir(parents=True,exist_ok=True)
    if (job/'complete.json').exists():return
    start=time.time();meta={'phase':a.phase,'job':a.job,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'start_utc':datetime.now(timezone.utc).isoformat()}
    dump(job/'started.json',meta);globals()[a.phase](a,c)
    dump(job/'complete.json',{**meta,'end_utc':datetime.now(timezone.utc).isoformat(),'wall_seconds':time.time()-start})


if __name__=='__main__':main()
