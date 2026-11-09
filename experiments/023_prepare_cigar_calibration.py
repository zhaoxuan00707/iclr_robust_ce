"""Repair failed pair preparation without changing any existing sample/checkpoint output.
Cigar has no unused training images: reserve 16 of its 33 original test images
for calibration and select 16 test images from the disjoint remainder.
"""
import json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'experiments/src'))
from lift_ce.galaxy.assets import TRAIN,ALL_TRAIN,TEST,CLASS_NAMES
from lift_ce.experiments.galaxy_grace_v1 import sha,dump,stamp,load
out=Path('/home/xu.zhao/robust_ce_results/galaxy_grace_v1/20260916_pairs_round_cigar_v1')
c=json.loads((out/'resolved_config.json').read_text());n=c['factuals_per_target_per_split']
if (out/'prepared.json').exists():raise SystemExit('already prepared')
transform=transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(150),transforms.Resize(128,antialias=True)])
train=ImageFolder(str(TRAIN),transform=transform);full=ImageFolder(str(ALL_TRAIN),transform=transform);test=ImageFolder(str(TEST),transform=transform)
known={Path(p).name for p,y in train.samples};cal=[i for i,(p,y) in enumerate(full.samples) if Path(p).name not in known]
x=load(out/'training.pt')['x'];manifest=[]
for target in c['task_targets']:
 td=out/'tasks'/f'target{target:02d}';td.mkdir(exist_ok=True);g=torch.Generator().manual_seed(c['seed']+target)
 choices=torch.tensor([i for i in range(len(train)) if train.targets[i]==target]);ids=choices[torch.randperm(len(choices),generator=g)[:c['references_per_target']]]
 if (td/'target_images.pt').exists():
  assert torch.equal(load(td/'target_images.pt'),x[ids]);assert np.array_equal(np.load(td/'target_ids.npy'),np.array([train.samples[i][0] for i in ids]))
 else:torch.save(x[ids],td/'target_images.pt');np.save(td/'target_ids.npy',np.array([train.samples[i][0] for i in ids]))
 source=c['source_by_target'][str(target)]
 if source==4:
  choices=torch.tensor([i for i,y in enumerate(test.targets) if y==4]);chosen=choices[torch.randperm(len(choices),generator=g)[:2*n]].tolist();assert len(chosen)==2*n
  specs=[('calibration',test,chosen[:n]),('test',test,chosen[n:])]
 else:
  specs=[]
  for split,ds,indices in [('calibration',full,cal),('test',test,range(len(test)))]:
   choices=torch.tensor([i for i in indices if ds.targets[i]==source]);chosen=choices[torch.randperm(len(choices),generator=g)[:n]].tolist();specs.append((split,ds,chosen))
 for split,ds,chosen in specs:
  paths=[ds.samples[i][0] for i in chosen];images=torch.stack([ds[i][0] for i in chosen]);labels=torch.tensor([ds.targets[i] for i in chosen]);assert labels.eq(source).all()
  assert not (td/f'{split}_factual.pt').exists();torch.save({'images':images,'labels':labels},td/f'{split}_factual.pt');np.save(td/f'{split}_factual_ids.npy',np.array(paths))
  manifest.extend([{'task':target,'source_class':source,'split':split,'row':j,'path':path} for j,path in enumerate(paths)])
import pandas as pd
pd.DataFrame(manifest).to_csv(out/'factual_manifest.csv',index=False)
np.save(out/'factual_ids.npy',np.array([r['path'] for r in manifest if r['split']=='test']));np.save(out/'target_ids.npy',np.concatenate([np.load(out/'tasks'/f'target{t:02d}'/'target_ids.npy') for t in c['task_targets']]))
assert not {r['path'] for r in manifest if r['split']=='calibration'}.intersection({r['path'] for r in manifest if r['split']=='test'})
amendment={'reason':'Cigar class has zero original training-complement calibration images','policy':'For Cigar source only, first16 seeded test samples reserved for calibration; next16 for final testing; no overlap, no outcome selection. Original f0 used old test folder for validation, as already disclosed.','source_code':str(Path(__file__)),'source_sha256':sha(Path(__file__)),'utc':stamp(),'original_test_cigar_count':33}
dump(out/'cohort_amendment.json',amendment)
dump(out/'dataset_manifest.json',{'class_names':CLASS_NAMES,'train_size':len(train),'calibration_pool_size':len(cal),'test_size':len(test),'factuals_per_target_per_split':n,'source_by_target':c['source_by_target'],'cohort_amendment':amendment,'files_sha256':sha(out/'dataset_files.csv'),'original_classifier_validation_test_reuse':True})
shutil.copyfile(out/'jobs/prepare/started.json',out/'jobs/prepare/failed_initial_started.json')
dump(out/'prepared.json',{'training_sha256':sha(out/'training.pt'),'factual_manifest_sha256':sha(out/'factual_manifest.csv'),'time':stamp(),'cohort_amendment_sha256':sha(out/'cohort_amendment.json')})
print('Prepared disjoint source-specific cohorts; existing outputs preserved.')
