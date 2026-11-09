"""One actual VGG hypergradient smoke; no models trained or outputs overwritten."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from lift_ce.galaxy.assets import load_classifier,TRAIN
from lift_ce.experiments.galaxy_grace_v1 import surrogate_scores

torch.set_num_threads(1);torch.manual_seed(7);torch.cuda.set_per_process_memory_fraction(.16)
m=load_classifier('cuda:0').eval();ds=ImageFolder(str(TRAIN),transform=transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(150),transforms.Resize(128)]))
x=torch.stack([ds[i*400][0] for i in range(4)]).cuda();y=torch.tensor([ds[i*400][1] for i in range(4)],device='cuda');d=torch.zeros_like(x,requires_grad=True)
p=surrogate_scores(m,x,y,d,x,(y+1)%10,lr=.0001);g=torch.autograd.grad(torch.nn.functional.softplus(10*(.5-p)).mean()/10,d)[0]
assert torch.isfinite(g).all()
r={'gradient_norm':float(g.norm()),'finite':True,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9}
(Path(__file__).resolve().parents[2]/'results/010_galaxy_gradient_smoke.json').write_text(json.dumps(r,indent=2));print(r)
