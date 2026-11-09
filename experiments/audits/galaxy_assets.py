"""Small original-checkpoint compatibility audit, with no training or old writes."""
import json,sys,hashlib
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),'/tmp/galaxy_grace_deps']
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from lift_ce.galaxy.assets import *

def main():
    torch.set_num_threads(1);torch.manual_seed(20260916);torch.backends.cudnn.deterministic=True
    m=load_classifier('cuda:0');ae=load_autoencoder('cuda:0');flow=load_flow('cuda:0')
    tf=transforms.Compose([transforms.ToTensor(),transforms.CenterCrop(150),transforms.Resize(128,antialias=True)])
    tr=ImageFolder(str(TRAIN),transform=tf);alltr=ImageFolder(str(ALL_TRAIN),transform=tf);te=ImageFolder(str(TEST),transform=tf)
    known={Path(p).name for p,_ in tr.samples};cal=[i for i,(p,y) in enumerate(alltr.samples) if Path(p).name not in known]
    ids=torch.randperm(len(cal),generator=torch.Generator().manual_seed(20260916))[:64].tolist();ids=[cal[i] for i in ids]
    x=torch.stack([alltr[i][0] for i in ids]);y=torch.tensor([alltr[i][1] for i in ids]);outs=[];recs=[];recon=[]
    with torch.no_grad():
        for start in range(0,len(x),4):
            q=x[start:start+4].cuda();r=ae.decode(ae.encode(q));outs.append(m(q).cpu());recs.append(m(r).cpu());recon.append(r.cpu())
    recon=torch.cat(recon);pred=torch.cat(outs).argmax(1);pr=torch.cat(recs).argmax(1)
    from lift_ce.galaxy.flow import progressive_leap
    import torch.nn.functional as F
    with torch.no_grad():
        z=ae.encode(x[:2].cuda());target=F.one_hot((y[:2].cuda()+1)%10,10).float()
        q=progressive_leap(flow,z,lambda h:F.one_hot(m(ae.decode(h)).argmax(1),10).float(),target,gamma_lift=.1,gamma_land=.1,max_itr=2,n_steps=15,mute=True)[-1]
        assert q.shape==(2,64) and torch.isfinite(q).all()
    result={'classifier':str(CLASSIFIER),'autoencoder':str(AUTOENCODER),'flow':str(FLOW),'strict_load_all':True,'classifier_parameters':sum(p.numel() for p in m.parameters()),'latent_dim':64,
      'class_to_idx':tr.class_to_idx,'train_size':len(tr),'full_train_size':len(alltr),'calibration_pool_excluding_base_train':len(cal),'test_size':len(te),'audit_n':len(x),'audit_ids':[alltr.samples[i][0] for i in ids],
      'classifier_accuracy':float((pred==y).float().mean()),'reconstruction_mse':float((recon-x).square().mean()),'reconstruction_decision_agreement':float((pred==pr).float().mean()),
      'original_flow_two_iteration_smoke':True,'cuda_device':torch.cuda.get_device_name(),'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9}
    dest=Path(__file__).resolve().parents[2]/'results/010_galaxy_asset_checks.json';dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
