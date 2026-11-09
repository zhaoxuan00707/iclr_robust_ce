import sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/src'))
import torch
from lift_ce.ffhq.assets import Generator,SmileClassifier,load_flow,GAN,FLOW,CLIP
from lift_ce.experiments.galaxy_grace_v1 import sha
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 torch.set_num_threads(1);torch.manual_seed(56789);torch.cuda.set_per_process_memory_fraction(.16)
 assert sha(CLIP)=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
 g=Generator('cuda');c=SmileClassifier('cuda');print('generator/classifier loaded',flush=True)
 with torch.no_grad():w=g.mapping(torch.randn(1,512,device='cuda'))
 w=w.detach().requires_grad_(True);start=time.time();x=g.decode(w);logits=c(x);grad=torch.autograd.grad(logits.softmax(1)[0,0],w)[0];print('gradient',grad.norm().item(),'seconds',time.time()-start,flush=True)
 assert torch.isfinite(grad).all() and grad.norm()>0
 with torch.no_grad():reference=g.G.synthesis(w.detach()[:,None,:].repeat(1,16,1),noise_mode='const')
 torch.testing.assert_close(x.detach(),reference,rtol=0,atol=0)
 del reference
 plt.imsave(ROOT/'results/013_ffhq_audit_face.png',((x[0].detach().cpu().permute(1,2,0).clamp(-1,1)+1)/2).numpy())
 del x;torch.cuda.empty_cache();flow=load_flow('cuda');y=flow(torch.zeros(1,device='cuda'),w.detach(),torch.zeros(1,dtype=torch.long,device='cuda'));assert y.shape==(1,512) and torch.isfinite(y).all()
 report={'generator_shape':[1,3,1024,1024],'latent_dim':512,'num_ws':g.G.num_ws,'classifier_prompts':['smiling face','face'],'smile_probability':logits.softmax(1)[0,0].item(),'latent_gradient_norm':grad.norm().item(),'flow_parameters':sum(p.numel() for p in flow.parameters()),'gradient_seconds':time.time()-start,'peak_memory_GB':torch.cuda.max_memory_allocated()/1e9,'hashes':{str(p):sha(p) for p in [GAN,FLOW,CLIP]},'operator_implementation':'original fused StyleGAN CUDA operators; activation checkpointing','classifier_dtype':'float32'}
 (ROOT/'results/013_ffhq_asset_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':main()
