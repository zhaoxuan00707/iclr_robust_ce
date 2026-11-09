import sys,pickle,functools,os
from pathlib import Path
os.environ.setdefault("CUDA_HOME","/tmp/ffhq_cuda")
os.environ.setdefault("TORCH_EXTENSIONS_DIR","/tmp/ffhq_torch_extensions")
os.environ.setdefault("MAX_JOBS","2")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST","8.0")
os.environ["PATH"]="/tmp/ffhq_grace_deps/bin:/tmp/ffhq_cuda/bin:"+os.environ["PATH"]
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/vendor/ffhq'))
import clip
from denoising_diffusion_pytorch import KarrasUnet1D
from torch_utils.ops import filtered_lrelu,upfirdn2d,bias_act
# Published reference implementations avoid incompatible CUDA11.7 extension build
# against the current PyTorch CUDA12.1 runtime. Weights and operators unchanged.
for module,name in [(filtered_lrelu,'filtered_lrelu'),(upfirdn2d,'upfirdn2d'),(bias_act,'bias_act')]:
 fn=getattr(module,name)
 if os.environ.get('FFHQ_REFERENCE_OPS')=='1' and not isinstance(fn,functools.partial):setattr(module,name,functools.partial(fn,impl='ref'))
GAN=Path('/mnt/big_disk/z.cao/StyleGAN_pkl/stylegan3-r-ffhq-1024x1024.pkl')
FLOW=Path('/mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual/notebooks/checkpoints/ffhq_unet1d_argmax_w_20000_final.pt')
CLIP=Path('/tmp/ViT-B-32.pt')
class Generator(nn.Module):
 def __init__(self,device='cpu'):
  super().__init__()
  with GAN.open('rb') as f:self.G=pickle.load(f)['G_ema'].to(device).eval().requires_grad_(False)
 def decode(self,w):
  ws=w[:,None,:].repeat(1,self.G.num_ws,1)
  if not torch.is_grad_enabled() or not w.requires_grad:return self.G.synthesis(ws,noise_mode='const')
  # Same synthesis computation with activation checkpointing, one original layer
  # at a time, so 1024px latent gradients fit beside existing GPU workloads.
  from torch.utils.checkpoint import checkpoint
  synthesis=self.G.synthesis;styles=ws.float().unbind(1);x=synthesis.input(styles[0])
  for name,style in zip(synthesis.layer_names,styles[1:]):
   layer=getattr(synthesis,name)
   x=checkpoint(lambda a,b,layer=layer:layer(a,b,noise_mode='const'),x,style,use_reentrant=False)
  return (x*synthesis.output_scale).float()
 def mapping(self,z):return self.G.mapping(z,None)[:,0]
class SmileClassifier(nn.Module):
 def __init__(self,device='cpu'):
  super().__init__();model,preprocess=clip.load(str(CLIP),device='cpu',jit=False);model=model.float().to(device).eval().requires_grad_(False)
  self.visual=model.visual;self.preprocess=nn.Sequential() # actual transform kept callable
  self.transform=__import__('torchvision').transforms.Compose(preprocess.transforms[:2]+preprocess.transforms[4:])
  with torch.no_grad():text=model.encode_text(clip.tokenize(['smiling face','face']).to(device));self.register_buffer('text',F.normalize(text,dim=-1));self.register_buffer('scale',model.logit_scale.exp().detach());self.register_buffer('base_projection',self.visual.proj.detach().clone())
  self.visual.proj=None
 def canonical(self,image):
  x=(image.clamp(-1,1)+1)/2
  # Original CLIP resize/crop, without normalization, retained for image costs.
  import torchvision
  return torchvision.transforms.functional.center_crop(torchvision.transforms.functional.resize(x,224,interpolation=torchvision.transforms.InterpolationMode.BICUBIC,antialias=True),224)
 def features(self,canonical):
  mean=canonical.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None];std=canonical.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
  return self.visual((canonical-mean)/std)
 def logits_features(self,features,projection=None):
  projection=self.base_projection if projection is None else projection
  return self.scale*F.normalize(features@projection,dim=-1)@self.text.T
 def forward(self,image):return self.logits_features(self.features(self.canonical(image)))
class FaceFlow(KarrasUnet1D):
 def __init__(self):super().__init__(seq_len=512,num_classes=2,channels=1)
 def forward(self,t,x,y):return super().forward(x[:,None,:],t,class_labels=y).squeeze(1)
def load_flow(device):
 model=FaceFlow().to(device);model.load_state_dict(torch.load(FLOW,map_location='cpu',weights_only=True));return model.eval().requires_grad_(False)
