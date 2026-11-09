"""Strict adapters for original LeapFactual Galaxy assets; no retraining/downloads."""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from .wae import WAE_MMD

ASSET_ROOT=Path('/mnt/mass_disk1/z.cao/workspace/Submitted/LeapFactual')
CLASSIFIER=ASSET_ROOT/'logs/Galaxy10_VGG_20/version_0/checkpoints/epoch=47-step=624.ckpt'
AUTOENCODER=ASSET_ROOT/'logs/Galaxy10_VAE/version_8/checkpoints/last.ckpt'
FLOW=ASSET_ROOT/'notebooks/checkpoints/galaxy10_flow_weights_mlp_finetune.pt'
TRAIN=Path('/mnt/big_disk/z.cao/Galaxy10/image_data_train_20')
ALL_TRAIN=Path('/mnt/big_disk/z.cao/Galaxy10/image_data_train')
TEST=Path('/mnt/big_disk/z.cao/Galaxy10/image_data_test')
CLASS_NAMES=['Disturbed Galaxies','Merging Galaxies','Round Smooth Galaxies','In-between Round Smooth Galaxies','Cigar Shaped Smooth Galaxies','Barred Spiral Galaxies','Unbarred Tight Spiral Galaxies','Unbarred Loose Spiral Galaxies','Edge-on Galaxies without Bulge','Edge-on Galaxies with Bulge']

class GalaxyVGG(nn.Module):
    def __init__(self):
        super().__init__();self.net=vgg16_bn(weights=None);self.net.classifier[-1]=nn.Linear(4096,10)
        self.transform=VGG16_BN_Weights.IMAGENET1K_V1.transforms()
    def forward(self,x):return self.net(self.transform(x))


def load_classifier(device='cpu',checkpoint=None):
    model=GalaxyVGG()
    if checkpoint is None:
        state=torch.load(CLASSIFIER,map_location='cpu',weights_only=False,mmap=True)['state_dict']
        state={k.removeprefix('model.'):v for k,v in state.items()}
    else:state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def load_autoencoder(device='cpu'):
    model=WAE_MMD(in_channels=3,total_latent_dim=64,num_classes=0,img_size=128,hidden_dims=[16,32,64,128,256],reg_weight=10,kernel_type='imq',latent_var=2.)
    raw=torch.load(AUTOENCODER,map_location='cpu',weights_only=False,mmap=True)['state_dict']
    state={k.removeprefix('model.'):v for k,v in raw.items() if k.removeprefix('model.').split('.')[0] in ['encoder','fc_in','fc_out','decoder','output']}
    model.load_state_dict(state,strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_flow(device='cpu'):
    from .flow import MLP_CFM
    model=MLP_CFM(dim=64,time_varying=True,cond=True,cond_dim=10,w=256)
    model.load_state_dict(torch.load(FLOW,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval().requires_grad_(False)
