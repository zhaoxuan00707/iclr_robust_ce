"""StyleGAN3 post-affine style controls; untouched channels remain factual."""
import torch
from torch.utils.checkpoint import checkpoint


def region_mask(size,device='cpu'):
    """Fixed FFHQ lower-face ellipse; a coarse mouth/cheek proxy, not parsing."""
    y,x=torch.meshgrid(torch.linspace(0,1,size,device=device),torch.linspace(0,1,size,device=device),indexing='ij')
    return (((x-.5)/.27).square()+((y-.70)/.20).square()<=1).float()[None,None]


class StyleEditor:
    def __init__(self,generator):
        self.g=generator
        self.synthesis=generator.G.synthesis
        self.names=list(self.synthesis.layer_names)
        self.sizes=[getattr(self.synthesis,n).in_channels for n in self.names]
        self.offsets=[0]
        for n in self.sizes:self.offsets.append(self.offsets[-1]+n)

    def encode(self,w):
        return torch.cat([getattr(self.synthesis,name).affine(w) for name in self.names],dim=1)

    def decode(self,w,styles):
        # The input Fourier transform stays factual; only selected post-affine
        # modulation channels can change. Hooks exist only inside each layer call.
        x=self.synthesis.input(w)
        for j,name in enumerate(self.names):
            layer=getattr(self.synthesis,name);style=styles[:,self.offsets[j]:self.offsets[j+1]]
            def forward(value,code,modulation,layer=layer):
                handle=layer.affine.register_forward_hook(lambda module,args,output:modulation)
                try:return layer(value,code,noise_mode='const')
                finally:handle.remove()
            if torch.is_grad_enabled() and styles.requires_grad:
                x=checkpoint(forward,x,w,style,use_reentrant=False)
            else:x=forward(x,w,style)
        return (x*self.synthesis.output_scale).float()


def apply_control(styles,direction,alpha):
    return styles+alpha*direction
