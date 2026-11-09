import unittest
import torch
from lift_ce.ffhq.style_editor import StyleEditor,apply_control,region_mask
class AffineLayer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.affine=torch.nn.Linear(3,3);self.in_channels=3
    def forward(self,x,w,noise_mode=None):return torch.sin(x+self.affine(w))
class Synthesis(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input=torch.nn.Linear(3,3);self.layer_names=['layer0','layer1'];self.layer0=AffineLayer();self.layer1=AffineLayer();self.output_scale=.25
class TestStyleEditor(unittest.TestCase):
    def test_checkpointed_style_gradients_and_zero_edit(self):
        class G:pass
        obj=G();obj.G=G();obj.G.synthesis=Synthesis();editor=StyleEditor(obj);w=torch.randn(2,3);s=editor.encode(w).detach()
        expected=obj.G.synthesis.layer1(obj.G.synthesis.layer0(obj.G.synthesis.input(w),w),w)*.25
        torch.testing.assert_close(editor.decode(w,s),expected)
        s.requires_grad_(True);actual=torch.autograd.grad(editor.decode(w,s).sum(),s)[0]
        direct=torch.sin(torch.sin(obj.G.synthesis.input(w)+s[:,:3])+s[:,3:])*.25
        torch.testing.assert_close(actual,torch.autograd.grad(direct.sum(),s)[0])
        self.assertFalse(obj.G.synthesis.layer0.affine._forward_hooks)
    def test_immutable_channels(self):
        s=torch.randn(2,20);d=torch.zeros(20);d[[2,8]]=.7;out=apply_control(s,d,3.)
        mask=d==0;self.assertTrue(torch.equal(s[:,mask],out[:,mask]));self.assertFalse(torch.equal(s,out));self.assertEqual(region_mask(224).shape,(1,1,224,224))
if __name__=='__main__':unittest.main()
