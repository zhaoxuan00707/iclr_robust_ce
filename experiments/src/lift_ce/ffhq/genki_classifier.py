"""Differentiable GENKI-supervised smile head over the frozen FFHQ CLIP encoder."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF, InterpolationMode

CLASS_ORDER = ['smiling', 'not_smiling']


def canonical(image):
    """Native generator RGB [-1,1] to the original differentiable CLIP crop."""
    x = (image.clamp(-1, 1) + 1) / 2
    return TF.center_crop(TF.resize(x, 224, interpolation=InterpolationMode.BICUBIC,
                                    antialias=True), 224)


def load_visual(checkpoint, device):
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'experiments/vendor/ffhq'))
    import clip
    model, _ = clip.load(str(checkpoint), device='cpu', jit=False)
    return model.visual.float().to(device).eval().requires_grad_(False)


def image_features(visual, x):
    mean = x.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = x.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    return F.normalize(visual((x-mean)/std), dim=-1)


class GenkiSmileClassifier(nn.Module):
    """Output index 0 is smiling, matching FFHQ; no text prompts are evaluated."""
    def __init__(self, checkpoint, device='cpu', visual=None):
        super().__init__()
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if data['class_order'] != CLASS_ORDER:
            raise ValueError('Checkpoint class order differs from FFHQ smile index 0')
        self.visual = load_visual(data['clip_checkpoint'], device) if visual is None else visual.to(device).eval().requires_grad_(False)
        self.head = nn.Linear(data['feature_dimension'], 2).to(device)
        self.head.load_state_dict(data['head_state'], strict=True)
        self.eval().requires_grad_(False)

    canonical = staticmethod(canonical)

    def features(self, x):
        return image_features(self.visual, x)

    def logits_features(self, features):
        return self.head(features)

    def forward(self, native_image):
        return self.logits_features(self.features(self.canonical(native_image)))
