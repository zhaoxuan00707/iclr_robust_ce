"""Audited LeNNon CelebA smile model, diagnostic only.
Architecture from AiresPucrs/LeNNon-Smile-Detector, revision
 e34d94965363927f60dea5a5b76770a8966048c5, Apache-2.0.
The released checkpoint retains ten logits. Upstream interprets argmax>0 as
smiling; preserve that mapping and expose extra-class predictions explicitly.
"""
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

class LeNNon(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(3,16,3,1,1);self.pool=nn.MaxPool2d(2,2);self.conv2=nn.Conv2d(16,32,3,1,1);self.fc1=nn.Linear(32*25*25,128);self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.conv1(x)));x=self.pool(torch.relu(self.conv2(x)));x=x.reshape(-1,32*25*25);return self.fc2(torch.relu(self.fc1(x)))

class SupervisedSmile:
    def __init__(self,checkpoint):
        self.model=LeNNon().eval().requires_grad_(False);self.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        self.transform=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
    def score(self,paths):
        x=torch.stack([self.transform(Image.open(p).convert('RGB')) for p in paths])
        with torch.no_grad():p=self.model(x).softmax(1);label=p.argmax(1)
        return [{'supervised_smile_probability':float(1-prob[0]),'supervised_smile_label':int(lab>0),'supervised_raw_class':int(lab),'extra_class_mass':float(prob[2:].sum())} for prob,lab in zip(p,label)]
