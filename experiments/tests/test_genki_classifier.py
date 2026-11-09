import tempfile
from pathlib import Path
import unittest
import pandas as pd
import torch
from torch import nn
from lift_ce.ffhq.genki_classifier import GenkiSmileClassifier, CLASS_ORDER
from lift_ce.experiments.genki_ffhq_classifier_v1 import split_groups


class TestGenkiClassifier(unittest.TestCase):
    def test_exact_duplicates_never_cross_split(self):
        rows=[dict(index=i,smile=i%2,pixel_sha256=str(i)) for i in range(100)]
        rows+=rows[:20]
        d=split_groups(pd.DataFrame(rows),123)
        self.assertEqual(d.groupby('pixel_sha256').split.nunique().max(),1)
        self.assertEqual(set(d.split),{'train','validation','test'})
        self.assertTrue(d.equals(split_groups(pd.DataFrame(rows),123)))

    def test_conflicting_duplicate_labels_rejected(self):
        with self.assertRaises(AssertionError):
            split_groups(pd.DataFrame([dict(smile=0,pixel_sha256='same'),dict(smile=1,pixel_sha256='same')]),1)

    def test_adapter_preserves_input_gradients_and_class_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'head.pt';head=nn.Linear(3,2)
            data=dict(class_order=CLASS_ORDER,feature_dimension=3,head_state=head.state_dict())
            torch.save(data,p)
            visual=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten())
            clf=GenkiSmileClassifier(p,visual=visual)
            x=torch.rand(1,3,32,32,requires_grad=True)
            grad,=torch.autograd.grad(clf(x).softmax(1)[0,0],x)
            self.assertGreater(float(grad.norm()),0)
            self.assertTrue(all(not p.requires_grad for p in clf.parameters()))
            data['class_order']=list(reversed(CLASS_ORDER));torch.save(data,p)
            with self.assertRaises(ValueError):GenkiSmileClassifier(p,visual=visual)


if __name__=='__main__':unittest.main()
