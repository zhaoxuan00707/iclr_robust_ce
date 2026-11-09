import unittest
import torch
from lift_ce.experiments.ffhq_smile_editor_v2 import passes_quality,gate_table
import pandas as pd
class TestExtension(unittest.TestCase):
    def test_preservation_gates_are_joint_and_fail_on_missing(self):
        c=dict(arcface_min=.8,facenet_min=.8,lpips_max=.15,outside_rms_max=.04)
        r=dict(arcface_similarity=.8,facenet_similarity=.8,LPIPS=.15,outside_rms=.04)
        self.assertTrue(passes_quality(r,c))
        for name,bad in [('arcface_similarity',.799),('facenet_similarity',float('nan')),('LPIPS',.151),('outside_rms',.041)]:self.assertFalse(passes_quality({**r,name:bad},c))
    def test_population_gate_counts_people_not_candidates(self):
        c=dict(arcface_min=.8,facenet_min=.8,lpips_max=.15,outside_rms_max=.04,smile_score_min=.5,smile_score_gain_min=.05)
        row=dict(editor='style4',calibration_index=0,arcface_similarity=.9,facenet_similarity=.9,LPIPS=.1,outside_rms=.01,smile_score=.6,smile_gain=.2)
        _,counts=gate_table(pd.DataFrame([row]*8),c);self.assertEqual(int(counts['style4']),1)
if __name__=='__main__':unittest.main()
