import unittest
import torch
from lift_ce.experiments.genki_grace_identity_v1 import population_risk,state_scores
from lift_ce.experiments.galaxy_grace_v1 import project_delta,paired_distance
from lift_ce.counterfactuals.geodesic_excess import exact_w2


class TestGenkiGrace(unittest.TestCase):
    def test_population_reduction_is_max_of_means(self):
        p=torch.tensor([[.2,.8],[.8,.2]])
        self.assertAlmostEqual(float(population_risk(p)),.15,places=6)
        self.assertAlmostEqual(float(torch.relu(.5-p).max(0).values.mean()),.3,places=6)

    def test_head_bank_smile_class_orientation_and_gradient(self):
        x=torch.tensor([[2.,1.]],requires_grad=True)
        states=[{'weight':torch.tensor([[1.,0.],[0.,1.]]),'bias':torch.tensor([.2,-.2])}]
        score=state_scores(x,states);self.assertAlmostEqual(float(score),1.4,places=6)
        grad,=torch.autograd.grad(score.sum(),x);torch.testing.assert_close(grad,torch.tensor([[1.,-1.]]))

    def test_paired_pixel_certificate_respects_full_training_mass(self):
        x=torch.rand(4,3,8,8);delta=project_delta(x,torch.ones_like(x),.1,100)
        self.assertLessEqual(float(paired_distance(delta,100)),.1000001)
        self.assertTrue(((x+delta>=0)&(x+delta<=1)).all())

    def test_exact_geometry_zero_detour_for_matched_interpolation(self):
        z=torch.tensor([[0.,0.],[2.,0.]]);target=z+torch.tensor([0.,1.]);q=(z+target)/2
        D=exact_w2(z,target).distance
        self.assertAlmostEqual(exact_w2(z,q).distance+exact_w2(q,target).distance,D,places=6)


if __name__=='__main__':unittest.main()
