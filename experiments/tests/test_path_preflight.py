import unittest
from lift_ce.counterfactuals.path_preflight import check_path_preconditions


class PathPreflightTests(unittest.TestCase):
    def test_cannot_rescale_budget_for_new_batch(self):
        result = check_path_preconditions(5.1, 5., 0., 0.)
        self.assertEqual(result['failures'], ['anchor_distance_exceeds_absolute_budget'])
        self.assertEqual(result['B_geo'], 5.)

    def test_endpoint_box_conflict_depends_on_declared_scope(self):
        self.assertTrue(check_path_preconditions(1., 1.2, 0., 0., .8)['necessary_conditions_pass'])
        self.assertFalse(check_path_preconditions(1., 1.2, 0., 0., .8, full_path_box=True)['necessary_conditions_pass'])

    def test_endpoint_validity_is_never_just_diagnostic(self):
        self.assertFalse(check_path_preconditions(1., 1.2, .001, 0.)['necessary_conditions_pass'])

    def test_passing_is_not_a_continuous_certificate(self):
        result = check_path_preconditions(1., 1., 0., 0.)
        self.assertTrue(result['necessary_conditions_pass'])
        self.assertFalse(result['continuous_path_feasibility_established'])
