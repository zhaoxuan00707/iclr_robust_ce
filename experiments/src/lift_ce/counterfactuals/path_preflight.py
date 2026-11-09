"""Necessary feasibility checks before optimizing a fixed-endpoint CF path."""
import math


def check_path_preconditions(anchor_distance, budget, source_invalid_mass,
                             target_invalid_mass, source_outside_box_mass=0.,
                             target_outside_box_mass=0., *, full_path_box=False,
                             geometry_tolerance=1e-6, mass_tolerance=0.):
    """Return necessary conditions only; passing does not prove a valid path exists.

    The budget is absolute and is never enlarged to match a new batch. A box on
    learned control supports is distinct from a box on every path point.
    """
    values = [anchor_distance, budget, geometry_tolerance, mass_tolerance,
              source_invalid_mass, target_invalid_mass,
              source_outside_box_mass, target_outside_box_mass]
    if not all(math.isfinite(float(v)) and v >= 0 for v in values):
        raise ValueError('finite nonnegative distances, budgets, masses and tolerances required')
    if any(v > 1 for v in values[3:]):
        raise ValueError('masses and mass tolerance must lie in [0,1]')
    failures = []
    if anchor_distance > budget + geometry_tolerance:
        failures.append('anchor_distance_exceeds_absolute_budget')
    if max(source_invalid_mass, target_invalid_mass) > mass_tolerance:
        failures.append('fixed_endpoint_current_invalid')
    if full_path_box and max(source_outside_box_mass, target_outside_box_mass) > mass_tolerance:
        failures.append('fixed_endpoint_outside_required_path_box')
    return {'necessary_conditions_pass': not failures, 'failures': failures,
            'anchor_distance': float(anchor_distance), 'B_geo': float(budget),
            'maximum_possible_geometry_slack': float(budget-anchor_distance),
            'continuous_path_feasibility_established': False}
