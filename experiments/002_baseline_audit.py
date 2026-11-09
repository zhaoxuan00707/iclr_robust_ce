"""Read-only audit of the preserved baseline for task 002; no model execution."""
import csv
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1')
OUT = REPO / 'results'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(name):
    with (RUN / 'heloc' / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    OUT.mkdir(exist_ok=True)
    completion = json.loads((RUN / 'execution_completion.json').read_text())
    checked = {}
    for name, expected in completion['artifact_hashes'].items():
        actual = digest(RUN / 'heloc' / name)
        assert actual == expected, f'Preserved artifact changed: {name}'
        checked[name] = actual
    assert completion['completed'] and not completion['missing_experiment_cells']
    instance = json.loads((RUN / 'heloc/instance_manifest.json').read_text())
    for name, expected in instance['files'].items():
        assert digest(RUN / 'heloc' / name) == expected, name
    config = json.loads((RUN / 'heloc/resolved_config.json').read_text())
    controls = rows('control_metrics.csv')
    assert len(controls) == 4
    assert all(float(r['current_invalid_mass']) == 0 for r in controls)
    assert all(float(r['slack']) >= -float(r['exact_tolerance']) for r in controls)
    risks = [float(r['true_bank_hinge']) for r in controls]
    assert all(b <= a + 1e-12 for a, b in zip(risks, risks[1:]))
    feasibility = rows('selected_output_feasibility.csv')
    geometry = rows('selected_output_geometry.csv')
    selected = []
    for row in rows('final_metrics.csv'):
        if row['operator'] != 'aligned_full_update':
            continue
        valid = next(r for r in feasibility if r['split'] == 'test' and r['method'] == row['method'])
        geo = next((r for r in geometry if r['split'] == 'test' and r['method'] == row['method']), None)
        selected.append({key: row[key] for key in [
            'method', 'lambda', 'cal_fresh_bank_hinge', 'mean_future_validity',
            'worst_future_validity', 'recourse_cost', 'current_validity']} | {
            'minimum_f0_score': valid['minimum_f0_score'],
            'invalid_support_count': valid['invalid_support_count'],
            'test_geometry_slack': geo['slack'] if geo else '',
            'deployment_status': valid['deployment_status'],
        })
    new = [r for r in selected if r['method'].startswith('S_beta')]
    assert len(new) == 4 and all(r['deployment_status'] == 'REJECTED' for r in new)
    assert all(float(r['test_geometry_slack']) < 0 for r in new if r['method'] != 'S_beta100')
    with (OUT / '002_baseline_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    endpoint = [r for r in rows('feasibility_checks.csv')
                if r['method'] == 'S_straight' and float(r['lambda']) == 0]
    bank_path = RUN / 'heloc/evaluation_bank.json'
    bank = json.loads(bank_path.read_text())['candidates']
    assert len(bank) == 5387 and len({r['candidate_id'] for r in bank}) == len(bank)
    assert all(r['split'] == 'train' for r in bank)
    assert all(r['full_update_seed'] == instance['full_update_seed'] for r in bank)
    assert all(r['measured_budget'] <= config['absolute_data_radius'] * (1 + 1e-6) for r in bank)
    task = REPO / 'tasks/002_baseline.md'
    text = task.read_text()
    sections = [int(n) for n in re.findall(r'^\s*(\d+)\.\s', text, re.MULTILINE)]
    audit = {
        'task': str(task), 'task_sha256': digest(task),
        'task_first_nonempty_line': next(line for line in text.splitlines() if line.strip()),
        'task_sections_present': sections,
        'task_sections_missing': [i for i in range(1, 21) if i not in sections],
        'baseline_run': str(RUN), 'baseline_artifact_hashes_verified': checked,
        'frozen_instance_hashes_verified': instance['files'],
        'recorded_bank_provenance_and_budgets_pass': True,
        'historical_model_replay_repeated': False,
        'seed': 1, 'fixed_radius_over_b_ref': config['fixed_radius_over_b_ref'],
        'D': config['D'], 'beta_geo': config['beta_geo'],
        'current_model': str(RUN / 'heloc/current.pt'),
        'previous_controls': str(RUN / 'heloc/controls_manifest.json'),
        'old_waypoint': str(RUN / 'heloc/frozen_waypoint.pt'),
        'available_expanded_bank': str(bank_path), 'bank_sha256': digest(bank_path),
        'available_bank_count': len(bank), 'bank_ids_unique': True,
        'training_control_hinges': risks, 'all_controls_feasible': True,
        'all_new_selected_test_outputs_rejected': True,
        'source_endpoint_training_box_violation_mass': {
            r['split']: float(r['training_range_violation_mass']) for r in endpoint},
        'new_solver_implemented': False, 'new_experiments_run': False,
        'raw_outputs_modified': False,
        'checks': 'Preserved artifact hashes, completed scope, unique bank IDs, nested control risks, exact control feasibility, selected-output failures',
    }
    (OUT / '002_baseline_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ['task_sections_missing', 'available_bank_count',
                                           'all_controls_feasible', 'all_new_selected_test_outputs_rejected']}, indent=2))


if __name__ == '__main__':
    main()
