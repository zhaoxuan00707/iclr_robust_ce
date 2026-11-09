"""Check completed Task100 reports without rerunning or modifying raw experiments."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads((args.run / 'jobs.json').read_text())
    assert len(jobs) == 8 and all(j['exitcode'] == 0 for j in jobs)
    expected = {f'{d}_{m}' for d in ['heloc', 'diabetes', 'german', 'fico']
                for m in ['mlp', 'logistic']}
    completed = {p.parent.name for p in args.run.glob('*/complete.json')}
    assert completed == expected, completed
    manifest = json.loads((args.recovery / 'completed_raw_hashes.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((args.run / name).read_bytes()).hexdigest() == digest, name
    required = {
        'exp1_mnist_T_mismatch': ['raw.csv', 'summary.csv', 'fv_vs_Teval.pdf',
                                'train_loss_vs_Teval.pdf', 'parameter_displacement_vs_Teval.pdf', 'summary.md'],
        'exp2_mnist_no_geometry': ['metrics.csv', 'summary.md'],
        'exp3_table1_single_run': ['current_mlp.csv', 'logistic_regression.csv',
                                  'dice_base_generator.csv', 'combined_summary.csv', 'summary.md',
                                  'independent_checks.csv', 'hard_category_shift_summary.csv'],
        'exp4_runtime_10cf': ['raw.csv', 'summary.csv', 'runtime_table.tex', 'summary.md'],
    }
    report_hashes = {}
    for folder, files in required.items():
        for name in files:
            path = ROOT / 'results' / folder / name
            assert path.is_file() and path.stat().st_size > 0, path
            report_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert any((ROOT / 'results/exp2_mnist_no_geometry/examples').iterdir())
    checks = pd.read_csv(ROOT / 'results/exp3_table1_single_run/independent_checks.csv')
    assert checks.passed.eq(True).all() and set(checks.worker) == expected
    table = pd.read_csv(ROOT / 'results/exp3_table1_single_run/combined_summary.csv')
    assert set(table.dataset + '_' + table.classifier) == expected
    timing = pd.read_csv(ROOT / 'results/exp4_runtime_10cf/summary.csv')
    assert set(timing.dataset) == {'heloc', 'diabetes', 'german', 'fico'}
    assert timing.n.eq(10).all()
    for _, group in timing.groupby('dataset'):
        assert set(group.method) == {'wachter', 'trex', 'roar', 'proplace', 'ellice', 'grace_wachter'}
    result = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  run=str(args.run), complete=True, workers=8,
                  independent_checks=len(checks), preserved_raw_files=len(manifest),
                  table_rows=len(table), runtime_rows=len(timing), report_sha256=report_hashes)
    with (args.recovery / 'final_completeness_check.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'report_sha256'}, indent=2))


if __name__ == '__main__':
    main()
