"""Read-only per-factual Wachter-to-GRACE proximity audit of Task100 outputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/src'))
from lift_ce.counterfactuals.mixed_update_aware import MixedDomain


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=['heloc', 'fico', 'diabetes', 'german'], default=['heloc', 'fico'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new report directory; raw runs are never written.')
    torch.set_num_threads(1)
    original = pd.read_csv(ROOT/'results/exp3_table1_single_run/combined_summary.csv')
    summaries, cases, hashes = [], [], {}
    for dataset in args.datasets:
        for classifier in ['mlp', 'logistic']:
            worker = args.run/f'{dataset}_{classifier}'
            for name in ['prepared.pt', 'test_wachter.pt', 'test_grace_wachter.pt']:
                hashes[str(worker/name)] = sha(worker/name)
            prepared = load(worker/'prepared.pt')
            base = load(worker/'test_wachter.pt')['cf']
            result = load(worker/'test_grace_wachter.pt')
            source = prepared['sources']['test']
            assigned = result['cf']
            expected_n = int(original[(original.dataset == dataset) & (original.classifier == classifier) & (original.method == 'wachter')].iloc[0]['n'])
            assert len(source) == len(base) == len(assigned) == expected_n and expected_n > 0
            assert torch.equal(assigned, result['q'][result['assignment']])
            assert len(result['assignment'].unique()) == len(source)
            baseline = (base-source).norm(dim=1).numpy().astype(np.float64)
            variants = [('relaxed' if prepared['groups'] else 'native', assigned)]
            if prepared['groups']:
                domain = MixedDomain(prepared['data']['train_x'], prepared['groups'])
                variants.append(('hard', domain.project(assigned, True)))
            base_row = original[(original.dataset == dataset) & (original.classifier == classifier) & (original.method == 'wachter')].iloc[0]
            assert abs(baseline.mean()-base_row.Cost) < 1e-5
            for realization, cf in variants:
                after = (cf-source).norm(dim=1).numpy().astype(np.float64)
                delta = after-baseline
                row = original[(original.dataset == dataset) & (original.classifier == classifier) & (original.method == 'grace_wachter') & (original.realization == realization)].iloc[0]
                assert abs(after.mean()-row.Cost) < 1e-5
                assert abs(delta.mean()-(after.mean()-baseline.mean())) < 1e-12
                summary = dict(dataset=dataset, classifier=classifier, realization=realization,
                    n=len(delta), status=row.status, geometry_feasible=bool(row.geometry_feasible),
                    before_mean=baseline.mean(), after_mean=after.mean(),
                    delta_mean=delta.mean(), delta_median=np.median(delta),
                    delta_p95=np.quantile(delta, .95, method='linear'), delta_max=delta.max(),
                    delta_min=delta.min(), increased_count=int((delta>0).sum()),
                    decreased_count=int((delta<0).sum()), unchanged_count=int((delta==0).sum()),
                    before_median=np.median(baseline), after_median=np.median(after),
                    before_p95=np.quantile(baseline,.95,method='linear'),
                    after_p95=np.quantile(after,.95,method='linear'),
                    before_max=baseline.max(), after_max=after.max())
                summaries.append(summary)
                for i in range(len(delta)):
                    cases.append(dict(dataset=dataset, classifier=classifier, realization=realization,
                        factual_index=i, source_id=prepared['source_ids']['test'][i],
                        matched_support_index=int(result['assignment'][i]),
                        proximity_before=baseline[i], proximity_after=after[i],
                        proximity_change=delta[i], status=row.status))
    for path, h in hashes.items():
        assert sha(Path(path)) == h, path
    out = args.output
    out.mkdir(parents=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(cases).to_csv(out/'per_factual.csv', index=False)
    (out/'provenance.json').write_text(json.dumps(dict(raw_run=str(args.run.resolve()),
        raw_sha256=hashes, method='GRACE-Wachter', split='test',
        definition='delta_i = ||assigned_GRACE_i - factual_i||_2 - ||Wachter_Base_i - factual_i||_2',
        units='Original standardized-feature Euclidean L2',
        quantile='NumPy linear interpolation; quantiles of paired signed changes, not differences of marginal quantiles',
        checks='Cohort sizes match original saved results; permutation and assignment exact; mean costs match original report within1e-5; original files unchanged'), indent=2)+'\n')
    main = summary[summary.realization != 'hard']
    columns = ['dataset','classifier','n','before_mean','after_mean','delta_median','delta_p95','delta_max']
    table = main[columns].to_markdown(index=False, floatfmt='.4f')
    text = '''# Individual proximity changes after GRACE--Wachter refinement

For each original held-out factual (cohort sizes shown below), compute
`delta_i = ||GRACE_assigned_i - factual_i||_2 - ||Wachter_Base_i - factual_i||_2`.
This uses the saved minimum-cost one-to-one assignment and original standardized
feature-space L2 metric. Positive changes mean greater factual-to-CF distance;
negative changes mean improved proximity. Statistics include all cases, not
only positive changes. The95th percentile uses linear interpolation. These are
quantiles of paired changes, not differences between before/after quantiles.

'''+table+'''

The original calibrated settings are retained. German/FICO headline rows are relaxed
categorical outputs. Hard realizations are separate below; FICO/logistic hard
realization, if included, violates geometry and is not a feasible constrained result.

'''+summary[summary.realization == 'hard'][columns+['status']].to_markdown(index=False,floatfmt='.4f')+'''

`summary.csv` also contains the full before/after marginal summaries, signed
mean/minimum changes, and counts of increases/decreases. `per_factual.csv`
contains the matched source/support identities and every paired change.
Raw files are read-only; no training, optimization, or historical output was rerun.
'''
    (out/'report.md').write_text(text)
    latex = r'''We additionally report the distribution of individual proximity changes
following GRACE refinement. For each factual $x_i$, we compute
$\Delta_i=\|x_i^{\mathrm{GRACE}}-x_i\|_2-
\|x_i^{\mathrm{Base}}-x_i\|_2$
after the prescribed one-to-one assignment, using the original standardized
feature-space metric. We report the median, 95th percentile, and maximum
of these paired changes over all test factuals in each cohort. Positive values indicate
increased recourse cost. German and FICO results use relaxed categorical realizations.

'''+main[columns].to_latex(index=False,float_format='%.4f')
    (out/'table.tex').write_text(latex)
    print(table)
    print(f'VERIFIED: {len(args.datasets)*2} cohorts; {len(summary)} realizations; {len(hashes)} raw input hashes unchanged.')


if __name__ == '__main__':
    main()
