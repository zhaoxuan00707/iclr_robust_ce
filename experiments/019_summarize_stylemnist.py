"""Append evidence-based interpretation to a completed StyleMNIST report."""
import argparse
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    review_path=ROOT/'results/009_stylemnist_grace_v1/completion_review.json'
    review=json.loads(review_path.read_text());assert review['status']=='passed'
    main=pd.read_csv(out/'main_results.csv').set_index('method')
    severity=pd.read_csv(out/'shift_severity.csv').groupby(['method','severity']).Future_Validity.mean().unstack()
    severity.to_csv(out/'severity_mean_across_families.csv')
    geo=pd.read_csv(out/'geometry_feasibility.csv')
    numerical=json.loads((out/'representation_checks.json').read_text())
    marker='## Completed-run interpretation and independent review'
    text=f'''{marker}

At nominal external severity 1, Base future validity is {main.loc['Base','Future_Validity']:.2%} at pixel L2 cost {main.loc['Base','pixel_L2_cost']:.4f}.
GRACE beta 1.05 and 1.10 both achieve {main.loc['GRACE_beta1.05','Future_Validity']:.2%} at cost {main.loc['GRACE_beta1.05','pixel_L2_cost']:.4f}.
Beta 1.20 achieves {main.loc['GRACE_beta1.20','Future_Validity']:.2%} at lower cost {main.loc['GRACE_beta1.20','pixel_L2_cost']:.4f}.
The repository's paper-inspired SNS baseline also achieves 100% nominal validity,
at cost {main.loc['sns_style','pixel_L2_cost']:.4f}. These results do not establish
uniform superiority over compatible baselines.

Mean Future Validity across the three equally weighted external shift families:

{severity.to_markdown()}

For each GRACE beta, this aggregate is nonincreasing across the predeclared
severity levels. Beta 1.05/1.10 falls from 100% to {severity.loc['GRACE_beta1.05',2.0]:.2%}
at severity 2. This supports gradual degradation on the tested aggregate curve,
not a guarantee for every image, shift family or untested severity. Base is
nonmonotonic. At severity 2, SNS achieves {severity.loc['sns_style',2.0]:.2%}, exceeding
GRACE beta 1.05/1.10 on this measured endpoint.

Geometry is feasible **within the predeclared numerical tolerance**
1e-6 × max(1,D), approximately {1e-6*geo.D.iloc[0]:.8f}.
Beta 1.00 exceeds its exact mathematical budget by {-geo.slack.iloc[0]:.8f}, within that
tolerance. The other budgets have positive slack. The tight beta-1 boundary is
particularly difficult for the local solver; the reported candidate is not a
global optimum. Observed monotonic finite-bank risk follows nested candidate
pooling; future validity need not be monotonic in beta.

The reconstruction gap is material: original Base current validity is
{numerical['base_validity']:.2%}, versus {numerical['base_encode_decode_validity']:.2%}
after encoding and decoding. Mean pixel L2 change from that round trip is
{numerical['base_encode_decode_pixel_l2']:.4f}. Nominal calibration reconstruction
agreement is {numerical['calibration_reconstruction_decision_agreement']:.2%}.
This experiment measures the requested latent-geometry formulation, but does not
establish lossless realization of the Base population. No repair was added.

Runtime columns use **amortized compute time**, not end-to-end latency:
GRACE runtime is Base generation plus one quarter of total optimization time
across both rounds and all betas. All beta settings share the
{main.loc['GRACE_beta1.05','shared_attack_bank_seconds']:.3f} seconds of attack-bank
construction. That shared cost should be counted once for the complete sweep,
not four times. Future training, preparation and reporting are separate jobs in
runtime.csv. Three deterministic translation/zero-shift copies are not three
independent statistical replicates; no confidence intervals across those copies
are claimed.

Independent review: all {review['complete_jobs']} jobs completed; all 24 attack checkpoint
and perturbation hashes, exact geometry, image costs/risk populations, IDs,
45 future-evaluation groups, and pre-evaluation freeze ordering passed.
The initial strict decoder equality check failed because repeated GPU
ConvTranspose decoding is not bitwise deterministic in this run. Direct repeat
tests found differences below 1e-4 per pixel and mean differences below 1e-6.
The review records these errors rather than requiring bitwise equality; **zero
future-model decisions changed** on re-decoding all four GRACE supports against
all 45 future checkpoints. Evaluation uses the original frozen image bytes.
This remains a numerical reproducibility limitation. There were no failed
experiment jobs; the stricter first review failure and its resolution are
disclosed here. Existing historical outputs remain unchanged.
'''
    (out/'interpretation.md').write_text(text)
    report=(out/'report.md').read_text().split(marker)[0].rstrip()+'\n\n'+text
    (out/'report.md').write_text(report)
    result=ROOT/'results/009_stylemnist_grace_v1'
    for filename in ['report.md','interpretation.md','severity_mean_across_families.csv']:
        shutil.copyfile(out/filename,result/filename)
    shutil.copyfile(review_path,out/'completion_review.json')

if __name__=='__main__':main()
