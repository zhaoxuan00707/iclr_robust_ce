## Completed-run interpretation and independent review

At nominal external severity 1, Base future validity is 75.00% at pixel L2 cost 5.5755.
GRACE beta 1.05 and 1.10 both achieve 100.00% at cost 6.6797.
Beta 1.20 achieves 99.13% at lower cost 6.1320.
The repository's paper-inspired SNS baseline also achieves 100% nominal validity,
at cost 6.7377. These results do not establish
uniform superiority over compatible baselines.

Mean Future Validity across the three equally weighted external shift families:

| method         |      0.0 |      0.5 |      1.0 |      1.5 |      2.0 |
|:---------------|---------:|---------:|---------:|---------:|---------:|
| Base           | 0.729167 | 0.6875   | 0.75     | 0.78125  | 0.765625 |
| GRACE_beta1.00 | 0.984375 | 0.980903 | 0.979167 | 0.954861 | 0.921875 |
| GRACE_beta1.05 | 1        | 1        | 1        | 0.984375 | 0.958333 |
| GRACE_beta1.10 | 1        | 1        | 1        | 0.984375 | 0.958333 |
| GRACE_beta1.20 | 1        | 0.994792 | 0.991319 | 0.987847 | 0.9375   |
| highconf       | 1        | 0.987847 | 0.947917 | 0.916667 | 0.909722 |
| sns_style      | 1        | 1        | 1        | 0.977431 | 0.96875  |
| trex_style     | 1        | 0.982639 | 0.939236 | 0.907986 | 0.90625  |

For each GRACE beta, this aggregate is nonincreasing across the predeclared
severity levels. Beta 1.05/1.10 falls from 100% to 95.83%
at severity 2. This supports gradual degradation on the tested aggregate curve,
not a guarantee for every image, shift family or untested severity. Base is
nonmonotonic. At severity 2, SNS achieves 96.88%, exceeding
GRACE beta 1.05/1.10 on this measured endpoint.

Geometry is feasible **within the predeclared numerical tolerance**
1e-6 × max(1,D), approximately 0.00005363.
Beta 1.00 exceeds its exact mathematical budget by 0.00005362, within that
tolerance. The other budgets have positive slack. The tight beta-1 boundary is
particularly difficult for the local solver; the reported candidate is not a
global optimum. Observed monotonic finite-bank risk follows nested candidate
pooling; future validity need not be monotonic in beta.

The reconstruction gap is material: original Base current validity is
100.00%, versus 12.50%
after encoding and decoding. Mean pixel L2 change from that round trip is
1.7128. Nominal calibration reconstruction
agreement is 99.10%.
This experiment measures the requested latent-geometry formulation, but does not
establish lossless realization of the Base population. No repair was added.

Runtime columns use **amortized compute time**, not end-to-end latency:
GRACE runtime is Base generation plus one quarter of total optimization time
across both rounds and all betas. All beta settings share the
134.975 seconds of attack-bank
construction. That shared cost should be counted once for the complete sweep,
not four times. Future training, preparation and reporting are separate jobs in
runtime.csv. Three deterministic translation/zero-shift copies are not three
independent statistical replicates; no confidence intervals across those copies
are claimed.

Independent review: all 68 jobs completed; all 24 attack checkpoint
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
