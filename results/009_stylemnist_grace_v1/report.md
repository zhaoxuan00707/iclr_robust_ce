# MNIST 4→9 with style shifts: GRACE v1



Completed new run; original checkpoints and historical outputs preserved. User confirmed that StyleMNIST denotes existing MNIST with style shifts.



## Assets and task

Frozen SmallCNN and StrongConvAutoencoder(64): `/mnt/big_disk/xu.zhao/robust_ce_results/mnist_roar_latent_calibrated/payload.pt`. 4 and 9 denote digit labels, not classifier IDs. 64 fixed saved factual digit-4 test images; 64 seed-fixed digit-9 training references; 5000 training images. Exact MNIST IDs and hashes are saved.



## Main result

Nominal severity 1; mean over three shift families and three realizations. Cost is factual-to-returned-image pixel L2. Runtime excludes shared bank construction; the shared bank time is reported separately.

| method         |   Phi_hat_r |   pixel_L2_cost |   realized_latent_L2_cost |   Future_Validity |   worst_future_validity |   runtime_seconds |   shared_attack_bank_seconds |
|:---------------|------------:|----------------:|--------------------------:|------------------:|------------------------:|------------------:|-----------------------------:|
| Base           |   0.31313   |         5.57552 |                   42.7755 |          0.75     |                0.640625 |          0.55841  |                        0     |
| highconf       |   0.0490278 |         5.79627 |                   45.341  |          0.947917 |                0.890625 |          0.167115 |                        0     |
| trex_style     |   0.0663217 |         5.80283 |                   45.3102 |          0.939236 |                0.890625 |          0.854933 |                        0     |
| sns_style      |   0         |         6.73772 |                   56.7008 |          1        |                1        |          0.501847 |                        0     |
| GRACE_beta1.00 |   0.0177071 |         7.63232 |                   71.5353 |          0.979167 |                0.96875  |         32.3883   |                      134.975 |
| GRACE_beta1.05 |   0         |         6.67965 |                   55.1561 |          1        |                1        |         32.3883   |                      134.975 |
| GRACE_beta1.10 |   0         |         6.67965 |                   55.1561 |          1        |                1        |         32.3883   |                      134.975 |
| GRACE_beta1.20 |   0         |         6.13198 |                   48.431  |          0.991319 |                0.96875  |         32.3883   |                      134.975 |



## Geometry budget ablation

| method         |   beta_geo |       D |   B_geo |   total_length |        slack | feasible   |   source_candidate |   Phi_hat_r |   pixel_L2_cost |   realized_latent_L2_cost |   Future_Validity |   worst_future_validity |   runtime_seconds |   shared_attack_bank_seconds |
|:---------------|-----------:|--------:|--------:|---------------:|-------------:|:-----------|-------------------:|------------:|----------------:|--------------------------:|------------------:|------------------------:|------------------:|-----------------------------:|
| GRACE_beta1.00 |       1    | 53.6267 | 53.6267 |        53.6268 | -5.36242e-05 | True       |                  3 |   0.0177071 |         7.63232 |                   71.5353 |          0.979167 |                 0.96875 |           32.3883 |                      134.975 |
| GRACE_beta1.05 |       1.05 | 53.6267 | 56.3081 |        53.8109 |  2.49717     | True       |                111 |   0         |         6.67965 |                   55.1561 |          1        |                 1       |           32.3883 |                      134.975 |
| GRACE_beta1.10 |       1.1  | 53.6267 | 58.9894 |        53.8109 |  5.1785      | True       |                111 |   0         |         6.67965 |                   55.1561 |          1        |                 1       |           32.3883 |                      134.975 |
| GRACE_beta1.20 |       1.2  | 53.6267 | 64.3521 |        59.8166 |  4.53543     | True       |                327 |   0         |         6.13198 |                   48.431  |          0.991319 |                 0.96875 |           32.3883 |                      134.975 |

Observed nonincreasing shared-bank risk across beta: True. Nested candidate pooling enforces the ordering for the retained candidate set; this is not a global-optimization guarantee.



## Geometry and realization

Optimization and exact unregularized W2 geometry use 64-dimensional encoder coordinates. Classifier evaluation uses Dec(h). Base uses original generator images, so Dec(Enc(Base)) can differ; representation_checks.json quantifies this. Pixel L2 is primary recourse cost; realized encode/decode latent L2 is separate. Raw solver receives no current classifier and no current-validity repair is run.



## Shift protocol

Historical rotation, translation and noise operators are reused. NEW extensions: external severity multipliers and fixed full updates from f0 replace historical scratch fitting. Future models train on shifted data; returned CF images remain fixed. Rotation/noise realizations vary transformation seeds; translation and zero-shift realizations duplicate the same deterministic distribution and are not independent replicates. This is one frozen classifier and factual set, not a multi-seed benchmark.



## Compute

One server, four assigned GPUs 1,2,3,5. Sum of worker elapsed GPU occupancy: 0.1064 GPU-hours (wall time per GPU job, not hardware utilization integral). Shared attack construction: 134.97 seconds. All optimization jobs: 127.32 seconds.



## Failures and limitations

Final bank contains 24 persisted full-update models; 372 retained Q candidates were cross-evaluated. Geometry feasibility is recorded independently. See logs for any interrupted/failed attempts; complete.json exists only for successful jobs.

Finite bank and local solver; truncated minibatch hypergradients with frozen BN during search differ from complete-update gradients. Accepted models use all training examples, all parameters, eight epochs and fixed AdamW protocol. Pixel perturbations are constrained by paired transport and verified by exact label-preserving OT. No final future outcomes configure GRACE. Existing T-Rex/SNS baselines are explicitly paper-inspired implementations, not canonical reproductions. No autoencoder or base classifier retraining. Broader class/seed coverage and semantic human evaluation remain limitations.



Raw run: `/home/xu.zhao/robust_ce_results/stylemnist_grace_v1/20260916_full_v1`. Configuration, immutable experiment manifest, bank and job provenance are saved alongside CSVs and PDF figures.

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
