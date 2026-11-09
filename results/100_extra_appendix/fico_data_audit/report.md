# FICO mixed-feature data downloaded and verified

Downloaded the original HELOC challenge CSV and original FICO data dictionary
from the challenge participant repository 5teffen/FICO-xML-Challenge, pinned at
056ca9505910db184c10d20457551ab48925e317. The historical FICO challenge URL now
redirects to an unrelated FICO news page; the files were obtained from a mirror,
not directly from an active provider download endpoint.

Source: https://github.com/5teffen/FICO-xML-Challenge/tree/056ca9505910db184c10d20457551ab48925e317/xML%20Challenge%20Dataset%20and%20Data%20Dictionary

External files:
- /home/xu.zhao/robust_ce_data/fico_mixed_task100_v1/heloc_dataset_v1.csv
- /home/xu.zhao/robust_ce_data/fico_mixed_task100_v1/heloc_data_dictionary-2.xlsx
- /home/xu.zhao/robust_ce_data/fico_mixed_task100_v1/provenance.json

## Corrected interpretation
The dictionary explicitly describes categories for MaxDelq2PublicRecLast12M and
MaxDelqEver. These are integer-encoded delinquency statuses, with unknown/other
categories as well as delinquency durations. Treating their numeric codes as
continuous scalar distances loses their categorical semantics. The earlier audit
correctly described the old code's behavior but missed this source-data meaning.
This resolves the data-schema obstacle: a mixed categorical FICO representation
can be constructed without arbitrary binning. It is a second representation of
the SAME underlying FICO/HELOC corpus, not a fourth independent dataset.

## Reproducible configuration and checks
`experiments/configs/fico_mixed_task100.json` marks these two predictors as
categorical and leaves21 predictors numerical, positive class Good. It reuses
`load_tabular_three_way` with seed0 and60/20/20 split. Numeric missing entries
are median-imputed; categorical missing entries are mode-imputed; both fitted
on training only. Special codes-7/-8/-9 follow the inherited missing-value rule.
No rows are silently dropped:588 rows have all predictors-9 and are retained.
They are not ordinary observed credit profiles; any later exclusion would require
an explicit new preprocessing choice applied consistently across representations.

Raw shape10459x24, with23 predictors plus target. Split sizes6275/2092/2092.
The resulting representation has37 coordinates:21 numerical and16 categorical
indicator coordinates (9+7). The inherited loader standardizes all coordinates,
including dummy columns; they are standardized one-hot vertices, not literal0/1
values at classifier input. Any simplex relaxation must account for that affine
transform. All category blocks in all splits match training vertices exactly;
all values finite, partitions disjoint, original download hashes verified.

Only data download/schema/preprocessing checks were executed here. No new model
training, Table1 comparison, or runtime values are claimed. This new explicit
mixed configuration is not proven to reproduce the undocumented Table1 encoding.
The historical HELOC preprocessing, runs and paper claims were not overwritten.

Reproduce audit:
`/home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/093_audit_fico_mixed.py`

Artifacts: download_provenance.json, dictionary_extracted.json,
schema_audit.json, split_ids_seed0.csv. Dataset bytes remain outside Git.
