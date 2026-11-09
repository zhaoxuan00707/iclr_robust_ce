"""Add descriptive post-run interpretation without modifying frozen experiment outputs."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent.parent
out=Path('/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1');result=ROOT/'results/014_ffhq_grace_v1'
selected=json.loads((out/'selected_configuration.json').read_text());pooled=torch.load(out/'jobs'/f'pool_test_{selected["r_index"]}'/'selected.pt',map_location='cpu',weights_only=True)
origins={k:v['origin'] for k,v in pooled.items()};first=next(iter(pooled.values()))['points'];same=all(torch.equal(first,v['points']) for v in pooled.values());assert same and set(origins.values())=={'geodesic0.5'}
base=[json.loads((out/'jobs'/f'base_{split}'/'quality.json').read_text()) for split in ['calibration','test']];base_seconds=sum(v['base_seconds'] for v in base);robust_seconds=base_seconds+sum(v['robust_extra_seconds'] for v in base)
main=pd.read_csv(result/'main_results.csv').set_index('method');ratio=float(main.loc['GRACE_selected','pixel_L2_224']/main.loc['Base','pixel_L2_224'])
record={'candidate_origins':origins,'identical_beta_outputs':same,'no_observed_benefit_from_extra_beta':True,'selected_pixel224_cost_ratio_to_Base':ratio,'Base_generator_worker_seconds_both_splits':base_seconds,'LeapFactual_generator_worker_seconds_both_splits':robust_seconds,'identity_metric':'unavailable','qualitative_review':'Substantial changes in facial appearance, hair, clothing and background; visual grid does not establish identity preservation.','shift_result':'Selected GRACE1.0 on all brightness/noise levels,0.875 at strongest blur; Base noise validity rises with severity, so severity is not monotonic damage under this projection update.'}
text='''# FFHQ result interpretation

All four beta values return the same retained **geodesic midpoint** population (`geodesic0.5`). It already has zero finite-bank hinge at beta1.00. Thus this run provides **no evidence of benefit from additional geometry freedom**. Calibration selected r=1, beta=1.00.

At nominal external severity, prompt-relative Future Validity is45.83% for Base,83.33% for LeapFactual and100% for selected GRACE. Mean pixel224L2 costs are38.74,54.64 and92.91 respectively; GRACE costs approximately2.40× Base. LPIPS is0.133,0.202 and0.382. All test comparisons use the same8 factuals.

The fixed qualitative grid was inspected after numerical verification. The GRACE columns show substantial changes in facial appearance, hair, clothing and background. This is a material limitation, not evidence of identity preservation. No identity model was available; the identity CSV explicitly reports unavailable values. The original CLIP prompt pair also misclassifies the visibly smiling audit face relative to literal smile appearance, so validity is only the original classifier's prompt-relative outcome.

Selected GRACE remains100% under the evaluated brightness/noise levels, but falls to87.5% at the strongest blur. Base validity increases under stronger noise updates; therefore greater external severity is not uniformly more damaging in this fixed projection-update experiment. No confidence interval is inferred from deterministic replica copies.

Construction wall time is538.55 seconds(8.98 minutes),67.32 seconds per test CF, and0.584 summed worker GPU-hours. It includes generated-data preparation, both Base/LeapFactual generation, the attack bank, calibration, all optimization and freezing; it excludes pretraining and final future evaluation. Generator-only worker times for both splits are saved below and are not directly comparable to the complete GRACE wall interval.

'''+json.dumps(record,indent=2)+'\n'
for path in [result/'interpretation.md',out/'interpretation.md']:
 if path.exists():assert path.read_text()==text
 else:path.write_text(text)
for path in [result/'postrun_interpretation.json',out/'postrun_interpretation.json']:
 if not path.exists():path.write_text(json.dumps(record,indent=2)+'\n')
p=result/'report.md';s=p.read_text();link='See [result interpretation](interpretation.md) for the identical midpoint outputs and substantial appearance-change limitation.\n\n'
if link not in s:p.write_text(s.replace('# FFHQ-generated face GRACE\n\n','# FFHQ-generated face GRACE\n\n'+link,1))
print(json.dumps(record,indent=2))
