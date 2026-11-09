"""Save final interpretation without overwriting initial run outputs."""
import json,sys
from pathlib import Path
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/xu.zhao/robust_ce_results/ffhq_grace_identity_v1/20260917_identity_v1')
DEST=ROOT/'results/015_ffhq_identity_v1'

def main():
    audit=OUT/'identity_detection_audit';main=pd.read_csv(audit/'full_vs_bottleneck.csv');geo=pd.read_csv(audit/'geometry_ablation.csv');all_results=pd.read_csv(audit/'all_results.csv')
    review=json.loads((audit/'review.json').read_text());assert review['missing_pairs']==0 and review['paired_count']==8
    cols=['method','Future_Validity','pixel_L2_224','Identity_Similarity','identity_std','identity_p10','LPIPS_squeeze','runtime_worker_seconds']
    text=(OUT/'report.md').read_text()
    start=text.index('## Table 1:');end=text.index('## Runtime',start)
    text=text[:start]+'\n\n'.join([
        '## Table 1: matched results at nominal severity',main[cols].to_markdown(index=False),
        'Cost is pixel L2 at224RGB; lower is better. Identity is ArcFace cosine similarity after five-landmark alignment; higher is better. All methods have8/8 valid identity pairs with the audited320px detector scale, confidence threshold0.5. Identity std uses ddof=0 and the low percentile is10th percentile. LPIPS uses native1024 images and SqueezeNet. Runtime is summed worker-seconds for eight-CF test generation/refinement, excluding shared costs.',
        '## Table 2: task-space geometry budget',geo[['beta_geo','Future_Validity','Phi_hat_r','pixel_L2_224','Identity_Similarity','identity_std','identity_p10','runtime_worker_seconds']].to_markdown(index=False),
        'All four test budgets selected the same task-anchor endpoint (geodesic1.0), so no beta effect was observed. Full GRACE selected geodesic0.5. Shared-pool beta runtimes must not be summed.',
        '## Outcome: success criterion not met',
        'The bottleneck retains the full nominal robustness gain (FV1.0 versus Base0.45833), but worsens mean identity similarity relative to full GRACE (0.19832 versus0.27228; paired difference−0.07397 across all8 factuals). Base similarity is0.82724. LPIPS increases from0.38203 to0.48935 and pixel224L2 from92.91 to128.82. Thus this approximate classifier-gradient bottleneck does not achieve useful identity preservation on the tested cohort.',
        'The content restriction is numerically enforced (maximum drift1.03e-7), and factual reconstruction is exact in the saved calibration check. This does not make the complement a validated identity representation. The29 task directions retain substantial appearance information; moving them to target anchors can change facial appearance, hair, clothing and background despite freezing483 complementary coordinates.',
        '## Conditional-Base diagnostic',all_results[all_results.method.isin(['Base','Conditional_Base'])][['method','Future_Validity','pixel_L2_224','Identity_Similarity','LPIPS_squeeze']].to_markdown(index=False),
        'Projecting the Base task code while restoring factual content gives slightly higher mean identity similarity but lower Future Validity. Subsequent task-space robust refinement selects much larger changes. This diagnostic separates the conditional realization from its robust optimization.',
        '## Detector-scale audit and authoritative identity files',
        'The initial640px detector missed5/8 factual faces, making initial identity averages incomplete and unsuitable for matched comparison. A read-only factual-image audit found all8 at320px, at the unchanged0.5 confidence threshold. That scale was fixed before reevaluating all CF methods; identity scores were not used to choose it. All factual/CF images are now detected. Aligned crops were visually checked. The original640px measurements remain intact for traceability.',
        f'Use `{audit}/identity_metrics.csv`, `full_vs_bottleneck.csv`, `geometry_ablation.csv`, `identity_per_example.csv`, `identity_paired.csv`, `protocol.json` and `review.json` for the corrected identity evaluation. They are also stored in `results/015_ffhq_identity_v1/identity_detection_audit/`. Original root identity summaries and the initial report are superseded for identity comparisons only. Optimization, future predictions and qualitative images are unchanged.',
        '## Qualitative inspection',
        'The complete fixed-order eight-row panel shows substantial changes in facial appearance, hairstyle, accessories and background in several bottleneck cases. The result does not support an identity-preservation claim. No case was removed or selected for appearance. Face-embedding similarity is an evaluation signal rather than an identity guarantee.',
    ])+'\n\n'+text[end:]
    text=text.replace('# FFHQ hard-content bottleneck GRACE','# FFHQ hard-content bottleneck GRACE — final report',1)
    text=text.replace('## Independent verification','## Original-run independent verification (before detector audit)')
    text+='\n\n## Final audit verification\n\n'+json.dumps(review,indent=2)+'\n\nFive focused unit tests passed. Both qualitative PDFs are one-page documents; image panels and aligned face crops were inspected. Original source hashes and all experiment outputs are preserved.\n'
    (OUT/'final_report.md').write_text(text);(DEST/'final_report.md').write_text(text)
    final={'status':'complete','scientific_success':False,'reason':'Robustness retained, but bottleneck identity similarity is lower and appearance costs higher than full GRACE','test_cases':8,'identity_valid_pairs_per_method':8,'report':str(OUT/'final_report.md'),'original_review':str(OUT/'completion_review.json'),'identity_review':str(audit/'review.json'),'focused_tests_passed':5,'visual_panels_checked':True,'all_eight_gpus_used':True}
    (OUT/'final_completion_review.json').write_text(json.dumps(final,indent=2)+'\n');(DEST/'final_completion_review.json').write_text(json.dumps(final,indent=2)+'\n')
    statepath=ROOT/'state.yaml';s=yaml.safe_load(statepath.read_text());s['status']='complete';s['active_task']='tasks/002_face_baseline.md';s['current_work'].update(phase='complete',report='results/015_ffhq_identity_v1/final_report.md',scientific_success=False,identity_preservation_established=False,selected_r=2.0,selected_beta=1.05,task_dimension=29,content_dimension=483,identity_valid_pairs=8)
    s['current_work'].pop('note',None);s['completed_work']['ffhq_grace_identity_v1']=s['current_work'].copy();statepath.write_text(yaml.safe_dump(s,sort_keys=False))
    with (ROOT/'notes/decisions.md').open('a') as f:
        f.write('\nFFHQ identity task completed:56 jobs across GPUs0–7, five focused tests and independent source/geometry/content/prediction checks passed. Initial640px ArcFace detector missed5/8 factual faces. A factual-only320px scale audit (same0.5 threshold) detected all8; corrected identity measurements and alignment checks are preserved separately in `identity_detection_audit/`. Final eight-pair means: Base FV0.45833/identity0.82724; full GRACE FV1.0/identity0.27228; bottleneck GRACE FV1.0/identity0.19832. Bottleneck also has higher LPIPS0.48935 versus0.38203 and cost128.82 versus92.91. All betas return the same target-task endpoint. Despite enforcing483 immutable content coordinates and exact reconstruction, this fallback fails the stated identity-preservation success criterion. Final authoritative report: `results/015_ffhq_identity_v1/final_report.md`; no positive identity claim or manuscript change.\n')
    print(json.dumps(final,indent=2))

if __name__=='__main__':main()
