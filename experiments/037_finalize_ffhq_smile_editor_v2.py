"""Record the reviewed outcome and preserve source-audit provenance."""
import json,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1];OUT=Path('/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v2/20260917_extended_v1');DEST=ROOT/'results/017_ffhq_smile_editor_v2'

def main():
    summary=pd.read_csv(OUT/'editor_comparison.csv');r=summary.set_index('editor');assert r.loc['independent4','passing_factuals']==6 and r.loc['independent8','passing_factuals']==6
    review=json.loads((OUT/'completion_review.json').read_text());assert review['status']=='passed'
    observations=[
      {'calibration_index':0,'observation':'Factual already has an open, smile-like mouth; edit strengthens it. Both classifiers are questionable as annotations of a non-smiling factual.'},
      {'calibration_index':1,'observation':'Numerical CLIP pass, but little clear visible smile change in selected independent edits.'},
      {'calibration_index':2,'observation':'Mouth corners appear more raised; subtle smile-like change, supervised classifier remains negative.'},
      {'calibration_index':3,'observation':'More teeth visible and mouth shape changes, supervised classifier remains negative; no definitive semantic label assigned.'},
      {'calibration_index':4,'observation':'Visible mouth/teeth change despite original CLIP failure; change could be expression other than smiling.'},
      {'calibration_index':5,'observation':'Mouth changes visibly; independent model labels factual and edits smiling, while original CLIP fails.'},
      {'calibration_index':6,'observation':'Four-channel edit appears more smile-like and independent model becomes positive; eight-channel selected edit is less clear and independent model stays negative.'},
      {'calibration_index':7,'observation':'Four-channel edit exposes teeth and looks more smile-like; independent model becomes positive. Eight-channel change is subtler.'}]
    pd.DataFrame(observations).to_csv(OUT/'qualitative_review.csv',index=False);pd.DataFrame(observations).to_csv(DEST/'qualitative_review.csv',index=False)
    outcome={'status':'complete','numerical_gate_passed':True,'candidate_editor':'independent4','candidate_reason':'Same6/8 original-gate success as independent8, with higher mean ArcFace and lower LPIPS; calibration-selected candidate only','candidate_mean_ArcFace':float(r.loc['independent4','mean_ArcFace']),'candidate_mean_FaceNet':float(r.loc['independent4','mean_FaceNet']),'candidate_mean_LPIPS':float(r.loc['independent4','mean_LPIPS']),'semantic_smile_validation_established':False,'GRACE_started':False,'reason_for_no_GRACE':'Original score, supervised diagnostic and qualitative appearances disagree; consistent non-smiling-to-smiling interpretation is not established on the required cohort','preservation_limits_relaxed':False,'final_FFHQ_test_accessed':False,'future_models_accessed':False,'focused_tests_passed':9,'qualitative_panels_inspected':True,'completed_utc':datetime.now(timezone.utc).isoformat()}
    report=(OUT/'report.md').read_text()
    report+='\n\n## Final reviewed outcome\n\nIndependent movement of4 or8 selected channels reaches the original numerical gate on6/8 calibration faces without relaxing any preservation limit. The4-channel candidate is preferable within this calibration comparison: mean ArcFace0.8585 versus0.8386, FaceNet0.9255 versus0.9299, LPIPS0.0300 versus0.0391. These means include all8 selected calibration outputs; they should not be compared directly to prior results on different test factuals.\n\nThis is real progress in constrained editor feasibility, but not established semantic smile success or robust recourse. Of six original-gate successes, the supervised model marks only two of the4-channel and one of the8-channel selected outputs smiling. Qualitative inspection shows useful mouth changes in several cases, but one numerical success has little visible smile change and another factual already appears smile-like. The supervised classifier also disagrees with some visibly smile-like images; its small source-domain audit does not validate it on FFHQ. It is not adopted as a replacement classifier.\n\nThe4-channel control is saved as a promising calibration-selected candidate, not a validated final editor. GRACE is not launched because the prior prerequisite—convincing smiles under preservation limits—has not been established consistently. The next unresolved work is a dependable smile-label/evaluation protocol on these generated faces; no labels, thresholds or existing outputs were silently replaced.\n\nPer-case observations in qualitative_review.csv are descriptive assistant review, not blinded human annotations, adjudicated ground truth or face-identity recognition. They must not be used as a numerical accuracy benchmark.\n\n'+json.dumps(outcome,indent=2)+'\n'
    (OUT/'final_report.md').write_text(report);(DEST/'final_report.md').write_text(report);(OUT/'final_outcome.json').write_text(json.dumps(outcome,indent=2)+'\n');(DEST/'final_outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
    target=OUT/'supervised_source_audit';target.mkdir(exist_ok=True);source=Path('/tmp/ffhq_celeba_smile_audit');hashes={}
    for p in source.glob('*'):
        if p.suffix not in ['.jpg','.json']:continue
        dest=target/p.name
        if not dest.exists():shutil.copyfile(p,dest)
        hashes[p.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    provenance={'source':'https://huggingface.co/datasets/eurecom-ds/celeba','revision':'e9092e921951927418a9e8f87822020206b292ce','split':'test','row_indices':list(range(32)),'smiling_attribute_index':31,'images_external_not_in_git':True,'potential_training_overlap_unknown':True,'sha256':hashes}
    (target/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n');(DEST/'supervised_source_audit_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    sfile=ROOT/'state.yaml';s=yaml.safe_load(sfile.read_text());s['status']='complete';s['active_task']='tasks/004_face_editor_extension.md';s['current_work'].update(phase='complete',report='results/017_ffhq_smile_editor_v2/final_report.md',numerical_gate_passed=True,semantic_smile_validation_established=False,candidate_editor='independent4',GRACE_started=False,final_FFHQ_test_accessed=False,focused_tests_passed=9,updated_utc=outcome['completed_utc']);s.setdefault('completed_work',{})['ffhq_smile_editor_v2']=s['current_work'].copy();sfile.write_text(yaml.safe_dump(s,sort_keys=False))
    with (ROOT/'notes/decisions.md').open('a') as f:
        f.write('\nExtended editor screen complete: stronger scalar controls pass4/8,1/8,3/8; independent4 and independent8 each pass6/8 original numerical gates. Independent4 is the calibration candidate (ArcFace0.8585, FaceNet0.9255, LPIPS0.0300; all preservation limits unchanged). Independent supervised smiles corroborate only2/6 selected original successes for independent4 and1/6 for independent8. Qualitative review shows meaningful changes in some cases but inconsistent source/target smile semantics and disagreement of both classifiers. Do not launch GRACE or claim validated smile recourse yet. Final FFHQ test/future models remain untouched. Nine focused tests and independent candidate/embedding/gate checks passed. Final report: `results/017_ffhq_smile_editor_v2/final_report.md`.\n')
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
