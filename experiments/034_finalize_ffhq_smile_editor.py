"""Persist reviewed screen outcome; do not launch GRACE on failed gates."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/xu.zhao/robust_ce_results/ffhq_smile_editor_v1/20260917_calibration_v3')
DEST=ROOT/'results/016_ffhq_smile_editor_v1'

def main():
    gate=json.loads((OUT/'editor_gate.json').read_text());assert not gate['numerical_gate_passed']
    comparison=pd.read_csv(OUT/'editor_comparison.csv');review=json.loads((OUT/'completion_review.json').read_text());assert review['missing_arcface_pairs']==review['missing_facenet_pairs']==0
    outcome={'status':'complete','screen_passed':False,'GRACE_started':False,'reason':'No localized editor satisfied smile and preservation gates on any of the8 calibration factuals;6 were required','visual_review':'Most facial appearance, hair, accessories and background stay similar. Some localized mouth changes are visible, but no reliable non-smiling-to-smiling behavior across the cohort. Some factuals already visually smile despite low original CLIP scores, so the original pseudo-label classifier remains a semantic limitation.','arcface_mean_range':[float(comparison[comparison.editor.str.startswith('style')].mean_arcface.min()),float(comparison[comparison.editor.str.startswith('style')].mean_arcface.max())],'style_passing_counts':[0,0,0],'w_linear_passing_count':1,'final_test_accessed':False,'future_models_accessed':False,'thresholds_relaxed':False,'focused_tests_passed':7,'pdf_inspected':True,'completed_utc':datetime.now(timezone.utc).isoformat()}
    text=(OUT/'report.md').read_text()
    text+='\n\n## Final reviewed outcome\n\nThe screen is complete. Localized edits retained high measured identity similarity (ArcFace means0.958–0.975; FaceNet0.980–0.987) with very small perceptual changes. However, each localized editor passed0/8 cases; the W baseline passed1/8. Six passing cases were required. No editor is accepted for the GRACE follow-up, and no final test or future models were accessed. These results show a useful small-edit preservation regime, not a successful counterfactual method.\n\nQualitative inspection of all8 calibration cases shows mostly preserved appearance and some small mouth changes, but not a reliable smile transition. Some factuals already visually smile despite original CLIP probabilities below0.5; therefore the smile pseudo-label model also needs independent validation before treating these scores as semantic smile success. This observation did not change thresholds, outputs or the gate. The next scientific question is whether a validated smile signal and a stronger localized control can jointly satisfy the fixed preservation constraints; that is not established here.\n\n'+json.dumps(outcome,indent=2)+'\n'
    (OUT/'final_report.md').write_text(text);(DEST/'final_report.md').write_text(text)
    (OUT/'final_outcome.json').write_text(json.dumps(outcome,indent=2)+'\n');(DEST/'final_outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
    runtime=pd.DataFrame([json.loads(p.read_text()) for p in (OUT/'jobs').glob('*/complete.json')]);runtime.to_csv(OUT/'runtime.csv',index=False);runtime.to_csv(DEST/'runtime.csv',index=False)
    shutil.copyfile(OUT/'discovery_reuse.json',DEST/'discovery_reuse.json')
    path=ROOT/'state.yaml';s=yaml.safe_load(path.read_text());s['status']='complete';s['active_task']='tasks/003_face_editor_screen.md';s['current_work'].update(phase='complete',report='results/016_ffhq_smile_editor_v1/final_report.md',screen_passed=False,GRACE_started=False,final_test_accessed=False,focused_tests_passed=7,updated_utc=outcome['completed_utc']);s.setdefault('completed_work',{})['ffhq_smile_editor_v1']=s['current_work'].copy();path.write_text(yaml.safe_dump(s,sort_keys=False))
    with (ROOT/'notes/decisions.md').open('a') as f:
        f.write('\nLocalized editor screen complete on all8 GPUs,192 calibration candidates. Style1/4/8 passed0/8 each; W-direction baseline1/8, below the predeclared6/8 gate. Localized ArcFace means0.958–0.975 and FaceNet0.980–0.987 indicate small-edit preservation, but mean smile scores remain0.269–0.294, below0.5. Visual review shows mostly stable appearance and weak/inconsistent smile edits; factual smile/pseudo-label mismatches persist. No GRACE follow-up started and final test/future models remained untouched. Seven focused tests and independent gate/identity/zero-edit checks passed. Final report: `results/016_ffhq_smile_editor_v1/final_report.md`.\n')
    print(json.dumps(outcome,indent=2))

if __name__=='__main__':main()
