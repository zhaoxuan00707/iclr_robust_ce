"""Finalize annotation layout and state without repeating classifier inference."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffhq_smile_labels import sha
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

out = ROOT / 'results/018_ffhq_smile_label_audit'
figs = ROOT / 'figures/ffhq_smile_label_audit'
key = pd.read_csv(out / 'annotation_key_do_not_show_raters.csv')
pdfpath = figs / 'smile_annotation_blinded_v2.pdf'
htmlpath = figs / 'smile_annotation_v2.html'
assert not pdfpath.exists() and not htmlpath.exists()
with PdfPages(pdfpath) as pdf:
    for start in range(0, 16, 8):
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        fig.subplots_adjust(left=.025, right=.975, bottom=.03, top=.87, wspace=.08, hspace=.22)
        for ax, r in zip(axes.flat, key.iloc[start:start+8].itertuples()):
            im = Image.open(r.image_path).convert('RGB'); im.thumbnail((384,384))
            ax.imshow(im); ax.set_title(r.item_id, pad=8); ax.axis('off')
        fig.suptitle('Label each image: smiling / not smiling / uncertain.\nJudge visible expression; exposed teeth alone do not establish a smile.', fontsize=12, y=.97)
        pdf.savefig(fig); plt.close(fig)
html = (figs / 'smile_annotation.html').read_text().replace("x.replaceAll('\"','\"\"')", "x.split('\"').join('\"\"')")
htmlpath.write_text(html)
manifest = json.loads((out / 'input_manifest.json').read_text())
assert all(sha(p) == digest for p, digest in manifest.items())
(out / 'finalization_manifest.json').write_text(json.dumps({str(p): sha(p) for p in [Path(__file__), pdfpath, htmlpath]}, indent=2)+'\n')
statepath = ROOT / 'state.yaml'; state = yaml.safe_load(statepath.read_text())
entry = dict(task='tasks/005_face_smile_label_audit.md', phase='automated_audit_complete',
             report='results/018_ffhq_smile_label_audit/report.md',
             updated_utc=datetime.now(timezone.utc).isoformat(),
             generated_images=16, source_diagnostic_images=32, image_gradient_checks_passed=8,
             focused_tests_passed=4, human_labels_available=False,
             semantic_smile_validation_established=False, final_FFHQ_test_accessed=False,
             classifier_replaced=False, identity_limits_changed=False, GRACE_started=False,
             device='cpu', annotation_html='figures/ffhq_smile_label_audit/smile_annotation_v2.html',
             annotation_pdf='figures/ffhq_smile_label_audit/smile_annotation_blinded_v2.pdf')
state.update(active_task=entry['task'], status='awaiting_human_labels', current_work=entry,
             next_actions=['Collect two independent human annotation CSVs for the 16 shuffled images; adjudicate uncertain/disagreeing labels.',
                           'Evaluate frozen classifier predictions against adjudicated labels before selecting a smile classifier or starting GRACE.'])
state.setdefault('completed_work', {})['ffhq_smile_label_automated_audit'] = dict(entry, status='complete')
statepath.write_text(yaml.safe_dump(state, sort_keys=False))
with (ROOT / 'notes/decisions.md').open('a') as f:
    f.write('\n\n## 2026-09-17: FFHQ smile-label audit\n\n'
            'Completed automated audit of 16 saved generated calibration images and 32 reused source-domain examples. '
            'All eight CLIP image-gradient checks pass (maximum relative discrepancy 0.7455%). '
            'Prompt wording changes positive-edit counts from 6/8 to 2/8 or 5/8; the third wording also marks 4/8 factuals smiling. '
            'Do not select a replacement classifier from this small diagnostic or use model/assistant judgments as human labels. '
            'Keep the independent4 editor and identity gates fixed. Human validation is pending; shuffled annotation HTML/PDF v2 are ready. '
            'No final-test access, GRACE run, old-output modification, or manuscript change. '
            'Report: results/018_ffhq_smile_label_audit/report.md. State records automated completion separately from pending domain validation.\n')
print('Final annotation package and state saved.')
