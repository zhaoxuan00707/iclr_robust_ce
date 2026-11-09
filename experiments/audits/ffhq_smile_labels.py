"""Read-only parent audit; new lightweight results and annotation figures only."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'experiments/src'), str(ROOT / 'experiments/vendor/ffhq'), '/tmp/ffhq_grace_deps']
import clip
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF, InterpolationMode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    cp = ROOT / 'experiments/configs/ffhq_smile_label_audit.json'
    c = json.loads(cp.read_text())
    out = ROOT / 'results/018_ffhq_smile_label_audit'
    figs = ROOT / 'figures/ffhq_smile_label_audit'
    assert not out.exists() and not figs.exists(), 'Never overwrite an existing audit'
    out.mkdir(); figs.mkdir()
    (out / 'resolved_config.json').write_text(json.dumps(c, indent=2) + '\n')
    torch.set_num_threads(c['torch_threads']); torch.manual_seed(c['seed'])
    parent = Path(c['parent_run']); old = Path(c['previous_screen'])
    selected = pd.read_csv(parent / 'selected_examples.csv')
    selected = selected[selected.editor == c['editor']].sort_values('calibration_index')
    facts = pd.read_csv(parent / 'supervised_factual_scores.csv').set_index('calibration_index')
    source = pd.read_csv(parent / 'supervised_smile_source_validation.csv')
    records = []
    for r in selected.itertuples():
        i = int(r.calibration_index)
        for kind, path, p, sp, sl in [
            ('factual', old / 'qualitative_examples' / f'{i}_factual.png', r.factual_smile_score,
             facts.loc[i, 'supervised_smile_probability'], facts.loc[i, 'supervised_smile_label']),
            ('edited', Path(r.image_path), r.smile_score, r.supervised_smile_probability, r.supervised_smile_label)]:
            records.append(dict(domain='generated_calibration', index=i, kind=kind, image_path=str(path),
                                stored_original=p, supervised_probability=sp, supervised_label=int(sl)))
    for r in source.itertuples():
        records.append(dict(domain='celeba_source_diagnostic', index=int(r.index), kind='source',
                            image_path=str(parent / 'supervised_source_audit' / f'{r.index}.jpg'),
                            human_source_label=int(r.smiling), supervised_probability=r.supervised_smile_probability,
                            supervised_label=int(r.supervised_smile_label)))
    assert len(records) == 48 and len(selected) == 8 and selected.quality_pass.all()
    manifest = {r['image_path']: sha(r['image_path']) for r in records}
    for p in [cp, Path(__file__), Path(c['clip_checkpoint']), parent / 'selected_examples.csv',
              parent / 'supervised_factual_scores.csv', parent / 'supervised_smile_source_validation.csv']:
        manifest[str(p)] = sha(p)
    model, _ = clip.load(c['clip_checkpoint'], device='cpu', jit=False)
    model = model.float().eval().requires_grad_(False)
    with torch.no_grad():
        texts = {name: F.normalize(model.encode_text(clip.tokenize(pair)), dim=-1)
                 for name, pair in c['prompt_pairs'].items()}
    mean = torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None]

    def score(x, name):
        emb = F.normalize(model.encode_image((x - mean) / std), dim=-1)
        return (model.logit_scale.exp() * emb @ texts[name].T).softmax(-1)[:, 0]

    gradients = []
    for n, row in enumerate(records):
        x = TF.to_tensor(Image.open(row['image_path']).convert('RGB'))[None]
        x = TF.center_crop(TF.resize(x, 224, interpolation=InterpolationMode.BICUBIC, antialias=True), 224)
        with torch.no_grad():
            for name in texts:
                row[name + '_probability'] = float(score(x, name))
                row[name + '_label'] = int(row[name + '_probability'] >= c['threshold'])
        if row['domain'] == 'generated_calibration' and row['kind'] == 'factual':
            x.requires_grad_(True)
            grad, = torch.autograd.grad(score(x, 'original').sum(), x)
            direction = grad / grad.norm()
            with torch.no_grad():
                eps = c['gradient_epsilon']
                fd = float((score(x + eps * direction, 'original') - score(x - eps * direction, 'original')) / (2 * eps))
            auto = float((grad * direction).sum())
            error = abs(fd - auto) / max(abs(auto), 1e-8)
            gradients.append(dict(index=row['index'], autograd=auto, finite_difference=fd,
                                  relative_error=error, passed=bool(auto > 0 and error < c['gradient_relative_tolerance'])))
        print(f'scored {n+1}/{len(records)}', flush=True)
    d = pd.DataFrame(records); d.to_csv(out / 'scores.csv', index=False)
    pd.DataFrame(gradients).to_csv(out / 'gradient_checks.csv', index=False)
    summaries = []
    for name in texts:
        src = d[d.domain == 'celeba_source_diagnostic']
        gen = d[d.domain == 'generated_calibration']
        factual = gen[gen.kind == 'factual'].set_index('index')
        edit = gen[gen.kind == 'edited'].set_index('index')
        summaries.append(dict(prompt=name, source_correct=int((src[name+'_label'] == src.human_source_label).sum()),
                              source_count=len(src), factual_positive=int(factual[name+'_label'].sum()),
                              edited_positive=int(edit[name+'_label'].sum()),
                              generated_disagreements_with_supervised=int((gen[name+'_label'] != gen.supervised_label).sum()),
                              negative_to_positive=int(((factual[name+'_label']==0)&(edit[name+'_label']==1)).sum())))
    pd.DataFrame(summaries).to_csv(out / 'prompt_summary.csv', index=False)
    gen = d[d.domain == 'generated_calibration'].copy()
    error = float((gen.original_probability - gen.stored_original).abs().max())
    # A quantized PNG audit is not an exact rerun of native floating-point images.
    assert error < .02, f'Unexpected PNG scoring discrepancy: {error}'
    assert all(g['passed'] for g in gradients), gradients
    order = np.random.default_rng(c['seed']).permutation(16)
    mapping = gen.iloc[order][['index', 'kind', 'image_path']].copy()
    mapping.insert(0, 'item_id', [f'S{i+1:02d}' for i in range(16)])
    mapping.to_csv(out / 'annotation_key_do_not_show_raters.csv', index=False)
    labels = pd.DataFrame({'item_id': mapping.item_id, 'label': '', 'rater_id': '', 'notes': ''})
    labels.to_csv(out / 'annotation_template.csv', index=False)
    cards = []
    with PdfPages(figs / 'smile_annotation_blinded.pdf') as pdf:
        for start in range(0, 16, 8):
            fig, axes = plt.subplots(2, 4, figsize=(12, 7))
            for ax, r in zip(axes.flat, mapping.iloc[start:start+8].itertuples()):
                im = Image.open(r.image_path).convert('RGB'); im.thumbnail((384,384))
                ax.imshow(im); ax.set_title(r.item_id); ax.axis('off')
                buf = io.BytesIO(); im.save(buf, format='JPEG', quality=90)
                encoded = base64.b64encode(buf.getvalue()).decode()
                cards.append(f'<article><h2>{r.item_id}</h2><img alt="Face {r.item_id}" src="data:image/jpeg;base64,{encoded}"><label>Expression <select data-id="{r.item_id}"><option value="">Choose a label</option><option>smiling</option><option>not_smiling</option><option>uncertain</option></select></label></article>')
            fig.suptitle('Label each image: smiling / not smiling / uncertain.\nJudge visible expression; exposed teeth alone do not establish a smile.', fontsize=12)
            fig.tight_layout(rect=(0,0,1,.92)); pdf.savefig(fig); plt.close(fig)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Smile annotation</title>
<style>body{font:18px sans-serif;max-width:1200px;margin:32px auto;padding:16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:24px}img{width:100%}article{border:1px solid #ccc;padding:16px}select,input,button{font:inherit;margin:10px 0;padding:8px}label{display:block}</style>
<h1>Smile annotation</h1><p>Label each face independently as smiling, not_smiling, or uncertain. Judge visible expression; exposed teeth alone do not establish a smile. Choose uncertain if the expression is ambiguous or obscured. All items require an explicit choice. Scores, methods, and pairings are hidden.</p>
<p>Two raters should complete separate files independently. Do not inspect the result report or mapping before labeling.</p>
<label>Rater identifier <input id="rater" placeholder="e.g. rater_1"></label><button onclick="downloadLabels()">Download labels CSV</button><p id="status" role="status">Labels are kept in this page until exported; export before closing.</p><main>''' + ''.join(cards) + '''</main><script>
function downloadLabels(){const r=document.getElementById('rater').value.trim();const s=[...document.querySelectorAll('select')];if(!r||s.some(x=>!x.value)){document.getElementById('status').textContent='Enter a rater identifier and label every image (uncertain is allowed).';return;}const q=x=>'"'+x.replaceAll('"','""')+'"';const csv='item_id,label,rater_id\\n'+s.map(x=>[x.dataset.id,x.value,r].map(q).join(',')).join('\\n')+'\\n';const u=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const a=document.createElement('a');a.href=u;a.download='smile_labels.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);document.getElementById('status').textContent='CSV exported.';}
</script></html>'''
    (figs / 'smile_annotation.html').write_text(html)
    assert all(sha(p) == digest for p, digest in manifest.items())
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    checks = dict(images_scored=48, generated_images=16, gradient_checks_passed=len(gradients),
                  maximum_gradient_relative_error=max(g['relative_error'] for g in gradients),
                  maximum_png_probability_error=error, parent_hashes_unchanged=True,
                  annotation_items=len(mapping), human_labels_available=False,
                  FFHQ_accuracy_established=False, final_FFHQ_test_accessed=False,
                  classifier_replaced=False, identity_limits_changed=False, GRACE_started=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2)); print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
