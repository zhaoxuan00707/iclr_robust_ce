"""Render a disclosed post-hoc subset of already frozen FFHQ counterfactuals."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ffhq_subset_mpl')
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/src'))
from lift_ce.ffhq.assets import Generator
import lpips


def main():
    raw = Path('/home/xu.zhao/robust_ce_results/ffhq_grace_v1/20260916_full_v1')
    dest = ROOT / 'figures/ffhq_grace_v1'
    summary = ROOT / 'results/014_ffhq_grace_v1'
    target = dest / 'ffhq_qualitative_1.pdf'
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    load = lambda name: torch.load(raw / name, map_location='cpu', weights_only=True)
    facts, frozen = load('test_factuals.pt'), load('frozen.pt')
    ids = np.load(raw / 'test_factual_ids.npy')
    betas = [f'GRACE_beta{b:.2f}' for b in [1, 1.05, 1.1, 1.2]]
    for name in betas:
        assert torch.equal(frozen[name]['points'], frozen['GRACE_selected']['points'])
    generator = Generator('cuda')
    native = {}
    with torch.no_grad():
        for name, points in [('Factual', facts['w']), ('Base', frozen['Base']['points']),
                             ('GRACE', frozen['GRACE_selected']['points'])]:
            native[name] = torch.cat([generator.decode(w.cuda()).clamp(-1, 1).cpu()
                                      for w in points.split(1)])
            print('Rendered frozen', name, flush=True)
    del generator
    torch.cuda.empty_cache()
    torch.hub.set_dir('/tmp/ffhq_torch_hub')
    metric = lpips.LPIPS(net='squeeze').cuda().eval()
    rows = []
    with torch.no_grad():
        for i, factual_id in enumerate(ids):
            row = {'test_index': i, 'factual_id': str(factual_id)}
            for name in ['Base', 'GRACE']:
                row[name + '_LPIPS_squeeze_1024'] = float(metric(
                    native['Factual'][i:i+1].cuda(), native[name][i:i+1].cuda()).item())
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(['GRACE_LPIPS_squeeze_1024', 'test_index'])
    table['rank'] = range(1, len(table) + 1)
    table['selected'] = table['rank'] <= 3
    chosen = table.loc[table.selected, 'test_index'].tolist()
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.3))
    titles = ['Factual', 'Base CF', r'GRACE $\beta=1.00$', r'GRACE $\beta=1.05$',
              r'GRACE $\beta=1.10$', r'GRACE $\beta=1.20$']
    for row, index in enumerate(chosen):
        for col, name in enumerate(['Factual', 'Base'] + ['GRACE'] * 4):
            axes[row, col].imshow(((native[name][index] + 1) / 2).permute(1, 2, 0).numpy())
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(f'Test {index}', fontsize=10)
    fig.subplots_adjust(left=.035, right=.995, top=.94, bottom=.075, wspace=.025, hspace=.035)
    fig.text(.5, .025, 'Post-hoc selection: 3 lowest factual-to-GRACE LPIPS cases among 8 test examples. '
             'All four budgets produced identical outputs.', ha='center', fontsize=8)
    fig.savefig(target, dpi=220)
    fig.savefig(target.with_suffix('.png'), dpi=180)
    plt.close(fig)
    table.to_csv(summary / 'qualitative_1_selection.csv', index=False)
    manifest = {'source_run': str(raw), 'selection': 'Three lowest native-1024 factual-to-GRACE SqueezeNet LPIPS; ties by test index',
                'post_hoc': True, 'candidate_count': 8, 'selected_test_indices_in_row_order': chosen,
                'future_validity_used_for_selection': False, 'all_beta_outputs_identical': True,
                'rendering': 'Original frozen StyleGAN3, constant noise, saved W codes; no optimization rerun',
                'source_sha256': {name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
                                  for name in ['test_factuals.pt', 'frozen.pt', 'test_factual_ids.npy']},
                'command': 'PYTHONPATH=/tmp/ffhq_grace_deps:/tmp/galaxy_grace_deps CUDA_VISIBLE_DEVICES=2 /home/xu.zhao/miniforge3/envs/wm_gpu/bin/python experiments/027_ffhq_qualitative_subset.py'}
    (summary / 'qualitative_1_selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    assert target.stat().st_size > 10000
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
