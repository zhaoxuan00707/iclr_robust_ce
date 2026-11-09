#!/usr/bin/env python3
"""Run the imported core from this checkout with explicit, isolated outputs."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RUN_ROOT = Path('/home/xu.zhao/robust_ce_results/iclr_runs')
MANIFEST = ROOT / 'provenance/core_import.json'
EXPERIMENTS = tuple(n for n in json.loads(MANIFEST.read_text())['scope']['roots']
                    if n != 'fixed_instance_stability_audit') + ('path_level_geodesic_wro_v1', 'favorable_region_geodesic_wro_v1', 'minimal_validity_repair_wro_v1', 'stylemnist_grace_v1', 'galaxy_grace_v1', 'ffhq_grace_v1', 'ffhq_grace_identity_v1', 'ffhq_smile_editor_v1', 'ffhq_smile_editor_v2', 'genki_ffhq_classifier_v1', 'genki_grace_identity_v1', 'genki_grace_beta_extension', 'genki_preservation_search', 'genki_expanded_editor', 'heloc_realization_comparison', 'mnist_update_aware', 'mnist_latent_base_fix', 'heloc_smooth_hinge', 'mnist_small_radius', 'mnist_weaker_classifier', 'mnist_low_margin_base', 'galaxy_update_aware', 'ffhq_update_aware', 'text_toxicity_update_aware')
LOCAL_PATHS = (ROOT / 'src', ROOT / 'vendor/RobustX', ROOT / 'vendor/ellice', ROOT / 'vendor/ffhq')


def bootstrap():
    """Prefer this checkout even when the environment has old editable installs."""
    sys.path[:0] = [str(p) for p in LOCAL_PATHS]
    for package in ('lift_ce', 'robustx', 'ellice'):
        module = importlib.import_module(package)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(ROOT):
            raise RuntimeError(f'{package} unexpectedly loaded from {origin}')


def code_fingerprint():
    digest = hashlib.sha256()
    for base in LOCAL_PATHS:
        for path in sorted(base.rglob('*.py')):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_output(path, experiment, resume=False):
    """Never claim an existing historical directory as a new run."""
    path = path.resolve()
    allowed_root = (Path('/home/xu.zhao/robust_ce_results') / experiment
                    if experiment in ('stylemnist_grace_v1', 'galaxy_grace_v1', 'ffhq_grace_v1', 'ffhq_grace_identity_v1', 'ffhq_smile_editor_v1', 'ffhq_smile_editor_v2', 'genki_ffhq_classifier_v1', 'genki_grace_identity_v1', 'genki_grace_beta_extension', 'genki_preservation_search', 'genki_expanded_editor') else RUN_ROOT)
    if path == allowed_root or not path.is_relative_to(allowed_root):
        raise ValueError(f'Output must be a new subdirectory of {allowed_root}')
    marker = path / 'repository_run.json'
    identity = {'repository': str(REPO), 'experiment': experiment,
                'code_sha256': code_fingerprint()}
    if resume:
        if not marker.is_file() or json.loads(marker.read_text())['identity'] != identity:
            raise ValueError('Resume requires a matching repository/module/code marker')
    elif path.exists():
        raise ValueError('Output already exists; use --resume only for a matching run')
    return path, identity


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    own, forwarded = argv[:split], argv[split+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'check', 'test', 'run'])
    parser.add_argument('experiment', nargs='?', choices=EXPERIMENTS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(own)
    bootstrap()
    if args.action == 'list':
        print('\n'.join(EXPERIMENTS)); return
    if args.action == 'check':
        for name in EXPERIMENTS:
            importlib.import_module('lift_ce.experiments.' + name)
        origins = {name: str(Path(mod.__file__).resolve()) for name, mod in sys.modules.items()
                   if name.split('.')[0] in ('lift_ce', 'robustx', 'ellice') and getattr(mod, '__file__', None)}
        if any(not Path(p).is_relative_to(ROOT) for p in origins.values()):
            raise RuntimeError('Historical source fallback detected')
        print(json.dumps({'imported_local_modules': len(origins), 'code_sha256': code_fingerprint()}, indent=2)); return
    if args.action == 'test':
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if not args.experiment:
        parser.error('run requires an experiment')
    if any(x == '--output' or x.startswith('--output=') for x in forwarded):
        parser.error('Use the launcher --output before the -- separator')
    # Help must remain read-only and requires no run directory.
    if '-h' in forwarded or '--help' in forwarded:
        sys.argv = [args.experiment, *forwarded]
        runpy.run_module('lift_ce.experiments.' + args.experiment, run_name='__main__'); return
    if not args.output:
        parser.error('run requires --output')
    path, identity = validate_output(args.output, args.experiment, args.resume)
    command = [args.experiment, *forwarded, '--output', str(path)]
    if args.dry_run:
        print(json.dumps({'argv': command, 'identity': identity}, indent=2)); return
    if not args.resume:
        path.mkdir(parents=True, exist_ok=False)
        (path / 'repository_run.json').write_text(json.dumps({'identity': identity, 'initial_argv': command}, indent=2)+'\n')
    sys.argv = command
    runpy.run_module('lift_ce.experiments.' + args.experiment, run_name='__main__')


if __name__ == '__main__':
    main()
