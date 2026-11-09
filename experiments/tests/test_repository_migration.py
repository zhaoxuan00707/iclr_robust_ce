"""Checks for local source resolution and protection of historical run outputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_runner', ROOT / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RepositoryMigrationTests(unittest.TestCase):
    def test_local_package_origins(self):
        runner.bootstrap()
        for name in ('lift_ce', 'robustx', 'ellice'):
            self.assertTrue(Path(importlib.util.find_spec(name).origin).is_relative_to(ROOT))

    def test_historical_output_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.validate_output(Path('/home/xu.zhao/robust_ce_results/single_budget_geodesic_wro_v1/20260915_single_budget_v1'), runner.EXPERIMENTS[0])

    def test_new_run_and_resume_require_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'RUN_ROOT', Path(directory)):
            path, identity = runner.validate_output(Path(directory)/'new', runner.EXPERIMENTS[0])
            self.assertFalse(path.exists())
            path.mkdir()
            with self.assertRaises(ValueError):
                runner.validate_output(path, runner.EXPERIMENTS[0])
            with self.assertRaises(ValueError):
                runner.validate_output(path, runner.EXPERIMENTS[0], resume=True)
            (path/'repository_run.json').write_text(json.dumps({'identity': identity}))
            runner.validate_output(path, runner.EXPERIMENTS[0], resume=True)
            with self.assertRaises(ValueError):
                runner.validate_output(path, runner.EXPERIMENTS[1], resume=True)

    def test_vendor_revision_comes_from_snapshot(self):
        from lift_ce.experiments.heloc_diabetes_official_lift_v1 import git_version
        manifest = json.loads(runner.MANIFEST.read_text())
        self.assertEqual(git_version(ROOT/'vendor/RobustX'), manifest['vendors']['RobustX']['commit'])
