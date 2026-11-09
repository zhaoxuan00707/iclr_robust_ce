# Release verification

All commands were executed from this release copy using the existing Python3.11 research environment. No dependency installation, model training, or expensive experiment was launched. Test commands used the local source/vendor paths and one CPU thread.

| Check | Outcome |
|---|---|
| `python deliverables/grace-wachter/run.py --help` | PASS |
| `python experiments/run.py list` | PASS |
| `python experiments/run.py run appendix_tabular -- --help` | PASS |
| `python experiments/108_run_tabular_replication.py --help` | PASS |
| `python experiments/069_run_galaxy_update_aware.py --help` | PASS |
| `python experiments/072_run_ffhq_update_aware.py --help` | PASS |
| `python experiments/106_run_adversary_ablation.py --help` | PASS |
| `python experiments/run.py run appendix_tabular --output <configured-run-root>/grace_release_documentation_dry_run --dry-run -- --dataset diabetes --model mlp --config experiments/configs/appendix_tabular.json` | PASS |
| `python -m unittest discover -s experiments/tests -p test_mixed_update_aware.py` | PASS |
| `python -m unittest discover -s experiments/tests -p test_sensitivity_aggregation.py` | PASS |

The standalone requirements and full-package installation instructions were inspected, but a clean-environment installation was not run. CUDA execution and external assets require separate validation. Both extracted figures were visually checked against the PDF pages.

Final checks: no broken overview/documentation/image links; all1,269 inventoried original files unchanged; copied source, tests, configuration, vendor code, and portable example are byte-identical to the original. Test-generated caches removed from the release copy. Raw tensor artifacts and the older compiled manuscript excluded from the copy are inventoried in [omitted artifacts](omitted_artifacts.json).
