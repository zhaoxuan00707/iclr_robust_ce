#!/usr/bin/env bash
# Run from the repository root. Requires Poppler's pdftoppm.
set -eu
pdftoppm -f 4 -l 4 -scale-to 5000 -x 2300 -y 2485 -W 900 -H 550 \
  -singlefile -png iclr_robust_ce-88.pdf assets/figures/grace_two_anchor_geometry
pdftoppm -f 1 -l 1 -scale-to 5000 -x 2060 -y 2750 -W 1130 -H 580 \
  -singlefile -png iclr_robust_ce-88.pdf assets/figures/shared_model_shift
