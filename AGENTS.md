# Project

This repository contains the code, experiments, results, and LaTeX source
for a research paper.

## Repository structure

- `paper/`: LaTeX manuscript
- `experiments/`: experiment code and configs
- `results/`: lightweight experiment summaries
- `figures/`: publication figures
- `tasks/`: research and experiment tasks
- `notes/`: decisions and open questions

## Existing experiments

Some experiments have already been run before this repository structure was finalized.

Do not delete, overwrite, rename, or rerun existing experiment outputs unless explicitly requested.

When existing runs are discovered:
- identify their current server paths;
- preserve the original outputs;
- summarize important metrics in `results/`;
- record the config, command, seed, checkpoint, and server path when available;
- do not copy large checkpoints or raw outputs into Git.

## General rules

1. Do not commit datasets, model checkpoints, or large raw experiment outputs.
2. Store reproducible experiment configs in `experiments/`.
3. Store concise experiment summaries in `results/`.
4. Publication-ready figures belong in `figures/`.
5. Do not change scientific claims in the manuscript unless supported by results.
6. Record important research decisions in `notes/decisions.md`.
7. Before finishing a task, run relevant tests or sanity checks.
8. Preserve existing experiment outputs unless explicitly instructed otherwise.

## Paper

The manuscript entry point is:

`paper/iclr2027_conference.tex`

## Experiments

Raw runs may live outside the Git repository.

When doing so, record their absolute server path in the result summary.

Existing raw runs should be treated as source data and should not be modified in place.
