# Release preparation notes

## Scope

The root README describes the supplied anonymous version88 manuscript. Its two
illustrations are direct PDF crops: Figure2 on page4 and Figure1 on page1.
[Figure provenance](../assets/figures/provenance.json) records the source hash,
crop coordinates, and output hashes; [the extraction script](../assets/figures/extract_from_paper.sh)
regenerates them with Poppler. The source manuscript remains unchanged.

No algorithm, model, experiment configuration, dependency constraint, or result
measurement was changed during release preparation. Existing modifications and
the pre-existing deletion of an older compiled manuscript were preserved. No
commit, push, or history rewrite was performed.

## Housekeeping

This is a separate release copy. The original working repository, its user changes,
and all historical outputs remain untouched. Hash checks confirmed preservation
of1,269 pre-existing inventoried files after reverting the initial in-place edits.

Session directories, credentials, local instructions, Python caches, build outputs,
and logs are excluded from this copy or ignored. No `.agent/`, `.codex/`, or root
`AGENTS.md` is bundled. The standalone example, research source, tests, configurations,
third-party licenses, lightweight results, and research histories are retained.
The large historical figure collection is not duplicated; reporters can generate
plots, and the paper figures used in the overview are included.

Operational recovery controllers110–112, the fixed-path task103 launch shell,
and finalizer097 are omitted: they manage old process IDs or local instruction
files and are not experiment entry points. The source algorithms and documented
experiment controllers are unchanged. `state.yaml` is retained for controllers
that use it; its historical content is not an experiment result.

The frozen task snapshot in results100 retains its original tool-name heading
as provenance; it is not a public setup instruction. Scientific provenance is
not rewritten merely to remove a keyword. Third-party source, author metadata,
acknowledgments, and licenses are preserved.

## Before publishing

- Add an owner-approved project license; third-party licenses do not license the
  project's own code automatically.
- Package and verify access to the frozen external assets needed by the full image
  and historical tabular workflows. The portable example does not require those
  parent-run artifacts, but full research workflows do.
- Review the publication set for double-blind compliance. Original provenance and
  server-specific scripts contain local paths; the new overview is anonymous,
  but the complete historical working tree is not asserted to be anonymized.
  Preserve original provenance when preparing any separate anonymous distribution.
- Include the new documentation, figures, manuscript, and intended source files
  in the chosen publication set. This folder has no new Git history or staged files. `git archive` exports
  committed content, not an untracked release folder.

See [verification](verification.md) for commands actually checked. Fresh-machine
installation, external downloads, GPU execution, and expensive experiments were
not rerun for this documentation update.
