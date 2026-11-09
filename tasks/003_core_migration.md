# Core experiment migration

Copy the HELOC/DIABETES core workflows and their internal dependencies from
`/home/xu.zhao/robust_ce/` into this repository. Preserve historical source and
raw results. Make the imported code authoritative for future core development.

Acceptance: local imports (including small baseline source dependencies), focused
regression tests, read-only frozen-instance compatibility, provenance hashes,
explicit new output paths, and documented commands. No scientific reruns or
manuscript changes. This does not resolve the missing specification in task 002.
