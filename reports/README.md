# Recorded verification evidence

`original-experiment.json` records the five fresh-process reloads from the original
September 10, 2026 macOS experiment: four legacy same-/cross-Python reloads and one
current-release reload. They total 200 full-array comparisons with maximum error 0.
It records the package versions and the actual NetCDF-only failure for each case.

`local-test-suites.json` records the automated test suites executed before publication.
These reran training and reload against the documented code, rather than merely
checking the original result files.

Recorded results describe the specific runs, not a promise that every package
upgrade or operating system works. The repository's Actions badge links to live
CI results. Workflow artifacts contain the generated models, references, individual
comparison records, and subprocess logs for those runs.

Reproduce locally using the commands in the root README and `tests/README.md`.
To regenerate `local-test-suites.json`, redirect the three full suite runs to
`tests-py311.log`, `tests-py313.log`, and `tests-current.log`, then run
`python record_results.py`. The recorder rejects failed or incomplete runs.
The synthetic model and test inputs are generated in code; no work data or secrets
are used. Large binary model files and virtual environments are not committed.
