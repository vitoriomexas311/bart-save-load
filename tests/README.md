# What the tests verify

Run with the Python environment for either published dependency lock:

```sh
python -m unittest discover -s tests -v
```

No extra test framework is required; these tests use the standard library's
`unittest`, plus the same numerical dependencies as the examples.

## Fast unit tests

`test_checks.py` verifies nine independent cases: accepted matching package versions,
rejected incompatible or missing versions, exact equality, measurable tiny roundoff,
changed values, broadcastable shape errors, non-finite values, and empty arrays.

`test_trees.py` constructs an actual two-leaf Python tree without running MCMC.
For inputs on opposite sides of its split, adding a second constant tree must yield
`[3, -2]`. A separate retained forest must yield `[7, 7]`. Three tests verify:

- Every saved ensemble is evaluated and its trees are summed correctly.
- Seeded posterior sampling selects complete ensembles with the expected shape.
- Both serializers preserve the real split nodes and leaf values.

These three tests intentionally skip on BART 0.13.1 because that release no longer
uses the Python `Tree` API. Its real tree histories are tested end to end below.

## Integration test

`test_runtime.py` adds four unit tests for version-specific compiler cache setup.
It checks version separation, preserving existing flags, honoring an explicit
cache directory, and idempotence. This prevents a real PyTensor cache collision
found when the two stacks were tested concurrently.

`test_integration.py` invokes the selected example through `subprocess.run`:

1. Start a Python process, train two chains, save all 160 retained ensembles, and exit.
2. Start another Python process, load the files, and predict without calling `pm.sample`.
3. Require both pickle formats, all five data cases, and all four prediction modes
   to pass: 40 full-array comparisons, not 40 individual scalar checks.

The reconstructed model first runs a NetCDF-only negative control. After tree
restoration it checks random tree draws, every ensemble, `mu`, and noisy `y`.
Shape equality is required before numerical comparison, and NaNs never count as equal.

Generated artifacts and complete subprocess logs remain in `.test-artifacts/`,
which Git ignores. Set `BART_TEST_OUTPUT` to choose another directory. CI uploads
these files as workflow artifacts, including logs when a test fails.

Expected suite counts: **17 tests on BART 0.11.0**, or **14 executed + 3 skipped on
BART 0.13.1**. The tests run short chains to verify persistence; they do not certify
posterior convergence or general predictive quality.

## Cross-environment experiment and CI

`run_matrix.py` additionally loads each of two independently trained legacy
archives in both Python 3.11 and Python 3.13. It tests migration of the *same*
saved files across interpreters while keeping the core package versions fixed.

The GitHub Actions matrix runs full fresh-process tests independently on Linux,
macOS, and Windows with the exact selected dependency lock. It does not claim
cross-OS transfer of a single artifact or cross-BART-version migration.
