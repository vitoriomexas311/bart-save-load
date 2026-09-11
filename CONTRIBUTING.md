# Contributing

Keep this utility small: two runtime modules, one artifact per BART variable and
no extra serving framework. Changes to private upstream APIs need real round-trip
tests on every claimed BART version; do not weaken compatibility checks to make an
untested release load.

## Develop and test

```sh
git clone https://github.com/vitoriomexas311/bart-save-load.git
cd bart-save-load
python -m venv .venv
# Activate the environment using the command for your shell, then:
python -m pip install -e '.[current,test,dev,notebook]'
python -m pytest
python -m ruff check bart_persistence tests
python -m build
python -m twine check --strict dist/*
```

Use a separate environment with `.[legacy,test,dev]` for the legacy stack.
Single-stack pytest reports coverage; CI combines the two implementations and
requires 100% statement and branch coverage. See the [reference](docs/reference.md)
for local combination commands. Do not exclude production branches merely to
improve the percentage. Test observable behavior, including failures.

Open a pull request with a concrete problem, resulting behavior and validation.
Keep commits focused: a function or method and its corresponding tests should
land together. Include a minimal synthetic reproducer for persistence bugs; never
attach confidential datasets, credentials or untrusted pickles. Record user-visible
changes under Unreleased in CHANGELOG.md.

## Compatibility policy

The public API is `save_bart`, `load_bart` and `BARTPredictor.predict`, with
`n_draws` and `n_features` for inspection. The predictor constructor and upstream
private helpers are implementation details. Version 0.x is pre-stable: breaking
API or artifact changes require a minor version bump and documented migration.
Patch releases preserve the documented API and readable artifact formats.

## Release checklist

1. Resolve licensing and redistribution approval before publishing a public release.
2. Update version metadata and the changelog; review API and artifact compatibility.
3. Require the quality, platform tests and combined coverage jobs to pass on the
   exact commit to be released. CI tests the installed wheel outside the checkout.
4. Download the wheel and source distribution from that CI run; inspect their
   metadata and SHA256SUMS. Preserve the tested dependencies recorded with them.
5. Tag the verified commit, create release notes with the support matrix and known
   limitations, and attach those exact artifacts. Do not rebuild unrelated bytes.
6. If PyPI publication is later enabled, configure trusted publishing explicitly.
   This repository currently contains no package-index credentials or auto-publisher.

No release or compatibility expansion should be inferred from a badge alone.
