# API and artifact reference

[![CI](https://github.com/vitoriomexas311/bart-save-load/actions/workflows/tests.yml/badge.svg)](https://github.com/vitoriomexas311/bart-save-load/actions/workflows/tests.yml)

Save a trained **scalar-output PyMC BART** variable to one file, then load a reusable
predictor in another Python process. No retraining, original training data, trace,
or reconstructed PyMC model is needed for prediction.

The runtime is exactly two files: `bart_persistence/save.py` and
`bart_persistence/load.py`. The directory is a Python namespace package, so it
needs no `__init__.py`. Tests, packaging metadata and the notebook are development
material, not runtime modules.

## Install

Choose the extra matching your training environment. Do not install both.
From [this repository](https://github.com/vitoriomexas311/bart-save-load):

```sh
python -m pip install '.[current]'   # BART 0.13.1 / PyMC 6.3.2 (Python >=3.12)
# OR
python -m pip install '.[legacy]'    # BART 0.11.0 / PyMC 5.25.1 (Python >=3.11)
```

In an environment that already has either supported stack, `pip install .` adds
only this package and NumPy. It deliberately does not upgrade your work's PyMC.
Installing the base package into an empty environment is insufficient: choose a
stack extra. Releases other than these two exact BART/PyMC pairs are rejected.
The package is not published to PyPI. Build an installable wheel with:

```sh
python -m pip install build
python -m build
# On your serving machine, using the wheel you built:
python -m pip install 'bart_persistence-0.1.0-py3-none-any.whl[current]'
```

Use a dedicated environment. After training, retain `python -m pip freeze >
training-requirements.txt` alongside your deployment configuration and install that
same lock in serving. Exact versions of BART, PyMC, PyTensor and NumPy (plus bartrs
for the current stack) must match the artifact. Python and OS are not checked;
keep them the same too for deployments. This is not an upgrade/migration format.

## Training script: save

```python
import numpy as np
import pymc as pm
import pymc_bart as pmb
from bart_persistence.save import save_bart

# Replace this synthetic data with your prepared numeric feature matrix and target.
rng = np.random.default_rng(42)
X = rng.normal(size=(100, 2))
y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, 100)

if __name__ == "__main__":  # Required for portable multiprocessing in scripts.
    with pm.Model() as model:
        x = pm.Data("X", X)
        mu = pmb.BART("mu", x, y, m=20)
        pm.Normal("observed", mu, 0.1, observed=y)
        trace = pm.sample(draws=100, tune=100, chains=2, cores=2, random_seed=42)
    save_bart(model["mu"], "model.bart",
              expected_draws=trace.posterior.sizes["chain"] * trace.posterior.sizes["draw"])
```

Those short chains illustrate the API; choose sampling settings and check
convergence appropriate to your actual model. For BART 0.13.1 use parallel chains (`cores=chains`) or a single chain: in local
testing sequential two-chain sampling retained only the last chain’s tree history
in the upstream variable, despite two chains in the trace. The saver can only
persist histories that BART retains.

Save only after sampling finishes, while the trained BART variable still exists. Pass the BART random variable itself,
not `trace`, the model, or a deterministic inverse-link expression.

`save_bart(rv, path, *, expected_draws=None)` returns `None`.
When supplied, `expected_draws` must be a positive Python integer matching the
number of retained ensembles. Use the trace chain count times draw count, as above.
A mismatch raises `ValueError` before writing; this catches missing upstream
histories rather than silently saving an incomplete fit. If omitted, all available
histories are saved without comparing them to a trace.

The parent directory must exist. Existing files raise `FileExistsError` instead of being overwritten; use a new versioned
filename for each fit. Serialization completes in memory before opening the file.
The file is staged in a temporary directory beside the destination, flushed with
`fsync`, then published with an exclusive hard link. Readers see a complete artifact
and existing files are never replaced. This requires a local filesystem with hard-link
support (such as APFS, ext4 or NTFS); unsupported filesystems raise `OSError`.
Normal failures clean up temporary files. A killed process can leave a hidden
`.bart-*` staging directory; a published artifact remains complete. This is an
atomic-publication guarantee, not a guarantee against disk failure or power loss.
Saving briefly needs memory for both tree state and serialized bytes.

## Separate prediction script: load

```python
import numpy as np
from bart_persistence.load import load_bart

predictor = load_bart("model.bart")  # Load once at application startup.
X_new = np.array([[0.2, 0.8], [-0.5, 0.1]])
samples = predictor.predict(X_new, draws=1000, seed=42)
mean = samples.mean(axis=0)
interval = np.quantile(samples, [0.025, 0.975], axis=0)
```

Pass `predictor` to functions or keep it in your service object. Different loaded
predictors have independent tree state. Loading reconstructs the current version's
Rust prediction samplers once, rather than doing so for every request. PyMC and
BART remain installed dependencies and are imported, but no model is compiled or
sampled. This is not a dependency-free tree engine.

`predict(X, *, draws=500, seed=None)` returns a NumPy array with shape
**(requested draws, input rows)**. Draws are sampled **with replacement** from all
retained posterior ensembles pooled across chains. `predictor.n_draws` reports the
number retained; `predictor.n_features` reports the expected column count. A fixed
integer seed repeats results for the same artifact and inputs. `None` uses fresh
randomness. Requesting more draws resamples the saved posterior; it does not fit
more trees. Batch large datasets to control output memory (approximately
`8 * draws * rows` bytes, plus prediction working memory).

X must be a nonempty finite numeric 2D matrix with the training column count.
**Preserve column order, encodings, units, imputation and scaling yourself.** The
artifact stores the column count, not names or preprocessing. DataFrames convert
to arrays in their existing order; columns are not matched by name. Invalid shape,
nonfinite input or a nonpositive/noninteger draw count raises `ValueError`.

## What is saved, and why

A numerical trace contains BART's evaluated values at the training rows, but not
the reusable tree state needed to evaluate new rows. This package saves that state
separately. The upstream [persistence discussion](https://github.com/pymc-devs/pymc-bart/issues/123)
and [storage changes](https://github.com/pymc-devs/pymc-bart/blob/main/CHANGELOG.md)
explain the background.

One `.bart` file contains:

1. A UTF-8 JSON line with format ID `bart-persistence/1`, exact dependency versions,
   feature count and a SHA-256 payload digest.
2. A protocol-5 pickle payload with the complete retained tree collection and tree
   count `m`. BART 0.11 stores posterior forests. BART 0.13.1 stores compressed
   per-chain histories and also needs `n_outputs` to reconstruct its samplers.

The saver turns the process-backed tree collection into a normal list; it does
not pickle the multiprocessing manager, PyMC model, trace or training matrix.
The loader checks versions and verifies the payload digest before unpickling,
then constructs an independent `BARTPredictor`. It does not modify global BART state or rely on an operator-ID cache.
It uses version-specific private upstream prediction APIs, which is why compatibility
is deliberately narrow. Old artifacts produced by the previous demo scripts are
not this format; save again from a live fitted variable using this API.

**Only load files from trusted sources. Pickle can execute arbitrary code.** A
version check is a compatibility guard, not authentication or safe deserialization.
The unkeyed checksum detects accidental corruption; it does not authenticate a file.
Early format-1 files without a digest remain readable but have no integrity check.
A present but incorrect digest raises `ValueError` before unpickling.
Corrupt/truncated files raise their native JSON, pickle, key or I/O exceptions.

## Scope

Predictions are draws of the **latent BART function**, on the scale on which BART
was trained. For a logistic model apply the sigmoid to *each draw*, then summarize
those probabilities. For a log-link model exponentiate each draw. This package
does not add likelihood noise, infer other parameters or preserve joint draw
alignment with a separate trace. Consequently these intervals describe the latent
function, not noisy future observations. Keep your trace separately for diagnostics
or downstream analysis if needed; it is not an argument to this API.

The supported contract is one scalar-output BART variable with a numeric feature
matrix. Multiple-output BART is explicitly rejected. Categorical split rules,
linear leaves, missing-value prediction, continued training, concurrent use of a
single predictor, and cross-version migration are outside the tested contract.

## Notebook and tests

`bart_persistence_walkthrough.ipynb` runs the same save/load workflow and verifies
predictions in a child interpreter. Explanations live in this Markdown reference;
notebook Markdown is limited to short step labels. Select the `bart-persistence`
kernel when opening it interactively.

```sh
python -m pip install -e '.[current,test,notebook]'
python -m ipykernel install --sys-prefix --name bart-persistence
python -m jupyterlab bart_persistence_walkthrough.ipynb
# Or execute all cells automatically; keep generated output out of source control:
python -m nbconvert --to notebook --execute bart_persistence_walkthrough.ipynb --ExecutePreprocessor.kernel_name=bart-persistence --output-dir artifacts
```

Run coverage across **both supported stacks**, since each has a different loader
branch. In two separate environments, installed with `.[current,test]` and
`.[legacy,test]`, run from the repository root:

```sh
COVERAGE_FILE=.coverage.current .venv-current/bin/python -m pytest --cov-fail-under=0
COVERAGE_FILE=.coverage.legacy .venv-legacy/bin/python -m pytest --cov-fail-under=0
.venv-current/bin/python -m coverage combine
.venv-current/bin/python -m coverage report --fail-under=100
```

100% means **statement and branch coverage of the two runtime modules**, combined
across supported versions. It does not mean every possible BART model or third-party
code path is tested. Tests include actual two-chain fits, complete array equality
against the live upstream sampler over several batch sizes, fresh-process loading
with training/model creation forbidden, independent predictors, and error cases.
CI tests legacy BART on Python 3.11, current BART on Python 3.12 and 3.13,
and current BART on Linux, macOS and Windows,
enforces the combined 100% threshold, builds the wheel and source distribution,
and executes the notebook outside the checkout against the installed wheel.
Each run publishes the wheel and executed notebook as downloadable artifacts.
A single-stack `python -m pytest` reports coverage without failing on the other
stack’s unexecuted branches; the separate combined CI job owns that gate. Example notebooks
and test code are not counted as production coverage.
