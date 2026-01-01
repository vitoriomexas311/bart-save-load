# PyMC BART persistence: save, restart, predict

[![Persistence tests](https://github.com/vitoriomexas311/pymc-bart-persistence-example/actions/workflows/tests.yml/badge.svg)](https://github.com/vitoriomexas311/pymc-bart-persistence-example/actions/workflows/tests.yml)

**A small, extensively commented example showing how to save the learned BART
trees, exit Python, and predict in a fresh process without retraining.**

Saving a numerical posterior to `trace.nc` is insufficient for BART. This example
saves a separate pickle containing all retained tree state, then demonstrates both
direct tree predictions and `pm.sample_posterior_predictive` after restoration.

There are two explicit examples because BART changed its internal storage format:

| Example | Pinned stack | What the tree archive contains |
|---|---|---|
| [demo.py](demo.py) | BART 0.11.0 / PyMC 5.25.1 | Every retained forest |
| [demo_current.py](demo_current.py) | BART 0.13.1 / PyMC 6.3.2 | Per-chain histories plus `n_outputs` |

## Try the current example

```sh
git clone https://github.com/vitoriomexas311/pymc-bart-persistence-example.git
cd pymc-bart-persistence-example
python3.13 -m venv .venv-current
.venv-current/bin/python -m pip install -r requirements-current.lock.txt

# This process trains, writes trace.nc and the tree archives, then exits.
.venv-current/bin/python demo_current.py train artifacts/current

# This is a NEW Python process. It is forbidden from calling pm.sample().
.venv-current/bin/python demo_current.py reload artifacts/current

# Run both unit tests and another real training/reload experiment.
.venv-current/bin/python -m unittest discover -s tests -v
```

On Windows, create the environment with `py -3.13 -m venv .venv-current` and
replace `.venv-current/bin/python` with `.venv-current\Scripts\python.exe`.
The two scripts use a main guard so multiprocessing can safely import them.
They also select a version-specific `.pytensor-cache/` directory before importing
PyMC. This avoids compiled-extension collisions when both stacks run concurrently;
virtual environments alone did not isolate that cache in our test.

On a successful reload, the JSON output includes `"passed": true`, 40 detailed
comparisons in the result file, and the measured `max_absolute_difference`.
Our recorded local runs returned **0.0**. A fresh training run may learn different
trees; the invariant is that its own before/after predictions agree.

## Read the code as a walkthrough

1. **Build a model:** `model_for` in [demo.py](demo.py) uses two synthetic features,
   a sum of 20 trees, and a Normal likelihood with fixed observation noise.
2. **Train and save:** the `train` phase saves numerical posterior data to NetCDF
   and materializes `all_trees` as a regular list before pickling it.
3. **Record ground truth for the experiment:** predict while the original trained
   model is alive, including predictions from every retained ensemble.
4. **Restart:** each reload is a separate Python invocation, not another function
   call that can still see the training model's memory.
5. **Demonstrate the failure:** load only NetCDF into a rebuilt model and try a new
   row count. This negative control must fail or disagree with the trained model.
6. **Restore the trees:** populate the rebuilt model's class-backed collection;
   the current implementation also restores the output dimension and clears its cache.
7. **Prove equivalence:** compare complete arrays of latent means and noisy outcomes,
   along with direct tree predictions. A version, shape, or value mismatch fails the run.

Every function is documented, with inline comments at the serialization and
prediction steps. [tests/README.md](tests/README.md) explains the automated checks;
[reports/README.md](reports/README.md) explains the committed verification evidence.

This repository contains executable experiments, exact dependency locks,
and machine-readable result summaries. Trained binary artifacts are generated locally
and excluded from Git. Training exits before each reload starts. The reload process
reconstructs the PyMC model, restores the tree state, loads NetCDF, and predicts without training.

## Verified PyMC 5 results

Stack: pymc-bart 0.11.0, PyMC 5.25.1, PyTensor 2.31.7, NumPy 2.2.6, ArviZ 0.22.0.
Host: macOS 26.3, Apple Silicon. Python environments have separate dependency installations;
the exact full sets are in `requirements-py311.lock.txt` and `requirements-py313.lock.txt`.

| Training | Loading in a new process | Result |
|---|---|---|
| Python 3.13.2, one chain | Python 3.13.2 | PASS, max difference 0 |
| Python 3.13.2, one chain | Python 3.11.4 | PASS, max difference 0 |
| Python 3.11.4, two parallel chains | Python 3.11.4 | PASS, max difference 0 |
| Python 3.11.4, two parallel chains | Python 3.13.2 | PASS, max difference 0 |

Both standard `pickle` and `cloudpickle` passed. Each reload performs 40 numerical comparisons:
two serializers × five input cases × four outputs. The input cases are new batches of 1, 17,
120, and 175 rows, plus the original 120 training rows. Four outputs are random posterior tree
draws, predictions from EVERY retained ensemble, `mu`, and noisy `y` from PyMC's standard
`sample_posterior_predictive`. Random seeds are fixed for before/after comparisons.

The trace-only negative control fails on a 17-row new batch with a 17-versus-120 shape mismatch.
Reload replaces `pm.sample` with a function that raises immediately, preventing retraining.
Results record separate training/loading process IDs.

The model learns `sin(3*x0) + 0.5*x1` with Normal observation noise of fixed sigma 0.15.
There are 20 trees per ensemble and 80 retained draws per chain after 100 tuning steps.
All 80 single-chain or 160 two-chain posterior ensembles are preserved, not only a final forest.
Warmup and rejected proposals are intentionally not archived. Holdout mean-function RMSE was
0.137 and 0.126 respectively, versus 0.778 for the constant training-mean baseline.

## Reproduce

For a single available Python environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python demo.py train artifacts/example --chains 2 --cores 2
.venv/bin/python demo.py reload artifacts/example --label fresh
```

These are deliberately separate Python invocations. To repeat the two-environment matrix:

```sh
python3.13 -m venv .venv
python3.11 -m venv .venv-py311
.venv/bin/pip install -r requirements-py313.lock.txt
.venv-py311/bin/pip install -r requirements-py311.lock.txt
python3 run_matrix.py
```

Paths to alternate Python installations can be supplied with `--python-a` and `--python-b`.
`--skip-training` reuses existing artifacts. A generated `matrix.json` records process exit codes;
`artifacts/*/result-*.json` contains individual comparisons. Logs are under `logs/`.

## Verified current-release results

Python 3.13.2 / pymc-bart 0.13.1 / PyMC 6.3.2 / PyTensor 3.3.1 / bartrs 0.4.0 /
NumPy 2.5.3 / ArviZ 1.3.0: two parallel training chains, followed by a separate reload
process, PASS. Both serializers and all 40 comparisons passed with maximum absolute
difference **0.0**. NetCDF alone failed with the same 17-versus-120 shape error.
Together with the four PyMC 5 runs, this is **200 passing numerical comparisons**.

This release stores two chain histories that reconstruct all 160 retained ensembles.
`demo_current.py` saves the plain history list AND `n_outputs`. After model reconstruction:

```python
op = model['mu'].owner.op
op.all_trees[:] = archive['trees']
type(op).n_outputs = archive['n_outputs']
# Clear the private sampler cache if replacing an already-used archive.
from pymc_bart.utils import _posterior_sampler_cache
_posterior_sampler_cache.clear()
with model:
    pm.set_data({'X': X_new})
    predictions = pm.sample_posterior_predictive(
        trace, var_names=['mu', 'y'], sample_vars=['mu', 'y'],
        predictions=True, backend='c')
```

Two additional compatibility details were discovered in actual runs:

1. PyMC 6 needs explicit `sample_vars` to regenerate BART values on changed covariates.
   `var_names` alone left `mu` frozen and generated an implicit-freeze warning.
2. Even with that fixed, the default Numba backend failed during this batch prediction
   test with `Vectorized input 0 has an incompatible shape in axis 0`. Using the public
   `backend='c'` option passed all tested batch sizes. This is a prediction/backend issue
   encountered before serialization, not evidence of corrupt saved trees.

NetCDF is loaded with `xarray.open_datatree` for PyMC 6's DataTree output. Reproduce:

```sh
python3.13 -m venv .venv-current
.venv-current/bin/pip install -r requirements-current.lock.txt
.venv-current/bin/python demo_current.py train artifacts/current
.venv-current/bin/python demo_current.py reload artifacts/current
```

The two version families were trained separately. Cross-BART-version migration was not
attempted or claimed; the cross-Python tests used BART 0.11.0 on both sides.

## Minimal integration for the tested PyMC 5 stack

```python
# In the training process, after pm.sample():
import pickle
trace.to_netcdf('trace.nc')
with open('trees.pkl', 'wb') as f:
    pickle.dump(list(model['mu'].owner.op.all_trees), f, protocol=5)

# In a new process, recreate the same model specification first:
import arviz as az
trace = az.from_netcdf('trace.nc')
with open('trees.pkl', 'rb') as f:
    trees = pickle.load(f)
model['mu'].owner.op.all_trees[:] = trees
with model:
    pm.set_data({'X': X_new})
    predictions = pm.sample_posterior_predictive(
        trace, var_names=['mu', 'y'], predictions=True)
```

Mutating the existing tree list matters: the prediction implementation accesses class-backed
state, so setting an instance-only attribute can fail to restore the trees used by predictions.
The direct `_sample_posterior` helper is exercised too, for serving BART means without a rebuilt
PyMC model. It is private and version-specific.

## Scope and production handoff

- These are persistence tests, not convergence certification; chains are deliberately short.
- Exact comparisons validate reload equivalence for this model. The likelihood uses fixed noise;
  this does not validate tree-draw alignment with other inferred parameters in complex models.
- Preserve feature order, transforms, model specification, output dimension and dependency locks.
  The synthetic inputs/reference files are test fixtures, not a requirement to serve tree-only means.
- Only load trusted pickle files. Pin the training/serving stack. Cross-Python success here is not
  a guarantee for arbitrary Python, NumPy, or BART upgrades.
- The original recorded experiment ran on macOS. The GitHub Actions workflow tests Linux,
  macOS and Windows; consult its actual run status for those results, rather than treating
  a configured matrix as a pass. Containers, categorical/multi-output BART, and linear leaves
  are outside this example. Docker was unavailable during the original local experiment.
- Current BART has changed its storage schema. Do not load 0.11 artifacts in newer BART versions
  by disabling the version check. `demo_current.py` exercises the 0.13.1 schema separately.

Background: [upstream issue #123](https://github.com/pymc-devs/pymc-bart/issues/123),
[original discussion](https://discourse.pymc.io/t/save-and-load-a-bart-model/13135).
