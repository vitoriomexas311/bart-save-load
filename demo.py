"""Demonstrate persistence with pymc-bart 0.11.0 and PyMC 5.25.1.

Run ``python demo.py train artifacts/example`` and let that process exit.
Then run ``python demo.py reload artifacts/example`` in a NEW process.
The second process has no access to the first one's in-memory tree manager.

Only synthetic data is used. The short chains are for a persistence experiment,
not a claim of reliable convergence for an arbitrary production model.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import pickle
import platform

# This must happen before importing the numerical stack; see runtime.py.
from runtime import configure_compiler_cache
configure_compiler_cache()

import arviz as az
import cloudpickle
import numpy as np
import pymc as pm
import pymc_bart as pmb
from pymc_bart.utils import _sample_posterior
from checks import check_versions, compare_predictions

PACKAGES = ['pymc-bart', 'pymc', 'pytensor', 'numpy', 'arviz', 'cloudpickle']
SEED = 281  # Reset this prediction seed to compare identical sampled draws.
DRAWS = 80  # Retained posterior draws PER CHAIN; tuning draws are discarded.
M = 20  # Each retained posterior draw contains this many additive trees.


def versions():
    """Record the interpreter, OS, and packages that produced an artifact."""
    return {'python': platform.python_version(), 'platform': platform.platform(),
            **{p: md.version(p) for p in PACKAGES}}


def model_for(X, y):
    """Build the same two-feature regression model in training and serving.

    ``pm.Data`` lets us replace the covariates without fitting again. The shape
    of the likelihood follows the new row count. Fixing sigma isolates tree
    persistence from the separate issue of pairing trees with inferred noise.
    """
    with pm.Model() as model:
        x = pm.Data('X', X)
        mu = pmb.BART('mu', x, y, m=M)
        pm.Normal('y', mu, sigma=0.15, observed=y, shape=x.shape[0])
    return model


def tree_draws(trees, X):
    """Sample BART means directly from saved forests, without a PyMC model.

    The private 0.11 helper takes a list of forests and returns an array with
    shape (128 posterior samples, number of input rows, 1 output). Its signature
    changed in later releases; see demo_current.py for the 0.13.1 equivalent.
    """
    return _sample_posterior(trees, X, np.random.default_rng(SEED), size=128, shape=1)


def every_forest(trees, X):
    """Evaluate EVERY saved ensemble to catch missing or corrupted draws.

    A 0.11 archive is indexed by [draw][output dimension][tree]. Our model has
    one output, so forest[0] selects it. BART adds tree predictions; it does not
    average them. A random subset alone could miss a damaged ensemble.
    """
    return np.stack([sum((t.predict(X, shape=1) for t in forest[0]))
                     for forest in trees])


def posterior(model, trace, X):
    """Generate latent means AND noisy observations through the PyMC 5 API.

    Resetting the random seed makes predictions before and after serialization
    comparable element by element, instead of merely comparing their means.
    """
    with model:
        pm.set_data({'X': X})
        out = pm.sample_posterior_predictive(
            trace, var_names=['mu', 'y'], random_seed=SEED,
            predictions=True, progressbar=False)
    return {v: out.predictions[v].values for v in ['mu', 'y']}


def train(folder, chains, cores):
    """Fit once, archive all retained trees, and save reference predictions.

    ``chains=2, cores=2`` also tests whether trees from parallel workers survive
    export. We expect draws*chains complete forests, not just the last forest.
    """
    if md.version('pymc-bart') != '0.11.0':
        raise ValueError('demo.py requires pymc-bart 0.11.0; use its dependency lock')
    folder.mkdir(parents=True, exist_ok=True)
    # A known nonlinear signal gives us a sanity check that the model learned.
    rng = np.random.default_rng(48)
    X = rng.uniform(-1, 1, (120, 2))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, .15, 120)
    # Include singleton, smaller, equal-size-but-different, and larger batches.
    cases = {f'new_{n}': rng.uniform(-1, 1, (n, 2)) for n in [1, 17, 120, 175]}
    cases['training'] = X.copy()
    np.savez(folder / 'inputs.npz', X=X, target=y, **cases)
    model = model_for(X, y)
    with model:
        trace = pm.sample(draws=DRAWS, tune=100, chains=chains, cores=cores,
                          random_seed=51, progressbar=False,
                          compute_convergence_checks=False)
    # Critical step: materialize the Manager proxy as a regular Python list.
    # Pickling the entire model would retain unnecessary multiprocessing state.
    trees = list(model['mu'].owner.op.all_trees)
    assert len(trees) == chains * DRAWS, (len(trees), chains * DRAWS)
    assert all(len(f[0]) == M for f in trees)
    # NetCDF preserves numerical posterior values, but does not contain trees.
    trace.to_netcdf(folder / 'trace.nc')
    for serializer in [pickle, cloudpickle]:
        with (folder / f'{serializer.__name__}.pkl').open('wb') as f:
            serializer.dump(trees, f, protocol=5)
    # References are generated while the original trained model is still alive.
    refs = {}
    for name, values in cases.items():
        refs[name + '_trees'] = tree_draws(trees, values)
        refs[name + '_all'] = every_forest(trees, values)
        for variable, predictions in posterior(model, trace, values).items():
            refs[name + '_' + variable] = predictions
    np.savez_compressed(folder / 'reference.npz', **refs)
    # This sanity check is separate from exact serialization equivalence.
    test = cases['new_175']
    truth = np.sin(3 * test[:, 0]) + .5 * test[:, 1]
    rmse = float(np.sqrt(np.mean((refs['new_175_trees'].mean(axis=(0, 2)) - truth)**2)))
    metadata = {'versions': versions(), 'training_pid': os.getpid(),
                'chains': chains, 'cores': cores, 'draws': DRAWS, 'trees_per_ensemble': M,
                'retained_ensembles': len(trees), 'features': ['x0', 'x1'],
                'response': 'constant', 'likelihood': 'Normal(mu, sigma=0.15)',
                'test_rmse': rmse, 'baseline_rmse': float(np.sqrt(np.mean((y.mean()-truth)**2)))}
    assert metadata['test_rmse'] < metadata['baseline_rmse']
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'trained': str(folder), **metadata}), flush=True)


def reload(folder, label):
    """Restore a model in a fresh process and verify both prediction routes.

    First demonstrate that NetCDF alone is insufficient. Then restore each
    archive format and compare five input batches against the original model.
    Only unpickle artifacts that you trust; pickle can execute arbitrary code.
    """
    metadata = json.loads((folder / 'metadata.json').read_text())
    # Fail before unpickling if this is a different BART/numerical package stack.
    check_versions(metadata['versions'], versions(), PACKAGES)
    data = np.load(folder / 'inputs.npz')
    refs = np.load(folder / 'reference.npz')
    trace = az.from_netcdf(folder / 'trace.nc')
    model = model_for(data['X'], data['target'])
    assert len(model['mu'].owner.op.all_trees) == 0
    # Prevent any accidental retraining in this process.
    def forbidden(*args, **kwargs):
        """Fail loudly if reload ever accidentally attempts posterior training."""
        raise AssertionError('Reload must never call pm.sample')
    pm.sample = forbidden
    results = {'versions': versions(), 'reload_pid': os.getpid(), 'training_pid': metadata['training_pid'],
               'checks': [], 'max_absolute_difference': 0.0}
    # Negative control: a rebuilt BART RV starts with an EMPTY tree collection.
    try:
        bad = posterior(model, trace, data['new_17'])
        matched = bad['mu'].shape == refs['new_17_mu'].shape and np.allclose(bad['mu'], refs['new_17_mu'])
        assert not matched, 'Trace-only control unexpectedly matched'
        results['trace_only'] = 'predictions differ from trained model'
    except (ValueError, TypeError) as exc:
        results['trace_only'] = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
    for serializer in [pickle, cloudpickle]:
        with (folder / f'{serializer.__name__}.pkl').open('rb') as f:
            trees = serializer.load(f)
        assert len(trees) == metadata['retained_ensembles']
        # Extend the existing class-backed Manager list. Assigning an instance
        # attribute alone is wrong: BARTRV.rng_fn accesses class state.
        model['mu'].owner.op.all_trees[:] = trees
        for name in [k for k in data.files if k not in ['X', 'target']]:
            actual = {'trees': tree_draws(trees, data[name]),
                      'all': every_forest(trees, data[name]),
                      **posterior(model, trace, data[name])}
            for kind, values in actual.items():
                reference = refs[name + '_' + kind]
                error = compare_predictions(values, reference)
                results['max_absolute_difference'] = max(results['max_absolute_difference'], error)
                results['checks'].append({'format': serializer.__name__, 'case': name,
                                         'kind': kind, 'shape': list(values.shape), 'max_error': error})
    results['passed'] = True
    (folder / f'result-{label}.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({k:v for k,v in results.items() if k != 'checks'}), flush=True)


# The main guard is essential: macOS/Windows multiprocessing imports this file
# in child processes. Importing it must not launch training recursively.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'reload'])
    parser.add_argument('folder', type=Path)
    parser.add_argument('--chains', type=int, default=1)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--label', default='local')
    args = parser.parse_args()
    if args.action == 'train':
        train(args.folder, args.chains, args.cores)
    else:
        reload(args.folder, args.label)
