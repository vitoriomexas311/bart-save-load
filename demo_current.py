"""Repeat the experiment with pymc-bart 0.13.1 / PyMC 6.3.2.

The newer Rust-backed implementation stores compressed per-chain histories.
The archive must include BOTH those histories and the number of outputs.
It cannot be exchanged with demo.py's 0.11 forest-object archives.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import pickle

# PyMC 5 and 6 must not race over a shared compiled-extension cache.
from runtime import configure_compiler_cache
configure_compiler_cache()

import cloudpickle
import numpy as np
import pymc as pm
import xarray as xr
from pymc_bart.utils import _get_posterior_sampler, _posterior_sampler_cache, _sample_posterior

from demo import DRAWS, SEED, model_for, versions
from checks import check_versions, compare_predictions


def posterior(model, trace, X):
    """Predict with the two settings needed by the tested PyMC 6 stack.

    ``sample_vars`` explicitly regenerates mu for the NEW inputs; ``var_names``
    alone just selects outputs and can leave mu frozen to training values.
    ``backend='c'`` avoids a shape error found with the default Numba backend.
    """
    with model:
        pm.set_data({'X': X})
        out = pm.sample_posterior_predictive(
            trace, var_names=['mu', 'y'], sample_vars=['mu', 'y'],
            random_seed=SEED, predictions=True, progressbar=False, backend='c')
    return {v: out.predictions[v].values for v in ['mu', 'y']}


def outputs(model, trace, X):
    """Return random tree draws, every retained ensemble, and PyMC predictions.

    _get_posterior_sampler reconstructs a prediction-only Rust sampler from
    the saved chain histories. n_draws counts recoverable posterior ensembles,
    unlike len(all_trees), which now counts chains rather than posterior draws.
    """
    sampler = _get_posterior_sampler(model['mu'].owner.op)
    return {'trees': _sample_posterior(sampler, X, np.random.default_rng(SEED), size=128),
            'all': sampler.sample_posterior(X, list(range(sampler.n_draws)), None),
            **posterior(model, trace, X)}


def main(action, folder):
    """Run exactly one phase, keeping training and reload in separate processes.

    This mirrors demo.py step by step so the small version-specific changes
    remain visible. The CLI intentionally never trains during a reload.
    """
    if md.version('pymc-bart') != '0.13.1':
        raise ValueError('demo_current.py requires pymc-bart 0.13.1')
    if action == 'train':
        folder.mkdir(parents=True, exist_ok=True)
        # Use the same synthetic problem and batch sizes as the PyMC 5 example.
        rng = np.random.default_rng(48)
        X = rng.uniform(-1, 1, (120, 2))
        y = np.sin(3*X[:, 0]) + .5*X[:, 1] + rng.normal(0, .15, 120)
        cases = {f'new_{n}': rng.uniform(-1, 1, (n, 2)) for n in [1, 17, 120, 175]}
        cases['training'] = X.copy()
        np.savez(folder/'inputs.npz', X=X, target=y, **cases)
        model = model_for(X, y)
        with model:
            trace = pm.sample(draws=DRAWS, tune=100, chains=2, cores=2,
                              random_seed=51, progressbar=False,
                              compute_convergence_checks=False)
        op = model['mu'].owner.op
        # In 0.13.1, each entry contains a baseline forest plus change batches.
        # n_outputs is set during training and is absent on a fresh BART RV.
        archive = {'trees': list(op.all_trees), 'n_outputs': op.n_outputs}
        sampler = _get_posterior_sampler(op)
        assert sampler.n_draws == 2 * DRAWS
        # PyMC 6 returns an xarray DataTree; it can still be saved to NetCDF.
        trace.to_netcdf(folder/'trace.nc', engine='h5netcdf')
        for serializer in [pickle, cloudpickle]:
            with (folder/f'{serializer.__name__}.pkl').open('wb') as f:
                serializer.dump(archive, f, protocol=5)
        refs = {name+'_'+kind: values for name, Xnew in cases.items()
                for kind, values in outputs(model, trace, Xnew).items()}
        np.savez_compressed(folder/'reference.npz', **refs)
        meta = {'versions': {**versions(), 'bartrs': md.version('bartrs')},
                'training_pid': os.getpid(), 'retained_ensembles': sampler.n_draws,
                'stored_chain_histories': len(archive['trees']), 'n_outputs': archive['n_outputs']}
        (folder/'metadata.json').write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)
        return
    # Everything below runs in the new process; training has already exited.
    data = np.load(folder/'inputs.npz')
    refs = np.load(folder/'reference.npz')
    meta = json.loads((folder/'metadata.json').read_text())
    current = {**versions(), 'bartrs': md.version('bartrs')}
    check_versions(meta['versions'], current, [k for k in current if k not in ['python', 'platform']])
    trace = xr.open_datatree(folder/'trace.nc', engine='h5netcdf').load()
    model = model_for(data['X'], data['target'])
    def forbidden(*args, **kwargs):
        """Guard against an accidental retrain disguised as a successful load."""
        raise AssertionError('Reload must never retrain')
    pm.sample = forbidden
    result = {'versions': meta['versions'], 'reload_pid': os.getpid(),
              'training_pid': meta['training_pid'], 'checks': [], 'max_absolute_difference': 0.0}
    # Show that the numerical trace alone still cannot predict a new batch.
    try:
        bad = posterior(model, trace, data['new_17'])
        assert bad['mu'].shape != refs['new_17_mu'].shape or not np.allclose(bad['mu'], refs['new_17_mu'])
        result['trace_only'] = 'predictions differ'
    except (ValueError, TypeError) as exc:
        result['trace_only'] = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
    # Both files are generated locally by this example. Never unpickle an
    # arbitrary downloaded file: this format is for trusted model artifacts.
    for serializer in [pickle, cloudpickle]:
        with (folder/f'{serializer.__name__}.pkl').open('rb') as f:
            archive = serializer.load(f)
        op = model['mu'].owner.op
        # Keep the class-backed list and restore the class-level output count.
        # An instance-only n_outputs assignment would be invisible to rng_fn.
        op.all_trees[:] = archive['trees']
        type(op).n_outputs = archive['n_outputs']
        # Discard any sampler cached while a previous archive occupied this RV.
        _posterior_sampler_cache.clear()
        assert _get_posterior_sampler(op).n_draws == meta['retained_ensembles']
        for name in [k for k in data.files if k not in ['X', 'target']]:
            for kind, values in outputs(model, trace, data[name]).items():
                reference = refs[name+'_'+kind]
                error = compare_predictions(values, reference)
                result['max_absolute_difference'] = max(result['max_absolute_difference'], error)
                result['checks'].append({'format': serializer.__name__, 'case': name,
                                         'kind': kind, 'shape': list(values.shape), 'max_error': error})
    result['passed'] = True
    (folder/'result-current.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'reload'])
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    main(args.action, args.folder)
