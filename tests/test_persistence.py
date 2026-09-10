import importlib.metadata as md
import json
import pickle
import subprocess
import sys

import numpy as np
import pymc as pm
import pymc_bart as pmb
import pytest

from bart_persistence.save import save_bart, _versions


@pytest.fixture(scope='session')
def trained():
    X = np.random.default_rng(3).normal(size=(24, 2))
    y = np.sin(X[:, 0]) + X[:, 1]
    with pm.Model() as model:
        x = pm.Data('X', X)
        mu = pmb.BART('mu', x, y, m=4)
        pm.Normal('y', mu, 0.2, observed=y)
        pm.sample(draws=6, tune=6, chains=2, cores=2, random_seed=4,
                  progressbar=False, compute_convergence_checks=False)
    return mu, X


def test_versions(monkeypatch):
    for bart, pmc in [('0.11.0', '5.25.1'), ('0.13.1', '6.3.2')]:
        versions = {'pymc-bart': bart, 'pymc': pmc, 'numpy': '1', 'pytensor': '2', 'bartrs': '3'}
        monkeypatch.setattr('bart_persistence.save.md.version', versions.__getitem__)
        result = _versions()
        assert ('bartrs' in result) == (bart == '0.13.1')
    for bart, pmc in [('0.12.0', '6.3.2'), ('0.11.0', '6.3.2')]:
        versions.update({'pymc-bart': bart, 'pymc': pmc})
        with pytest.raises(ValueError, match='Supported stacks'):
            _versions()


def test_save_bart_records_state_without_overwriting(trained, tmp_path):
    rv, _ = trained
    path = tmp_path / 'model.bart'
    assert save_bart(rv, path) is None
    contents = path.read_bytes()
    header_bytes, payload = contents.split(b'\n', 1)
    header, state = json.loads(header_bytes), pickle.loads(payload)
    assert header == {'format': 'bart-persistence/1', 'versions': _versions(), 'n_features': 2}
    assert state['m'] == rv.owner.op.m
    assert len(state['trees']) == len(rv.owner.op.all_trees)
    if md.version('pymc-bart') == '0.13.1':
        assert state['n_outputs'] == 1
    with pytest.raises(FileExistsError):
        save_bart(rv, path)
    assert path.read_bytes() == contents


def test_untrained_and_wrong_variable(tmp_path):
    import pytensor.tensor as pt
    with pm.Model():
        rv = pmb.BART('mu', np.zeros((3, 2)), np.ones(3), m=2)
        multi = pmb.BART('multi', np.zeros((3, 2)), np.ones(3), m=2, shape=(2, 3))
        normal = pm.Normal('normal')
    for wrong in [pt.vector(), normal]:
        with pytest.raises(TypeError, match='BART random variable'):
            save_bart(wrong, tmp_path/'no')
    with pytest.raises(ValueError, match='scalar-output'):
        save_bart(multi, tmp_path/'no')
    with pytest.raises(ValueError, match='No posterior trees'):
        save_bart(rv, tmp_path/'no')
    assert not (tmp_path/'no').exists()
