import hashlib
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
from bart_persistence.load import load_bart, BARTPredictor


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
    assert header == {'format': 'bart-persistence/1', 'versions': _versions(), 'n_features': 2,
                      'sha256': hashlib.sha256(payload).hexdigest()}
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


@pytest.fixture
def saved_state(trained, tmp_path):
    path = tmp_path / 'state.bart'
    save_bart(trained[0], path)
    return pickle.loads(path.read_bytes().split(b'\n', 1)[1])


@pytest.fixture
def predictor(saved_state):
    return BARTPredictor(saved_state, 2, md.version('pymc-bart'))


def test_construct_predictor(predictor):
    assert predictor.n_features == 2
    assert predictor.n_draws == 12


def test_empty_legacy_artifact():
    with pytest.raises(ValueError, match='no posterior draws'):
        BARTPredictor({'trees': []}, 2, '0.11.0')


def test_predict_matches_live_trees(trained, predictor):
    rv, _ = trained
    from pymc_bart.utils import _sample_posterior
    if md.version('pymc-bart') == '0.13.1':
        from pymc_bart.utils import _get_posterior_sampler
        original = _get_posterior_sampler(rv.owner.op)
    else:
        original = list(rv.owner.op.all_trees)
    for n in [1, 7, 24, 31]:
        new = np.random.default_rng(n).normal(size=(n, 2))
        expected = _sample_posterior(original, new, np.random.default_rng(10), size=100)[:, :, 0]
        np.testing.assert_array_equal(predictor.predict(new, draws=100, seed=10), expected)


def test_invalid_prediction_inputs(predictor):
    p = predictor
    for X in [[], [1, 2], np.zeros((2, 3)), np.zeros((0, 2)), [[np.nan, 0]], [[np.inf, 0]]]:
        with pytest.raises(ValueError):
            p.predict(X)
    for draws in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match='positive integer'):
            p.predict([[1, 2]], draws=draws)


def test_bad_artifact_before_unpickle(tmp_path, monkeypatch):
    path = tmp_path/'bad'
    def forbidden(*args):
        raise AssertionError('Must check header before unpickling')
    monkeypatch.setattr('bart_persistence.load.pickle.load', forbidden)
    for header, error in [({'format': 'other'}, 'format'),
                          ({'format': 'bart-persistence/1', 'versions': {}}, 'versions differ')]:
        path.write_bytes(json.dumps(header).encode() + b'\ninvalid pickle')
        with pytest.raises(ValueError, match=error):
            load_bart(path)


def test_real_roundtrip(trained, tmp_path):
    rv, X = trained
    path = tmp_path / 'model.bart'
    save_bart(rv, path)
    predictor = load_bart(path)
    assert predictor.n_draws == 12
    np.save(tmp_path / 'X.npy', X)
    np.save(tmp_path / 'expected.npy', predictor.predict(X, draws=100, seed=10))
    # Run outside the repo: proves the installed package works without source cwd.
    code = '''
import numpy as np
import pymc as pm
from bart_persistence.load import load_bart

def forbidden(*args, **kwargs):
    raise AssertionError("Loading must not train or reconstruct a model")
pm.sample = forbidden
pm.Model = forbidden
p = load_bart('model.bart')
np.testing.assert_array_equal(p.predict(np.load('X.npy'), draws=100, seed=10), np.load('expected.npy'))
'''
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True, timeout=120)
    np.testing.assert_array_equal(load_bart(path).predict(X, seed=2), predictor.predict(X, seed=2))


def test_independent_models(trained, tmp_path):
    save_bart(trained[0], tmp_path/'a')
    a = load_bart(tmp_path/'a')
    reference = a.predict(trained[1], seed=1)
    with pm.Model():
        # Creating another BART RV must not replace a loaded predictor's trees.
        pmb.BART('other', np.zeros((3, 2)), np.ones(3), m=2)
    b = load_bart(tmp_path/'a')
    assert a._trees is not b._trees
    np.testing.assert_array_equal(a.predict(trained[1], seed=1), reference)


@pytest.mark.parametrize("operation", ["fsync", "link"])
def test_save_failure_does_not_publish_partial_artifact(trained, tmp_path, monkeypatch, operation):
    def fail(*args):
        raise OSError("simulated storage failure")
    monkeypatch.setattr(f"bart_persistence.save.os.{operation}", fail)
    with pytest.raises(OSError, match="storage failure"):
        save_bart(trained[0], tmp_path / "model.bart")
    assert list(tmp_path.iterdir()) == []
