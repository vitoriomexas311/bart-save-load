import importlib.metadata as md
import json
import pickle
import subprocess
import sys

import numpy as np
import pymc as pm
import pymc_bart as pmb
import pytest

from bart_persistence.save import _versions


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
