"""Load trees once and pass a standalone predictor between application functions."""

import json
from pathlib import Path
import pickle

import numpy as np

from .save import _versions


class BARTPredictor:
    """Scalar latent BART posterior. Construct with load_bart()."""

    def __init__(self, state, n_features, version):
        self.n_features = n_features
        self._current = version == "0.13.1"
        if self._current:
            from pymc_bart.pymc_bart import PosteriorSampler
            from pymc_bart.utils import _MultiChainSampler

            self._trees = _MultiChainSampler([
                PosteriorSampler.from_history(batches, baseline, state["m"], state["n_outputs"])
                for baseline, batches in state["trees"]
            ])
            self.n_draws = self._trees.n_draws
        else:
            self._trees = state["trees"]
            self.n_draws = len(self._trees)
        if self.n_draws == 0:
            raise ValueError("Artifact contains no posterior draws")
