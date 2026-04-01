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

    def predict(self, X, *, draws=500, seed=None):
        """Return latent draws shaped (draws, rows), sampled with replacement.

        X must be a nonempty finite numeric matrix in training feature order.
        Apply your own inverse link to each draw; observation noise is not included.
        """
        from pymc_bart.utils import _sample_posterior

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features or X.shape[0] == 0:
            raise ValueError(f"X must have shape (nonzero rows, {self.n_features})")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive integer")
        result = _sample_posterior(self._trees, np.ascontiguousarray(X),
                                   np.random.default_rng(seed), size=draws)
        return result[:, :, 0]
