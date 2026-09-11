"""Save scalar PyMC BART posterior trees to one trusted-local artifact."""

import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import pickle
import tempfile


SUPPORTED = {"0.11.0": "5.25.1", "0.13.1": "6.3.2"}


def _versions():
    versions = {name: md.version(name) for name in
                ("pymc-bart", "pymc", "pytensor", "numpy")}
    bart = versions["pymc-bart"]
    if bart not in SUPPORTED or versions["pymc"] != SUPPORTED[bart]:
        raise ValueError("Supported stacks: BART 0.11.0 / PyMC 5.25.1 or BART 0.13.1 / PyMC 6.3.2")
    if bart == "0.13.1":
        versions["bartrs"] = md.version("bartrs")
    return versions


def save_bart(rv, path, *, expected_draws=None):
    """Save a trained scalar BART random variable; refuse to overwrite a file.

    Pass model["mu"], not the trace or a transformed deterministic variable.
    The parent directory must exist. Only load artifacts you trust.
    Set expected_draws to trace chains * draws to detect missing tree histories.
    """
    from pymc_bart.bart import BARTRV

    versions = _versions()
    if getattr(rv, "owner", None) is None or not isinstance(rv.owner.op, BARTRV):
        raise TypeError("Expected a BART random variable, e.g. model['mu']")
    op = rv.owner.op
    if rv.ndim != 1:
        raise ValueError("Only scalar-output BART is supported")
    trees = list(op.all_trees)
    if not trees:
        raise ValueError("No posterior trees: train with pm.sample() before saving")
    # Materialize shared/process-backed state, never pickle the model or manager.
    state = {"trees": trees, "m": op.m}
    if versions["pymc-bart"] == "0.13.1":
        state["n_outputs"] = op.n_outputs
    if expected_draws is not None:
        from .load import BARTPredictor

        if isinstance(expected_draws, bool) or not isinstance(expected_draws, int) or expected_draws < 1:
            raise ValueError("expected_draws must be a positive integer")
        retained = BARTPredictor(state, 0, versions["pymc-bart"]).n_draws
        if retained != expected_draws:
            raise ValueError(f"BART retained {retained} draws; expected {expected_draws}. "
                             "Use parallel chains or a single chain before saving.")
    header = {"format": "bart-persistence/1", "versions": versions,
              "n_features": int(op.X.eval().shape[1])}
    # Serialize before opening so serialization errors cannot leave a partial file.
    payload = pickle.dumps(state, protocol=5)
    header["sha256"] = hashlib.sha256(payload).hexdigest()
    path = Path(path)
    # A same-filesystem hard link publishes the complete file without replacing
    # an existing destination. Failures leave the destination untouched.
    with tempfile.TemporaryDirectory(prefix=".bart-", dir=path.parent) as directory:
        staging = Path(directory) / "artifact"
        with staging.open("wb") as stream:
            stream.write(json.dumps(header).encode() + b"\n" + payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)
