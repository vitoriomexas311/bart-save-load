"""Save scalar PyMC BART posterior trees to one trusted-local artifact."""

import importlib.metadata as md
import json
from pathlib import Path
import pickle


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
