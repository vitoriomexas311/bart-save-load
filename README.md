# BART save and load

[![CI](https://github.com/vitoriomexas311/bart-save-load/actions/workflows/tests.yml/badge.svg)](https://github.com/vitoriomexas311/bart-save-load/actions/workflows/tests.yml)

**Save trained PyMC-BART trees. Restart Python. Predict without retraining.**

An independent companion package for [PyMC-BART](https://github.com/pymc-devs/pymc-bart).
It persists the posterior tree state that a numerical trace alone does not retain,
and loads a reusable predictor without reconstructing your PyMC model.

## Install

For the current supported stack, directly from GitHub:

```sh
python -m pip install "bart-persistence[current] @ git+https://github.com/vitoriomexas311/bart-save-load.git"
```

Use `[legacy]` for BART 0.11.0 / PyMC 5.25.1. In an existing supported environment,
omit the extra to install only this utility and NumPy without upgrading PyMC.
A bare install does **not** supply PyMC-BART; choose a stack extra for a new environment.
No serializer, database or web-service dependency is added to your runtime.

This is an early, narrowly supported package, not a PyMC-endorsed project or a PyPI
release. For repeatable installs, append `@<full-commit-SHA>` to the Git URL or use
a wheel from a successful [CI run](https://github.com/vitoriomexas311/bart-save-load/actions/workflows/tests.yml).
Keep the same dependency lock when training and loading.

## Save after training

```python
from bart_persistence.save import save_bart

# model["mu"] must be the BART random variable, not a deterministic transform.
save_bart(
    model["mu"],
    "model.bart",
    expected_draws=trace.posterior.sizes["chain"] * trace.posterior.sizes["draw"],
)
```

## Load in another script

```python
from bart_persistence.load import load_bart

predictor = load_bart("model.bart")
samples = predictor.predict(X_new, draws=1000, seed=42)  # (draws, rows)
mean = samples.mean(axis=0)
```

The predictor owns its trees and can be passed between application functions.
Load once, then reuse it for new batches. The runtime remains two Python modules.

## Compatibility and guarantees

| Supported BART | Supported PyMC | Python |
| --- | --- | --- |
| 0.11.0 | 5.25.1 | 3.11+ |
| 0.13.1 | 6.3.2 | 3.12+ |

- **Scalar latent BART predictions.** Likelihood noise, other model parameters,
  multi-output models and continued training are outside this API.
- **Exact environment matching.** BART, PyMC, PyTensor and NumPy versions must
  match the artifact; current BART also checks bartrs. This is not a migration tool.
- **Complete-file publication.** Saves never overwrite an existing artifact;
  a checksum detects payload corruption on reload. Local hard-link support is required.
- **Retained-draw validation.** `expected_draws` detects missing histories. In BART
  0.13.1, use parallel chains or one chain; sequential chains can lose upstream history.
- **Trusted artifacts only.** Pickle can execute code. A checksum does not make an
  untrusted file safe. Preserve feature order and preprocessing yourself.

CI checks real training/reload equivalence in a fresh process, installed wheels,
the notebook, and **100% combined statement and branch coverage of the runtime**.
It covers Linux, macOS and Windows. Coverage describes tested code paths, not a
proof that every BART model or environment is supported.

## Documentation and maintenance

- [Full API and artifact reference](docs/reference.md): complete training example,
  input/output shapes, file format, failure behavior and limitations.
- [Executable notebook](bart_persistence_walkthrough.ipynb): train, save, load,
  and verify predictions in a separate interpreter.
- [Contributing](CONTRIBUTING.md): development setup, tests and release checklist.
- [Changelog](CHANGELOG.md) · [Security policy](SECURITY.md) ·
  [Report a bug](https://github.com/vitoriomexas311/bart-save-load/issues/new/choose)
