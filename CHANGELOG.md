# Changelog

## Unreleased — 0.1.0

Initial package; no stable release has been published.

- Two-module save/load API for scalar PyMC-BART posterior trees, supporting BART
  0.11.0 / PyMC 5.25.1 and BART 0.13.1 / PyMC 6.3.2.
- Standalone seeded latent predictions without a trace, reconstructed model or retraining.
- Atomic no-overwrite publication and payload checksums. Earlier format-1 artifacts
  without checksums remain readable; they cannot be checked for corruption.
- Optional expected posterior draw count to detect incomplete upstream histories.
- Exact dependency-version checks and explicit prediction limitations.
- Real fresh-process tests, cross-platform CI, a 100% combined coverage gate,
  installed-wheel notebook checks and distribution metadata validation.
- Developer documentation, bug reporting and a private security-reporting path.
