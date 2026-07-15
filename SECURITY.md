# Security policy

## Trust boundary

Only load artifacts produced by a trusted training process and transferred through
a trusted channel. The format uses Python pickle, which can execute arbitrary code.
Dependency checks and SHA-256 digests are not authentication and do not sandbox
unpickling. Do not expose `load_bart` as an upload endpoint for untrusted users.
Treat saved models as potentially sensitive derived data.

## Reporting

Use [GitHub private vulnerability reporting](https://github.com/vitoriomexas311/bart-save-load/security/advisories/new)
for a security issue. Include the affected commit, supported dependency stack and a
minimal synthetic reproducer. Do not post working exploit artifacts or confidential
training data in a public issue. Ordinary compatibility bugs belong in Issues.

This project is maintained on a best-effort basis with no promised response time.
Security fixes target the documented supported stacks. Pin dependencies and only
upgrade training and serving together after verifying compatibility.
