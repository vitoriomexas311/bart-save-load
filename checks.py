"""Small, testable checks shared by the two executable examples.

These checks deliberately raise exceptions instead of using bare ``assert``:
Python's ``-O`` flag must not turn off compatibility or prediction validation.
"""

import numpy as np


def check_versions(saved, current, packages):
    """Reject a different package stack before attempting to unpickle trees.

    Only the requested package keys are compared. Python and OS are recorded for
    provenance, but are intentionally not compared: the experiment also tests
    cross-Python reloads. This is a compatibility check, not a pickle safety check.
    """
    for package in packages:
        if saved.get(package) != current.get(package) or package not in saved:
            raise ValueError(
                f"Incompatible {package}: saved={saved.get(package)!r}, "
                f"installed={current.get(package)!r}. Use the matching dependency lock."
            )


def compare_predictions(actual, expected):
    """Require the same shape and finite values, then compare every element.

    An explicit shape check prevents NumPy broadcasting from disguising a
    prediction-length error. Reject NaNs even when both arrays contain a NaN at
    the same location. Return the maximum error for the readable test report.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"Prediction shapes differ: {actual.shape} != {expected.shape}")
    if actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Predictions must be nonempty and finite")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(actual - expected)))
