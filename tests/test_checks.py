"""Fast unit tests for checks that prevent misleading persistence results."""

import unittest

import numpy as np

from checks import check_versions, compare_predictions


class CompatibilityTests(unittest.TestCase):
    """Archives must match packages, while deliberate cross-Python tests work."""

    def test_same_packages_different_python_are_allowed(self):
        """Python provenance is not accidentally treated as a package lock."""
        check_versions({'bart': '1', 'python': '3.11'},
                       {'bart': '1', 'python': '3.13'}, ['bart'])

    def test_different_package_is_rejected(self):
        """Do not silently use an archive from a different tree implementation."""
        with self.assertRaisesRegex(ValueError, 'Incompatible bart'):
            check_versions({'bart': '1'}, {'bart': '2'}, ['bart'])

    def test_missing_metadata_is_rejected(self):
        """Missing version information cannot certify compatibility."""
        with self.assertRaises(ValueError):
            check_versions({}, {}, ['bart'])


class PredictionChecksTests(unittest.TestCase):
    """Equality must cover actual values and shapes, not merely summary means."""

    def test_identical_predictions_have_zero_error(self):
        """An exact serialization round trip reports zero numerical error."""
        self.assertEqual(compare_predictions([[1., 2.]], [[1., 2.]]), 0)

    def test_small_roundoff_is_measured(self):
        """Report an allowed tiny error instead of rounding it away."""
        self.assertGreater(compare_predictions([1. + 1e-13], [1.]), 0)

    def test_changed_prediction_fails(self):
        """A corrupted prediction must fail even if array shapes still match."""
        with self.assertRaises(AssertionError):
            compare_predictions([1., 2.], [1., 3.])

    def test_broadcastable_shape_mismatch_fails(self):
        """A singleton result must not broadcast into a false successful batch."""
        with self.assertRaisesRegex(AssertionError, 'shapes differ'):
            compare_predictions([1.], [[1.]])

    def test_matching_nan_or_infinity_fails(self):
        """Matching invalid predictions do not count as successful inference."""
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                compare_predictions([value], [value])

    def test_empty_predictions_fail(self):
        """An empty batch cannot pass a numerical verification vacuously."""
        with self.assertRaises(AssertionError):
            compare_predictions([], [])
