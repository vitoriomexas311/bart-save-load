"""Verify cache isolation without loading or compiling PyTensor."""

import os
import unittest
from unittest.mock import patch

from runtime import configure_compiler_cache


class CompilerCacheTests(unittest.TestCase):
    """Respect explicit settings while avoiding cross-version cache collisions."""

    def test_default_cache_contains_pytensor_version(self):
        """Two different PyTensor versions must not select the same base cache."""
        with patch.dict(os.environ, {}, clear=True), patch('runtime.md.version', return_value='2.31.7'):
            configure_compiler_cache()
            first = os.environ['PYTENSOR_FLAGS']
        with patch.dict(os.environ, {}, clear=True), patch('runtime.md.version', return_value='3.3.1'):
            configure_compiler_cache()
            second = os.environ['PYTENSOR_FLAGS']
        self.assertNotEqual(first, second)
        self.assertIn('2.31.7', first)
        self.assertIn('3.3.1', second)

    def test_existing_noncache_flags_are_preserved(self):
        """Adding cache isolation must not drop unrelated user configuration."""
        with patch.dict(os.environ, {'PYTENSOR_FLAGS': 'optimizer=fast_compile'}, clear=True):
            configure_compiler_cache()
            self.assertTrue(os.environ['PYTENSOR_FLAGS'].startswith('optimizer=fast_compile,'))

    def test_explicit_cache_is_preserved(self):
        """Either explicit cache key takes precedence over the example default."""
        for key in ['compiledir', 'base_compiledir']:
            value = f'{key}=/user/chosen/cache'
            with self.subTest(key=key), patch.dict(os.environ, {'PYTENSOR_FLAGS': value}, clear=True):
                configure_compiler_cache()
                self.assertEqual(os.environ['PYTENSOR_FLAGS'], value)

    def test_repeated_configuration_is_idempotent(self):
        """Both example modules may call the setup function during one import."""
        with patch.dict(os.environ, {}, clear=True):
            configure_compiler_cache()
            first = os.environ['PYTENSOR_FLAGS']
            configure_compiler_cache()
            self.assertEqual(os.environ['PYTENSOR_FLAGS'], first)
