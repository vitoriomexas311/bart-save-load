"""Unit-test additive tree predictions against a hand-computable tiny forest.

These tests need no MCMC sampling. The 0.11 Python tree objects expose a small
constructor, whereas the newer release uses different Rust-backed histories.
The current release is covered by the real training/reload integration test.
"""

import importlib.metadata as md
import pickle
import unittest

import cloudpickle
import numpy as np


@unittest.skipUnless(md.version('pymc-bart') == '0.11.0', 'Python Tree API belongs to BART 0.11')
class TreePredictionTests(unittest.TestCase):
    """Verify the demo sums trees and preserves all posterior ensembles."""

    def setUp(self):
        """Build one split tree and two constant trees with known predictions."""
        from pymc_bart.tree import Node, Tree
        from pymc_bart.split_rules import ContinuousSplitRule

        rules = [ContinuousSplitRule, ContinuousSplitRule]
        split = Tree(
            tree_structure={0: Node(value=np.array(0.), idx_split_variable=0),
                            1: Node(value=np.array([2.])),
                            2: Node(value=np.array([-3.]))},
            output=None, split_rules=rules, idx_leaf_nodes=[1, 2])
        one = Tree({0: Node(value=np.array([1.]))}, None, rules, [0])
        seven = Tree({0: Node(value=np.array([7.]))}, None, rules, [0])
        # Two retained draws, each with a single output. Their means differ.
        self.trees = [[[split, one]], [[seven]]]
        self.X = np.array([[-.5, 0.], [.5, 0.]])

    def test_every_forest_matches_manual_addition(self):
        """First draw is (2+1, -3+1); the second draw is (7, 7)."""
        from demo import every_forest
        np.testing.assert_array_equal(every_forest(self.trees, self.X),
                                      np.array([[[3., -2.]], [[7., 7.]]]))

    def test_posterior_sampling_is_seeded_and_selects_whole_ensembles(self):
        """Posterior draws must use a complete forest, not average all draws."""
        from demo import tree_draws
        actual = tree_draws(self.trees, self.X)
        np.testing.assert_array_equal(actual, tree_draws(self.trees, self.X))
        self.assertEqual(actual.shape, (128, 2, 1))
        self.assertEqual(set(map(tuple, actual[:, :, 0])), {(3., -2.), (7., 7.)})

    def test_both_serializers_preserve_real_tree_nodes(self):
        """Standard pickle and cloudpickle preserve split thresholds and leaves."""
        from demo import every_forest
        for serializer in [pickle, cloudpickle]:
            with self.subTest(serializer=serializer.__name__):
                restored = serializer.loads(serializer.dumps(self.trees, protocol=5))
                self.assertIsNot(restored[0][0][0], self.trees[0][0][0])
                np.testing.assert_array_equal(every_forest(restored, self.X),
                                              np.array([[[3., -2.]], [[7., 7.]]]))
