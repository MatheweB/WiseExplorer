"""
Tests for wise_explorer.core.bayes

Tests Bayesian clustering utilities for anchor merging decisions.
"""

import math
import pytest

from wise_explorer.core.bayes import log_bayes_factor, compatible, similarity


class TestLogBayesFactor:
    """Tests for log_bayes_factor function."""

    def test_identical_distributions_positive(self):
        """Identical distributions favor same-distribution hypothesis."""
        a = b = (100, 50, 50)
        assert log_bayes_factor(a, b) > 0

    def test_different_distributions_negative(self):
        """Very different distributions favor different-distribution hypothesis."""
        a, b = (100, 0, 0), (0, 0, 100)
        assert log_bayes_factor(a, b) < 0

    def test_symmetric(self):
        """log_bayes_factor(a, b) == log_bayes_factor(b, a)."""
        a, b = (80, 10, 10), (70, 15, 15)
        assert log_bayes_factor(a, b) == pytest.approx(log_bayes_factor(b, a))

    def test_empty_distributions(self):
        """Empty distributions produce finite result."""
        assert math.isfinite(log_bayes_factor((0, 0, 0), (10, 5, 5)))
        assert math.isfinite(log_bayes_factor((0, 0, 0), (0, 0, 0)))


class TestCompatible:
    """Tests for compatible function."""

    def test_identical_compatible(self):
        """Identical distributions are compatible."""
        a = (100, 50, 50)
        assert compatible(a, a) is True

    def test_very_different_not_compatible(self):
        """Very different distributions are not compatible."""
        assert compatible((100, 0, 0), (0, 0, 100)) is False

    def test_empty_compatible(self):
        """Empty distributions are compatible (no evidence against)."""
        assert compatible((0, 0, 0), (0, 0, 0)) is True
        assert compatible((0, 0, 0), (100, 50, 50)) is True

    def test_symmetric(self):
        """compatible(a, b) == compatible(b, a)."""
        a, b = (80, 10, 10), (70, 15, 15)
        assert compatible(a, b) == compatible(b, a)


class TestSimilarity:
    """Tests for similarity function."""

    def test_identical_similarity_one(self):
        """Identical distributions have similarity 1.0."""
        a = (100, 50, 50)
        assert similarity(a, a) == pytest.approx(1.0)

    def test_maximally_different_similarity_zero(self):
        """Maximally different distributions have similarity 0.0."""
        assert similarity((100, 0, 0), (0, 0, 100)) == pytest.approx(0.0)

    def test_similarity_in_range(self):
        """Similarity is always in [0, 1]."""
        test_cases = [
            ((100, 0, 0), (50, 50, 0)),
            ((33, 33, 34), (34, 33, 33)),
            ((1, 1, 1), (10, 10, 10)),
        ]
        for a, b in test_cases:
            assert 0.0 <= similarity(a, b) <= 1.0

    def test_empty_similarity_zero(self):
        """Empty distribution has similarity 0 with anything."""
        assert similarity((0, 0, 0), (100, 50, 50)) == 0.0

    def test_symmetric(self):
        """similarity(a, b) == similarity(b, a)."""
        a, b = (80, 10, 10), (70, 15, 15)
        assert similarity(a, b) == pytest.approx(similarity(b, a))

    def test_proportional_same_similarity(self):
        """Proportionally equal distributions have similarity 1.0."""
        assert similarity((10, 5, 5), (100, 50, 50)) == pytest.approx(1.0)