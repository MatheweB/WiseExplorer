"""
Tests for wise_explorer.core.types

Tests Stats class and scoring calculations - central to move evaluation.
"""

import math
import pytest

from wise_explorer.agent.agent import State
from wise_explorer.core.types import (
    Stats,
    W_WEIGHT, T_WEIGHT, L_WEIGHT,
    SCORE_MIN, SCORE_MAX, SCORE_RANGE,
    OUTCOME_INDEX, UNEXPLORED_ANCHOR_ID,
)


class TestConstants:
    """Tests for module constants."""

    def test_score_range_positive(self):
        """Score range is positive and consistent."""
        assert SCORE_RANGE > 0
        assert SCORE_RANGE == SCORE_MAX - SCORE_MIN

    def test_outcome_index_complete(self):
        """OUTCOME_INDEX covers WIN, TIE, LOSS with indices 0-2."""
        assert set(OUTCOME_INDEX.values()) == {0, 1, 2}
        assert State.WIN in OUTCOME_INDEX
        assert State.TIE in OUTCOME_INDEX
        assert State.LOSS in OUTCOME_INDEX

    def test_unexplored_anchor_negative(self):
        """Sentinel is negative (won't conflict with real IDs)."""
        assert UNEXPLORED_ANCHOR_ID < 0


class TestStatsBasic:
    """Basic Stats functionality tests."""

    def test_creation_and_access(self):
        """Stats can be created and accessed."""
        s = Stats(10, 20, 30)
        assert s.wins == 10
        assert s.ties == 20
        assert s.losses == 30
        assert s[0] == 10 and s[1] == 20 and s[2] == 30

    def test_immutable(self):
        """Stats is immutable (NamedTuple)."""
        s = Stats(1, 2, 3)
        with pytest.raises(AttributeError):
            s.wins = 99

    def test_equality_and_hashing(self):
        """Stats supports equality and hashing."""
        s1, s2, s3 = Stats(10, 20, 30), Stats(10, 20, 30), Stats(1, 2, 3)
        assert s1 == s2
        assert s1 != s3
        assert len({s1, s2, s3}) == 2


class TestStatsTotal:
    """Tests for Stats.total property."""

    def test_total(self, zero_stats, balanced_stats, winning_stats):
        """Total sums all counts."""
        assert zero_stats.total == 0
        assert balanced_stats.total == 30
        assert winning_stats.total == 115


class TestStatsDistribution:
    """Tests for Stats.distribution property."""

    def test_distribution_zero(self, zero_stats):
        """Zero stats return zero distribution."""
        assert zero_stats.distribution == (0.0, 0.0, 0.0)

    def test_distribution_sums_to_one(self, winning_stats):
        """Non-zero distribution sums to 1.0."""
        assert sum(winning_stats.distribution) == pytest.approx(1.0)

    def test_distribution_uniform(self, balanced_stats):
        """Balanced stats have uniform distribution."""
        assert balanced_stats.distribution == pytest.approx((1/3, 1/3, 1/3))


class TestStatsMeanScore:
    """Tests for Stats.mean_score property."""

    def test_mean_score_in_range(self, winning_stats, losing_stats, balanced_stats):
        """Mean score is always in [0, 1]."""
        for stats in [winning_stats, losing_stats, balanced_stats]:
            assert 0.0 <= stats.mean_score <= 1.0

    def test_mean_score_ordering(self, winning_stats, losing_stats):
        """Higher win rate = higher mean score."""
        assert winning_stats.mean_score > losing_stats.mean_score

    def test_mean_score_uses_pseudocounts(self, zero_stats):
        """Zero stats still have valid mean_score (Bayesian smoothing)."""
        score = zero_stats.mean_score
        assert 0.0 <= score <= 1.0


class TestStatsStdError:
    """Tests for Stats.std_error property."""

    def test_std_error_infinite_for_low_samples(self, zero_stats):
        """Low sample counts have infinite std_error."""
        assert zero_stats.std_error == float('inf')
        assert Stats(1, 0, 0).std_error == float('inf')

    def test_std_error_finite_for_many_samples(self, winning_stats):
        """Many samples have finite std_error."""
        assert math.isfinite(winning_stats.std_error)
        assert winning_stats.std_error >= 0.0

    def test_std_error_decreases_with_samples(self):
        """Std error decreases as sample count increases."""
        small = Stats(10, 5, 5)
        large = Stats(100, 50, 50)
        assert large.std_error < small.std_error


class TestStatsUtility:
    """Tests for Stats.utility property."""

    def test_utility_zero_for_empty(self, zero_stats):
        """Zero stats have zero utility."""
        assert zero_stats.utility == 0.0

    def test_utility_matches_weights(self):
        """Pure outcomes match weight constants."""
        assert Stats(100, 0, 0).utility == W_WEIGHT
        assert Stats(0, 100, 0).utility == T_WEIGHT
        assert Stats(0, 0, 100).utility == L_WEIGHT


class TestStatsCertainty:
    """Tests for Stats.certainty property."""

    def test_certainty_zero_for_low_samples(self, zero_stats):
        """Low sample counts have zero certainty."""
        assert zero_stats.certainty == 0.0
        assert Stats(1, 0, 0).certainty == 0.0

    def test_certainty_in_range(self, winning_stats):
        """Certainty is in [0, 1]."""
        assert 0.0 <= winning_stats.certainty <= 1.0

    def test_certainty_increases_with_samples(self):
        """Certainty increases with more samples."""
        small = Stats(10, 5, 5)
        large = Stats(100, 50, 50)
        assert large.certainty > small.certainty