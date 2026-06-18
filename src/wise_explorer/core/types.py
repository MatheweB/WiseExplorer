"""
This module contains the fundamental types used throughout the game AI system:
- Stats: Outcome counts with scoring properties
- Outcome weights and derived constants
- Common data classes
"""

from __future__ import annotations

import math
from typing import NamedTuple

from wise_explorer.agent.agent import State

Counts = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Outcome Weights — maps [LOSS, TIE, WIN] → [0, 0.5, 1]
# ---------------------------------------------------------------------------

L_WEIGHT = 0.0   # Utility for a LOSS
T_WEIGHT = 0.5   # Utility for a TIE
W_WEIGHT = 1.0   # Utility for a WIN

SCORE_MIN = L_WEIGHT   # 0.0
SCORE_MAX = W_WEIGHT   # 1.0
SCORE_RANGE = SCORE_MAX - SCORE_MIN  # 1.0

OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.LOSS: 2}
OUTCOME_SCORE = {State.WIN: W_WEIGHT, State.TIE: T_WEIGHT, State.LOSS: L_WEIGHT}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats(NamedTuple):
    """Outcome counts with Bayesian scoring via pseudocounts."""

    wins: float = 0
    ties: float = 0
    losses: float = 0

    @property
    def total(self) -> float:
        return self.wins + self.ties + self.losses

    @property
    def distribution(self) -> tuple[float, float, float]:
        t = self.total
        if t == 0:
            return (0.0, 0.0, 0.0)
        return (self.wins / t, self.ties / t, self.losses / t)

    def _moments(self) -> tuple[float, float]:
        """
        Bayesian mean and variance using pseudocounts (α=1).

        Returns:
            (mean, variance) in the [0, 1] utility space.
        """
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l

        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        mean_sq = (w * W_WEIGHT**2 + t * T_WEIGHT**2 + l * L_WEIGHT**2) / n

        return mean, mean_sq - mean**2

    @property
    def mean_score(self) -> float:
        """Bayesian mean score in [0, 1]."""
        mean, _ = self._moments()
        return mean

    @property
    def std_error(self) -> float:
        """Standard error from pseudocount posterior."""
        _, variance = self._moments()
        n = self.total + 3
        return math.sqrt(max(0, variance / n))

    @property
    def utility(self) -> float:
        """Raw expected value (not normalized), no pseudocounts."""
        if self.total == 0:
            return 0.0
        return (self.wins * W_WEIGHT + self.ties * T_WEIGHT + self.losses * L_WEIGHT) / self.total

    @property
    def certainty(self) -> float:
        """Confidence in estimate, [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.std_error))


# ---------------------------------------------------------------------------
# Decisive evidence test
# ---------------------------------------------------------------------------

def is_decisive(stats: Stats, base_rate: float = 0.5) -> bool:
    """Is this outcome unanimously significant given the base rate?

    Returns True when ALL outcomes are the same type (all wins, all
    losses, or all ties) AND the probability of that happening by
    chance is below 5% (binomial test).

    The base_rate is what we'd expect without this specific pattern —
    typically the position's expected win rate. Higher base rates require
    more unanimous samples to be considered decisive.

    Examples:
        base_rate=0.5 → need 5+ unanimous  (balanced game)
        base_rate=0.7 → need 9+ unanimous  (skewed game)
        base_rate=0.3 → need 3+ unanimous  (rare wins)

    When True, use stats.utility (exact ratio) instead of
    stats.mean_score (Bayesian smoothed).
    """
    if stats.total == 0:
        return False

    w, t, l = stats.wins, stats.ties, stats.losses
    n_types = (w > 0) + (t > 0) + (l > 0)
    if n_types != 1:
        return False

    p = max(min(base_rate, 0.95), 0.5)
    threshold = max(3, math.ceil(math.log(0.05) / math.log(p)))
    return stats.total >= threshold