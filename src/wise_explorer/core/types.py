"""
This module contains the fundamental types used throughout the game AI system:
- Stats: Outcome counts with scoring properties
- Outcome weights and derived constants
- Common data classes
"""

from __future__ import annotations

import math
from typing import NamedTuple, Tuple

import numpy as np

from wise_explorer.agent.agent import State

Counts = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Outcome Weights — maps [LOSS, TIE, WIN] → [0, 0.5, 1]
# ---------------------------------------------------------------------------

L_WEIGHT = 0.0   # Utility for a LOSS
T_WEIGHT = 0.5   # Utility for a TIE
W_WEIGHT = 1.0   # Utility for a WIN

WEIGHTS = (W_WEIGHT, T_WEIGHT, L_WEIGHT)

SCORE_MIN = L_WEIGHT   # 0.0
SCORE_MAX = W_WEIGHT   # 1.0
SCORE_RANGE = SCORE_MAX - SCORE_MIN  # 1.0

OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.LOSS: 2}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats(NamedTuple):
    """Outcome counts with Bayesian scoring via pseudocounts."""

    wins: float = 0
    ties: float = 0
    losses: float = 0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.wins, self.ties, self.losses)

    @property
    def total(self) -> float:
        return self.wins + self.ties + self.losses

    @property
    def distribution(self) -> Tuple[float, float, float]:
        t = self.total
        if t == 0:
            return (0.0, 0.0, 0.0)
        return (self.wins / t, self.ties / t, self.losses / t)

    def _moments(self) -> Tuple[float, float]:
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

    def sample_score(self, method: str = 'dirichlet') -> float:
        """Thompson sampling from posterior."""
        if method == 'dirichlet':
            alpha = [self.wins + 1, self.ties + 1, self.losses + 1]
            probs = np.random.dirichlet(alpha)
            return probs[0] * W_WEIGHT + probs[1] * T_WEIGHT + probs[2] * L_WEIGHT

        raise ValueError(f"Unknown method: {method}")