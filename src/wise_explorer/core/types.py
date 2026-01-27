"""
Core types, constants, and data structures.

This module contains the fundamental types used throughout the game AI system:
- Stats: Outcome counts with scoring properties
- Outcome weights and derived constants
- Common data classes
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple, Tuple

import numpy as np

from wise_explorer.agent.agent import State


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURABLE OUTCOME WEIGHTS                             ║
# ║                                                                             ║
# ║  Modify these to experiment with different scoring strategies:              ║
# ║                                                                             ║
# ║  Loss-averse:            W=1.0, T=0.5, L=-1.5  → ties are half-wins         ║
# ║  Symmetric:    W=1.0, T=0.0, L=-1.0  → pure win/loss                        ║
# ║  Win-focused:            W=1.0, T=0.0, L=-0.5  → losses hurt less           ║
# ║  Tie-positive:           W=1.0, T=0.8, L=-1.0  → ties nearly as good        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

W_WEIGHT = 1  # Utility for a WIN
T_WEIGHT = 0    # Utility for a TIE
L_WEIGHT = -1   # Utility for a LOSS

# Derived constants (calculated from weights above - don't modify directly)
SCORE_MIN = min(W_WEIGHT, T_WEIGHT, L_WEIGHT)
SCORE_MAX = max(W_WEIGHT, T_WEIGHT, L_WEIGHT)
SCORE_RANGE = SCORE_MAX - SCORE_MIN

# Outcome indexing for array operations
OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.LOSS: 2}

# Sentinel value for unexplored moves
UNEXPLORED_ANCHOR_ID = -999


class Stats(NamedTuple):
    """Outcome counts with derived scoring properties."""

    wins: int = 0
    ties: int = 0
    losses: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def distribution(self) -> Tuple[float, float, float]:
        t = self.total
        if t == 0:
            return (0.0, 0.0, 0.0)
        return (self.wins / t, self.ties / t, self.losses / t)

    @property
    def mean_score(self) -> float:
        """Mean utility normalized to [0, 1]. Uses Bayesian pseudocounts (α=1)."""
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        return (mean - SCORE_MIN) / SCORE_RANGE

    @property
    def std_error(self) -> float:
        """Standard error (UNCAPPED). Returns inf for insufficient data."""
        if self.total <= 1:
            return float('inf')
        
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        mean_sq = (w * W_WEIGHT**2 + t * T_WEIGHT**2 + l * L_WEIGHT**2) / n
        variance = mean_sq - mean**2
        
        raw_se = math.sqrt(max(0, variance / n))
        return raw_se / SCORE_RANGE

    def sample_score(self, method: str = 'gaussian') -> float:
        """(Unused) Thompson sampling from posterior. Returns [0, 1]."""
        if method == 'gaussian':
            mean, se = self.mean_score, self.std_error
            if se == float('inf'):
                return random.random()
            return max(0.0, min(1.0, random.gauss(mean, se)))
        
        elif method == 'dirichlet':
            alpha = [self.wins + 1, self.ties + 1, self.losses + 1]
            probs = np.random.dirichlet(alpha)
            utility = probs[0] * W_WEIGHT + probs[1] * T_WEIGHT + probs[2] * L_WEIGHT
            return (utility - SCORE_MIN) / SCORE_RANGE
        
        raise ValueError(f"Unknown method: {method}")

    @property
    def optimistic_score(self) -> float:
        """
        UCB score normalized to [0, 1].

        Note: Unused. We use mean_score() + std_error() to separate concerns.
        This achieves the same but with normalization.
        """
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        mean = (w * W_WEIGHT + t * T_WEIGHT + l * L_WEIGHT) / n
        mean_sq = (w * W_WEIGHT**2 + t * T_WEIGHT**2 + l * L_WEIGHT**2) / n
        variance = mean_sq - mean**2
        se = math.sqrt(max(0, variance / n))
        ucb = mean + se
        return (ucb - SCORE_MIN) / SCORE_RANGE

    @property
    def utility(self) -> float:
        """Raw expected value (not normalized)."""
        if self.total == 0:
            return 0.0
        return (self.wins * W_WEIGHT + self.ties * T_WEIGHT + self.losses * L_WEIGHT) / self.total

    @property
    def certainty(self) -> float:
        """Confidence in estimate, capped to [0, 1]."""
        if self.total <= 1:
            return 0.0
        se = self.std_error
        return 0.0 if se == float('inf') else max(0.0, min(1.0, 1.0 - se))