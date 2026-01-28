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
# ║  Symmetric:              W=1.0, T=0.0, L=-1.0  → pure win/loss              ║
# ║  Win-focused:            W=1.0, T=0.0, L=-0.5  → losses hurt less           ║
# ║  Tie-positive:           W=1.0, T=0.8, L=-1.0  → ties nearly as good        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

W_WEIGHT = 1.0   # Utility for a WIN
T_WEIGHT = 0.0   # Utility for a TIE
L_WEIGHT = -1.0  # Utility for a LOSS

# Tie-breaking epsilon 
# Prefers a raw MAX score over a boosted TIE, but keeps them close.
EPSILON = 0.01

# Derived constants (calculated dynamically for robustness)
SCORE_MIN = min(W_WEIGHT, T_WEIGHT, L_WEIGHT)
SCORE_MAX = max(W_WEIGHT, T_WEIGHT, L_WEIGHT)
SCORE_RANGE = SCORE_MAX - SCORE_MIN

# ─── Strategy Configurations ──────────────────────────────────────────────────

# Standard: Uses the raw weights defined above
WEIGHTS_STANDARD = (W_WEIGHT, T_WEIGHT, L_WEIGHT)

# Loss Averse (Tie-Optimistic):
# Treats Ties as "Near-Max" outcomes to encourage safety/exploration.
# Logic: Take the absolute best possible score (SCORE_MAX) and subtract epsilon.
# The max() check ensures we never inadvertently lower the value of a Tie
# if the Tie was already the best outcome.
_TIE_BOOSTED = max(T_WEIGHT, SCORE_MAX - EPSILON)

WEIGHTS_LOSS_AVERSE = (W_WEIGHT, _TIE_BOOSTED, L_WEIGHT)

# ──────────────────────────────────────────────────────────────────────────────

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

    def _calculate_moments(self, w_val: float, t_val: float, l_val: float) -> Tuple[float, float]:
        """
        Helper to calculate Bayesian mean and variance using pseudocounts (α=1).
        
        Args:
            w_val, t_val, l_val: The utility weights to use for calculation.
            
        Returns: 
            (raw_mean, raw_variance)
        """
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l

        # First moment (Mean)
        mean = (w * w_val + t * t_val + l * l_val) / n

        # Second moment (Mean of squares) used for variance calculation
        # Var(X) = E[X^2] - (E[X])^2
        mean_sq = (w * (w_val**2) + t * (t_val**2) + l * (l_val**2)) / n
        variance = mean_sq - (mean**2)

        return mean, variance

    @property
    def mean_standard(self) -> float:
        """Mean utility normalized to [0, 1] using standard weights."""
        if SCORE_RANGE == 0: 
            return 0.0
        mean, _ = self._calculate_moments(*WEIGHTS_STANDARD)
        return (mean - SCORE_MIN) / SCORE_RANGE

    @property
    def mean_loss_averse(self) -> float:
        """Mean utility normalized to [0, 1] using tie-optimistic weights."""
        if SCORE_RANGE == 0: 
            return 0.0
        mean, _ = self._calculate_moments(*WEIGHTS_LOSS_AVERSE)
        return (mean - SCORE_MIN) / SCORE_RANGE

    @property
    def mean_score(self) -> float:
        """Optimistic envelope: returns the higher of the two mean strategies."""
        return max(self.mean_standard, self.mean_loss_averse)

    @property
    def std_error(self) -> float:
        """Standard error (UNCAPPED). Returns inf for insufficient data."""
        if self.total <= 1:
            return float('inf')

        # Select weights based on which mean strategy is currently dominant.
        # This ensures the error bars match the mean we are actually using.
        if self.mean_standard > self.mean_loss_averse:
            weights = WEIGHTS_STANDARD
        else:
            weights = WEIGHTS_LOSS_AVERSE

        _, variance = self._calculate_moments(*weights)

        # n includes pseudocounts (total + 3)
        n = self.total + 3
        raw_se = math.sqrt(max(0, variance / n))
        
        return raw_se / SCORE_RANGE if SCORE_RANGE > 0 else 0.0

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
        if SCORE_RANGE == 0: 
            return 0.0
            
        mean, variance = self._calculate_moments(*WEIGHTS_STANDARD)
        n = self.total + 3
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
