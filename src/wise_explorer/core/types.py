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

EPSILON = 0.01   # Tie-breaking margin

SCORE_MIN = min(W_WEIGHT, T_WEIGHT, L_WEIGHT)
SCORE_MAX = max(W_WEIGHT, T_WEIGHT, L_WEIGHT)
SCORE_RANGE = SCORE_MAX - SCORE_MIN

# Standard: Uses the raw weights defined above
WEIGHTS_STANDARD = (W_WEIGHT, T_WEIGHT, L_WEIGHT)

# Loss Averse (Tie-Optimistic):
# Treats Ties as "Near-Max" outcomes to encourage safety/exploration.
# Logic: Take the absolute best possible score (SCORE_MAX) and subtract epsilon.
# The max() check ensures we never inadvertently lower the value of a Tie
# if the Tie was already the best outcome.
_TIE_BOOSTED = max(T_WEIGHT, SCORE_MAX - EPSILON)
WEIGHTS_LOSS_AVERSE = (W_WEIGHT, _TIE_BOOSTED, L_WEIGHT)

OUTCOME_INDEX = {State.WIN: 0, State.TIE: 1, State.LOSS: 2}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Stats(NamedTuple):
    """Outcome counts with Bayesian scoring via pseudocounts."""

    wins: int = 0
    ties: int = 0
    losses: int = 0

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.wins, self.ties, self.losses)

    @property
    def total(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def distribution(self) -> Tuple[float, float, float]:
        t = self.total
        if t == 0:
            return (0.0, 0.0, 0.0)
        return (self.wins / t, self.ties / t, self.losses / t)

    def _moments(self, weights: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Helper to calculate Bayesian mean and variance using pseudocounts (α=1).
        
        Args:
            w_val, t_val, l_val: The utility weights to use for calculation.
            
        Returns: 
            (raw_mean, raw_variance)
        """
        w, t, l = self.wins + 1, self.ties + 1, self.losses + 1
        n = w + t + l
        w_val, t_val, l_val = weights

        # First moment (Mean)
        mean = (w * w_val + t * t_val + l * l_val) / n

        # Second moment (Mean of squares) used for variance calculation
        # Var(X) = E[X^2] - (E[X])^2
        mean_sq = (w * w_val**2 + t * t_val**2 + l * l_val**2) / n

        return mean, mean_sq - mean**2

    @property
    def mean_standard(self) -> float:
        """Normalized ([0, 1]) mean using standard weights."""
        if SCORE_RANGE == 0:
            return 0.0
        mean, _ = self._moments(WEIGHTS_STANDARD)
        return (mean - SCORE_MIN) / SCORE_RANGE

    @property
    def mean_loss_averse(self) -> float:
        """Normalized ([0, 1]) mean using tie-optimistic weights."""
        if SCORE_RANGE == 0:
            return 0.0
        mean, _ = self._moments(WEIGHTS_LOSS_AVERSE)
        return (mean - SCORE_MIN) / SCORE_RANGE

    @property
    def mean_score(self) -> float:
        """Optimistic envelope: max of both strategies."""
        return max(self.mean_standard, self.mean_loss_averse)

    @property
    def std_error(self) -> float:
        """Standard error from pseudocount posterior."""
        weights = WEIGHTS_LOSS_AVERSE if self.mean_loss_averse > self.mean_standard else WEIGHTS_STANDARD
        _, variance = self._moments(weights)

        n = self.total + 3
        raw_se = math.sqrt(max(0, variance / n))

        return raw_se / SCORE_RANGE if SCORE_RANGE > 0 else 0.0

    @property
    def utility(self) -> float:
        """Raw expected value (not normalized)."""
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
            utility = probs[0] * W_WEIGHT + probs[1] * T_WEIGHT + probs[2] * L_WEIGHT
            return (utility - SCORE_MIN) / SCORE_RANGE

        raise ValueError(f"Unknown method: {method}")