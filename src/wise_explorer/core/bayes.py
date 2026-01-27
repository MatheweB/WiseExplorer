"""
Bayes factor utilities for anchor clustering.

Uses Dirichlet-Multinomial model to determine if two outcome
distributions are statistically compatible (should be pooled).

This version uses Numba JIT compilation for ~10-50x speedup on the
hot path functions.
"""

from __future__ import annotations

import math
from typing import Tuple

# Try to import numba, fall back to pure Python if not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fake decorator that does nothing
    def njit(func):
        return func
    NUMBA_AVAILABLE = False


# =============================================================================
# Core Math (JIT-compiled if numba available)
# =============================================================================

@njit
def _lgamma(x: float) -> float:
    """Log-gamma function (math.lgamma equivalent for numba)."""
    # Numba supports math.lgamma directly
    return math.lgamma(x)


@njit
def _log_dm_marginal(counts: Tuple[int, int, int], prior: float = 1.0) -> float:
    """
    Log marginal likelihood under Dirichlet-Multinomial model.
    
    P(data | DM) with symmetric Dirichlet prior.
    """
    k = 3  # W, T, L
    n = counts[0] + counts[1] + counts[2]
    
    # log B(counts + prior) - log B(prior)
    # = sum(lgamma(c + prior)) - lgamma(n + k*prior) - [sum(lgamma(prior)) - lgamma(k*prior)]
    
    log_num = 0.0
    for c in counts:
        log_num += _lgamma(c + prior)
    log_num -= _lgamma(n + k * prior)
    
    log_denom = k * _lgamma(prior) - _lgamma(k * prior)
    
    return log_num - log_denom


@njit
def _log_bayes_factor_impl(
    a0: int, a1: int, a2: int,
    b0: int, b1: int, b2: int,
    prior: float = 1.0
) -> float:
    """
    Log Bayes factor for "same distribution" vs "different distributions".
    
    BF > 0 means evidence for same distribution (compatible).
    BF < 0 means evidence for different distributions (incompatible).
    """
    # Combined counts
    c0, c1, c2 = a0 + b0, a1 + b1, a2 + b2
    
    # P(data | same) = P(a + b | single DM)
    log_same = _log_dm_marginal((c0, c1, c2), prior)
    
    # P(data | different) = P(a | DM_a) * P(b | DM_b)
    log_diff = _log_dm_marginal((a0, a1, a2), prior) + _log_dm_marginal((b0, b1, b2), prior)
    
    return log_same - log_diff


@njit
def _compatible_impl(
    a0: int, a1: int, a2: int,
    b0: int, b1: int, b2: int
) -> bool:
    """
    Check if two distributions are statistically compatible.
    
    Uses early-exit heuristics before full Bayes factor.
    """
    na = a0 + a1 + a2
    nb = b0 + b1 + b2
    
    # Empty distributions
    if na == 0 or nb == 0:
        return True  # No evidence against compatibility
    
    # Quick L1 distance check on distributions
    # If obviously different, skip expensive Bayes computation
    da0, da1, da2 = a0 / na, a1 / na, a2 / na
    db0, db1, db2 = b0 / nb, b1 / nb, b2 / nb
    
    l1_dist = abs(da0 - db0) + abs(da1 - db1) + abs(da2 - db2)
    
    # Very different distributions - quick reject
    if l1_dist > 0.6:
        return False
    
    # Very similar distributions - quick accept
    if l1_dist < 0.1 and na > 10 and nb > 10:
        return True
    
    # Full Bayes factor check
    return _log_bayes_factor_impl(a0, a1, a2, b0, b1, b2, 1.0) > 0.0


@njit
def _similarity_impl(
    a0: int, a1: int, a2: int,
    b0: int, b1: int, b2: int
) -> float:
    """
    Similarity score between two distributions.
    
    Returns value in [0, 1] where 1 = identical distributions.
    Uses negative L1 distance transformed to [0, 1].
    """
    na = a0 + a1 + a2
    nb = b0 + b1 + b2
    
    if na == 0 or nb == 0:
        return 0.0
    
    da0, da1, da2 = a0 / na, a1 / na, a2 / na
    db0, db1, db2 = b0 / nb, b1 / nb, b2 / nb
    
    l1_dist = abs(da0 - db0) + abs(da1 - db1) + abs(da2 - db2)
    
    # L1 distance is in [0, 2], convert to similarity in [0, 1]
    return 1.0 - l1_dist / 2.0


# =============================================================================
# Public API (tuple-based wrappers)
# =============================================================================

def log_bayes_factor(a: Tuple[int, int, int], b: Tuple[int, int, int], prior: float = 1.0) -> float:
    """
    Log Bayes factor for "same distribution" vs "different distributions".
    
    BF > 0 means evidence for same distribution (compatible).
    BF < 0 means evidence for different distributions (incompatible).
    """
    return _log_bayes_factor_impl(a[0], a[1], a[2], b[0], b[1], b[2], prior)


def compatible(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> bool:
    """
    Check if two distributions are statistically compatible.
    
    Returns True if Bayes factor favors "same distribution" hypothesis.
    """
    return _compatible_impl(a[0], a[1], a[2], b[0], b[1], b[2])


def similarity(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """
    Similarity score between two distributions.
    
    Returns value in [0, 1] where 1 = identical distributions.
    """
    return _similarity_impl(a[0], a[1], a[2], b[0], b[1], b[2])


# =============================================================================
# Initialization - warm up JIT on first import
# =============================================================================

def _warmup():
    """Pre-compile JIT functions on module load."""
    if NUMBA_AVAILABLE:
        # Call each function once to trigger compilation
        _compatible_impl(10, 5, 5, 10, 5, 5)
        _log_bayes_factor_impl(10, 5, 5, 10, 5, 5, 1.0)
        _similarity_impl(10, 5, 5, 10, 5, 5)

_warmup()