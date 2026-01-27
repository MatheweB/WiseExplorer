"""
Move selection for training.

Training uses probabilistic selection weighted by uncertainty and promise
to ensure diverse exploration while focusing on promising lines.

The Formula:
    weight = std_error + promise
    
    - std_error is UNCAPPED (infinity for unknowns → always explored)
    - promise = mean_score (exploit) or 1-mean_score (prune)
    - Probabilistic selection maintains diversity
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from wise_explorer.core.types import Stats


# Large weight for infinite uncertainty (ensures unknowns are explored)
INF_WEIGHT = 100.0


def _exploration_weight(stats: Stats, pick_best: bool) -> float:
    """
    Calculate exploration weight for a move/anchor.
    
    weight = std_error + promise
    
    Args:
        stats: Move statistics
        pick_best: If True, promise = mean_score (exploit mode)
                  If False, promise = 1 - mean_score (prune mode)
    """
    se = stats.std_error
    
    if se == float('inf'):
        return INF_WEIGHT
    
    promise = stats.mean_score if pick_best else (1.0 - stats.mean_score)
    return se + promise


def select_anchor(anchor_stats: Dict[int, Stats], pick_best: bool) -> int:
    """
    Probabilistic weighted selection of anchor.
    
    Weights combine uncertainty (explore) with promise (exploit/prune).
    
    Args:
        anchor_stats: Dict mapping anchor IDs to their Stats
        pick_best: If True, favor high scores; if False, favor low scores
        
    Returns:
        Selected anchor ID
    """
    if not anchor_stats:
        raise ValueError("No anchors provided")
    if len(anchor_stats) == 1:
        return next(iter(anchor_stats.keys()))
    
    aids = list(anchor_stats.keys())
    weights = [_exploration_weight(anchor_stats[a], pick_best) for a in aids]
    return random.choices(aids, weights=weights, k=1)[0]


def select_move(moves: List[Tuple[np.ndarray, Stats]], pick_best: bool) -> np.ndarray:
    """
    Probabilistic weighted selection within anchor.
    
    All moves in an anchor are statistically equivalent, but we still
    use weighted selection for diversity.
    
    Args:
        moves: List of (move, stats) tuples
        pick_best: If True, favor high scores; if False, favor low scores
        
    Returns:
        Selected move array
    """
    if len(moves) == 1:
        return moves[0][0]
    
    weights = [_exploration_weight(stats, pick_best) for _, stats in moves]
    return random.choices(moves, weights=weights, k=1)[0][0]