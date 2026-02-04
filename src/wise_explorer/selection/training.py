"""
Move selection for training.

Training uses probabilistic selection weighted by uncertainty and promise
to ensure diverse exploration while focusing on promising lines.

The Formula:
    weight = promise
    
    - promise = mean_score (exploit) or 1-mean_score (prune)
    - Probabilistic selection maintains diversity
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from wise_explorer.core.types import Stats

def _exploration_weight(stats: Stats, pick_best: bool) -> float:
    """
    Calculate exploration weight for a move/anchor.
    
    The intuition is the higher the score of the move we're looking at, the less we need to
    try it again (because we're confident about its outcome). So we explore.
    The same intuition applies for the pruning phase (bad outcome = confident, so explore).
    
    Args:
        stats: Move statistics
        pick_best:  If True, promise = mean_score (exploit mode)
                    If False, promise = 1 - mean_score (prune mode)
    """
    promise = stats.mean_score if pick_best else (1.0 - stats.mean_score)
    return promise


def select_anchor_deterministic(anchor_stats: Dict[int, Stats], pick_best: bool) -> int:
    """Deterministic selection of best/worst anchor by mean_score."""
    if not anchor_stats:
        raise ValueError("No anchors provided")

    if pick_best:
        return max(anchor_stats.keys(), key=lambda a: anchor_stats[a].mean_score)
    else:
        return min(anchor_stats.keys(), key=lambda a: anchor_stats[a].mean_score)

def select_move_random(moves: List[Tuple[np.ndarray, Stats]], pick_best: bool) -> np.ndarray:
    """
    Random selection within anchor.
    
    All moves in an anchor are statistically equivalent.
    
    Args:
        moves: List of (move, stats) tuples
        pick_best: If True, favor high scores; if False, favor low scores
        
    Returns:
        Selected move array
    """
    if len(moves) == 1:
        return moves[0][0]
    
    return random.choice(moves)[0]
