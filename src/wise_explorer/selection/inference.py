"""
Move selection for inference (competitive play).

Inference uses deterministic selection: pick the best anchor by mean_score,
then either random or best move within that anchor.

The key insight is that moves within the same anchor are statistically
equivalent, so random selection among them is (asymptotically) fine.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from wise_explorer.core.types import Stats


def best_anchor(anchor_stats: Dict[int, Stats], pick_best: bool) -> int:
    """
    Deterministic: pick anchor with best/worst mean_score.
    
    Args:
        anchor_stats: Dict mapping anchor IDs to their Stats
        pick_best: If True, maximize score; if False, minimize score
        
    Returns:
        Selected anchor ID
    """
    if pick_best:
        return max(anchor_stats.keys(), key=lambda k: anchor_stats[k].mean_score)
    else:
        return min(anchor_stats.keys(), key=lambda k: anchor_stats[k].mean_score)


def best_move(moves: List[Tuple[np.ndarray, Stats]], pick_best: bool) -> np.ndarray:
    """
    Deterministic: pick move with best/worst mean_score.
    
    Args:
        moves: List of (move, stats) tuples
        pick_best: If True, maximize score; if False, minimize score
        
    Returns:
        Selected move array
    """
    if len(moves) == 1:
        return moves[0][0]
    
    if pick_best:
        return max(moves, key=lambda m: m[1].mean_score)[0]
    else:
        return min(moves, key=lambda m: m[1].mean_score)[0]


def random_move(moves: List[Tuple[np.ndarray, Stats]]) -> np.ndarray:
    """
    Random: all moves in anchor are equivalent.
    
    Since moves are clustered by statistical similarity, any move
    in the anchor should have similar expected outcomes.
    """
    return random.choice(moves)[0]