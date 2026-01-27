"""
Selection module - move selection strategies for training and inference.

Provides the main entry points:
- select_move(): For competitive play (deterministic)
- select_move_for_training(): For training (probabilistic)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import UNEXPLORED_ANCHOR_ID
from wise_explorer.selection import training, inference

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.memory.game_memory import GameMemory


def select_move(
    game: "GameBase",
    memory: "GameMemory",
    is_prune: bool = False,
    random_in_anchor: bool = True,
    debug: bool = False,
) -> np.ndarray:
    """
    Select move for inference (competitive play).
    
    Pure exploitation:
    - Anchor: deterministic best by mean_score
    - Move: random (default, all equivalent) or deterministic best
    
    Args:
        game: Current game state
        memory: GameMemory instance
        is_prune: If True, prefer worst moves (for adversarial play)
        random_in_anchor: If True, random among anchor's moves (they're equivalent).
                        If False, pick best by mean_score.
        debug: If True, show debug visualization
        
    Returns:
        Selected move as numpy array
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves(game, valid_moves)
    
    anchors_with_moves = evaluation.get("anchors_with_moves", {})
    anchor_stats = evaluation.get("anchor_stats", {})
    
    if not anchors_with_moves:
        return random.choice(valid_moves)
    
    pick_best = not is_prune
    
    # Filter to explored moves only
    known_stats = {k: v for k, v in anchor_stats.items() if k != UNEXPLORED_ANCHOR_ID}
    known_moves = {k: v for k, v in anchors_with_moves.items() if k != UNEXPLORED_ANCHOR_ID}
    
    if not known_stats:
        # All moves unexplored - pick randomly
        if UNEXPLORED_ANCHOR_ID in anchors_with_moves:
            return random.choice(anchors_with_moves[UNEXPLORED_ANCHOR_ID])[0]
        return random.choice(valid_moves)
    
    # Deterministic best anchor
    best_anchor_id = inference.best_anchor(known_stats, pick_best)
    
    # Move selection within anchor
    if random_in_anchor:
        selected_move = inference.random_move(known_moves[best_anchor_id])
    else:
        selected_move = inference.best_move(known_moves[best_anchor_id], pick_best)
    
    if debug:
        memory.debug_move_selection(game, valid_moves, selected_move)
    
    return np.asarray(selected_move)


def select_move_for_training(
    game: "GameBase",
    memory: "GameMemory",
    is_prune: bool,
    debug: bool = False,
) -> np.ndarray:
    """
    Select move for training.
    
    Probabilistic weighted by: std_error + promise
    
    - Unknowns (std_error=∞) always prioritized
    - High uncertainty → explored
    - Low uncertainty → score dominates
    
    Args:
        game: Current game state
        memory: GameMemory instance
        is_prune: If True, promise = 1 - mean_score (find bad lines)
                 If False, promise = mean_score (reinforce good lines)
        debug: If True, show debug visualization
        
    Returns:
        Selected move as numpy array
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves(game, valid_moves)
    
    anchors_with_moves = evaluation.get("anchors_with_moves", {})
    anchor_stats = evaluation.get("anchor_stats", {})
    
    if not anchors_with_moves:
        return np.asarray(random.choice(valid_moves))
    
    pick_best = not is_prune
    
    # Probabilistic weighted selection
    selected_anchor = training.select_anchor(anchor_stats, pick_best)
    selected_move = training.select_move(anchors_with_moves[selected_anchor], pick_best)
    
    if debug:
        memory.debug_move_selection(game, valid_moves, selected_move)
    
    return np.asarray(selected_move)


__all__ = [
    "select_move",
    "select_move_for_training",
]