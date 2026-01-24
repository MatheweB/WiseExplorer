"""
Move selection for game AI.

TRAINING: Probabilistic weighted by (std_error + promise)
INFERENCE: Deterministic best anchor, then best or random move

The Formula (training):
    weight = std_error + promise
    
    - std_error is UNCAPPED (infinity for unknowns)
    - promise = mean_score (exploit) or 1-mean_score (prune)
    - Probabilistic selection maintains diversity
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from agent.agent import Agent
from games.game_base import GameBase
from omnicron.manager import GameMemory, Stats, UNEXPLORED_ANCHOR_ID


# Large weight for infinite uncertainty
INF_WEIGHT = 100.0


# ---------------------------------------------------------------------------
# Training Selection (Probabilistic: std_error + promise)
# ---------------------------------------------------------------------------


def _exploration_weight(stats: Stats, pick_best: bool) -> float:
    """weight = std_error + promise (uncapped)"""
    se = stats.std_error
    
    if se == float('inf'):
        return INF_WEIGHT
    
    promise = stats.mean_score if pick_best else (1.0 - stats.mean_score)
    return se + promise


def _select_anchor_for_training(anchor_stats: Dict[int, Stats], pick_best: bool) -> int:
    """Probabilistic weighted selection by uncertainty + promise."""
    if not anchor_stats:
        raise ValueError("No anchors")
    if len(anchor_stats) == 1:
        return next(iter(anchor_stats.keys()))
    
    aids = list(anchor_stats.keys())
    weights = [_exploration_weight(anchor_stats[a], pick_best) for a in aids]
    return random.choices(aids, weights=weights, k=1)[0]


def _select_move_for_training(moves: List[Tuple[np.ndarray, Stats]], pick_best: bool) -> np.ndarray:
    """Probabilistic weighted selection within anchor."""
    if len(moves) == 1:
        return moves[0][0]
    
    weights = [_exploration_weight(stats, pick_best) for _, stats in moves]
    return random.choices(moves, weights=weights, k=1)[0][0]


# ---------------------------------------------------------------------------
# Inference Selection (Deterministic Best)
# ---------------------------------------------------------------------------


def _best_anchor(anchor_stats: Dict[int, Stats], pick_best: bool) -> int:
    """Deterministic: pick anchor with best/worst mean_score."""
    if pick_best:
        return max(anchor_stats.keys(), key=lambda k: anchor_stats[k].mean_score)
    else:
        return min(anchor_stats.keys(), key=lambda k: anchor_stats[k].mean_score)


def _best_move(moves: List[Tuple[np.ndarray, Stats]], pick_best: bool) -> np.ndarray:
    """Deterministic: pick move with best/worst mean_score."""
    if len(moves) == 1:
        return moves[0][0]
    if pick_best:
        return max(moves, key=lambda m: m[1].mean_score)[0]
    else:
        return min(moves, key=lambda m: m[1].mean_score)[0]


def _random_move(moves: List[Tuple[np.ndarray, Stats]]) -> np.ndarray:
    """Random: all moves in anchor are equivalent."""
    return random.choice(moves)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_move(
    game: GameBase,
    memory: GameMemory,
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
        random_in_anchor: If True, random among anchor's moves (they're equivalent).
                         If False, pick best by mean_score.
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves_for_selection(game, valid_moves)
    
    anchors_with_moves = evaluation.get("anchors_with_moves", {})
    anchor_stats = evaluation.get("anchor_stats", {})
    
    if not anchors_with_moves:
        return random.choice(valid_moves)
    
    pick_best = not is_prune
    
    # Filter to explored moves only
    known_stats = {k: v for k, v in anchor_stats.items() if k != UNEXPLORED_ANCHOR_ID}
    known_moves = {k: v for k, v in anchors_with_moves.items() if k != UNEXPLORED_ANCHOR_ID}
    
    if not known_stats:
        if UNEXPLORED_ANCHOR_ID in anchors_with_moves:
            return random.choice(anchors_with_moves[UNEXPLORED_ANCHOR_ID])[0]
        return random.choice(valid_moves)
    
    # Deterministic best anchor
    best_anchor_id = _best_anchor(known_stats, pick_best)
    
    # Move selection within anchor
    if random_in_anchor:
        selected_move = _random_move(known_moves[best_anchor_id])
    else:
        selected_move = _best_move(known_moves[best_anchor_id], pick_best)
    
    if debug:
        memory.debug_move_selection(game, valid_moves, selected_move)
    
    return np.asarray(selected_move)


def select_move_for_training(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
    debug: bool = False,
) -> np.ndarray:
    """
    Select move for training.
    
    Probabilistic weighted by: std_error + promise
    
    - Unknowns (std_error=∞) always prioritized
    - High uncertainty → explored
    - Low uncertainty → score dominates
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves_for_selection(game, valid_moves)
    
    anchors_with_moves = evaluation.get("anchors_with_moves", {})
    anchor_stats = evaluation.get("anchor_stats", {})
    
    if not anchors_with_moves:
        return np.asarray(random.choice(valid_moves))
    
    pick_best = not is_prune
    
    # Probabilistic weighted selection
    selected_anchor = _select_anchor_for_training(anchor_stats, pick_best)
    selected_move = _select_move_for_training(anchors_with_moves[selected_anchor], pick_best)
    
    if debug:
        memory.debug_move_selection(game, valid_moves, selected_move)
    
    return np.asarray(selected_move)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def update_agent(
    agent: Agent,
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
) -> None:
    """Convenience wrapper."""
    agent.core_move = select_move_for_training(game, memory, is_prune)
