"""
Move selection for game AI training and inference.

Two main functions:
    select_move()              - For real play. Uses known moves only.
    select_move_for_training() - For training. Explores unknowns.

Selection modes:
    DETERMINISTIC  - Pick best/worst by score
    PROBABILISTIC  - Weighted random by score
    RANDOM         - Uniform random
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

from agent.agent import Agent
from games.game_base import GameBase
from omnicron.manager import GameMemory


class SelectionMode(Enum):
    DETERMINISTIC = 1
    PROBABILISTIC = 2
    RANDOM = 3


# Synthetic anchor ID for unexplored moves (used by manager.py)
UNEXPLORED_ANCHOR_ID = -999


# ---------------------------------------------------------------------------
# Core Selection Helpers
# ---------------------------------------------------------------------------


def _weighted_select(
    items: List,
    scores: List[float],
    pick_best: bool,
) -> any:
    """
    Select item with probability weighted by score.
    
    If pick_best=True, higher scores get higher probability.
    If pick_best=False, lower scores get higher probability.
    """
    weights = scores if pick_best else [1.0 - s for s in scores]
    weights = [max(0.01, w) for w in weights]  # Ensure all items have some chance
    return random.choices(items, weights=weights, k=1)[0]


def _pick_from_dict(
    scores: Dict[int, float],
    mode: SelectionMode,
    pick_best: bool,
) -> int:
    """Select a key from {key: score} dict based on mode."""
    if mode == SelectionMode.RANDOM:
        return random.choice(list(scores.keys()))
    
    if mode == SelectionMode.DETERMINISTIC:
        fn = max if pick_best else min
        return fn(scores.keys(), key=lambda k: scores[k])
    
    # PROBABILISTIC
    keys = list(scores.keys())
    return _weighted_select(keys, [scores[k] for k in keys], pick_best)


def _pick_move_from_list(
    moves: List[Tuple[np.ndarray, float]],
    mode: SelectionMode,
    pick_best: bool,
) -> np.ndarray:
    """Select a move from [(move, score), ...] list based on mode."""
    if len(moves) == 1:
        return moves[0][0]
    
    if mode == SelectionMode.RANDOM:
        return random.choice(moves)[0]
    
    if mode == SelectionMode.DETERMINISTIC:
        fn = max if pick_best else min
        return fn(moves, key=lambda m: m[1])[0]
    
    # PROBABILISTIC
    return _weighted_select(moves, [m[1] for m in moves], pick_best)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_move(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool = False,
    anchor_mode: SelectionMode = SelectionMode.DETERMINISTIC,
    move_mode: SelectionMode = SelectionMode.DETERMINISTIC,
    debug: bool = False,
) -> np.ndarray:
    """
    Select a move for real play (inference).
    
    Uses KNOWN moves only - a known move with score 0.499 beats an unknown.
    Falls back to random only if no recorded data exists.
    
    Parameters
    ----------
    game : GameBase
        Current game state.
    memory : GameMemory
        Game memory database.
    is_prune : bool
        False = pick best (default), True = pick worst.
    anchor_mode : SelectionMode
        How to select anchor.
    move_mode : SelectionMode
        How to select move within anchor.
    debug : bool
        If True, render debug visualization.
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves_for_selection(game, valid_moves)
    
    anchors_with_moves = evaluation.get("anchors_with_moves", {})
    anchor_scores = evaluation.get("anchor_scores", {})
    pick_best = not is_prune
    
    # Filter to known moves only (exclude synthetic unexplored anchor)
    known_anchors = {k: v for k, v in anchor_scores.items() if k != UNEXPLORED_ANCHOR_ID}
    known_moves = {k: v for k, v in anchors_with_moves.items() if k != UNEXPLORED_ANCHOR_ID}
    
    # Select from known moves if available
    if known_anchors:
        anchor_id = _pick_from_dict(known_anchors, anchor_mode, pick_best)
        move = _pick_move_from_list(known_moves[anchor_id], move_mode, pick_best)
    # Fall back to unknown moves
    elif UNEXPLORED_ANCHOR_ID in anchors_with_moves:
        move = random.choice(anchors_with_moves[UNEXPLORED_ANCHOR_ID])[0]
    # Ultimate fallback
    else:
        move = random.choice(valid_moves)
    
    if debug:
        memory.debug_move_selection(game, valid_moves, move)
    return move


def select_move_for_training(
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
    primary_move: SelectionMode = SelectionMode.RANDOM,
    fallback_anchor: SelectionMode = SelectionMode.RANDOM,
    fallback_move: SelectionMode = SelectionMode.RANDOM,
    debug: bool = False,
) -> np.ndarray:
    """
    Select a move for training with probability-gated exploration.
    
    Includes unknown moves (scored at 0.5) to encourage exploration.
    
    Two-stage selection:
    
    PRIMARY (fires with probability = anchor_score):
        - Anchor: Always deterministic (best/worst)
        - Move: Uses `primary_move` mode
        
    FALLBACK (fires when primary doesn't):
        - Anchor: Uses `fallback_anchor` mode
        - Move: Uses `fallback_move` mode
    
    This naturally explores uncertain positions more:
        - Score 0.9 → 90% primary, 10% fallback
        - Score 0.5 → 50% primary, 50% fallback
        - Score 0.2 → 20% primary, 80% fallback
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves_for_selection(game, valid_moves)
    
    anchors_with_moves = evaluation.get("anchors_with_moves", {})
    anchor_scores = evaluation.get("anchor_scores", {})
    
    if not anchors_with_moves:
        move = random.choice(valid_moves)
        if debug:
            memory.debug_move_selection(game, valid_moves, move)
        return move
    
    pick_best = not is_prune
    
    # PRIMARY: Try best/worst anchor with probability = score
    primary_anchor = _pick_from_dict(anchor_scores, SelectionMode.DETERMINISTIC, pick_best)
    score = anchor_scores[primary_anchor]
    fire_prob = score if pick_best else (1.0 - score)
    
    if random.random() < fire_prob:
        move = _pick_move_from_list(
            anchors_with_moves[primary_anchor], primary_move, pick_best
        )
    else:
        # FALLBACK: Configurable exploration
        anchor_id = _pick_from_dict(anchor_scores, fallback_anchor, pick_best)
        move = _pick_move_from_list(
            anchors_with_moves[anchor_id], fallback_move, pick_best
        )
    
    if debug:
        memory.debug_move_selection(game, valid_moves, move)
    return move


def update_agent(
    agent: Agent,
    game: GameBase,
    memory: GameMemory,
    is_prune: bool,
    primary_move: SelectionMode = SelectionMode.RANDOM,
    fallback_anchor: SelectionMode = SelectionMode.RANDOM,
    fallback_move: SelectionMode = SelectionMode.RANDOM,
) -> None:
    """Convenience wrapper: select training move and assign to agent.core_move."""
    agent.core_move = select_move_for_training(
        game, memory, is_prune, primary_move, fallback_anchor, fallback_move
    )
