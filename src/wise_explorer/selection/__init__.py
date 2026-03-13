"""
Selection module - move selection strategies for training and inference.

Provides the main entry points:
- select_move(): For competitive play (deterministic)
- select_move_for_training(): For training (probabilistic)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from wise_explorer.core.types import Stats
from wise_explorer.selection import training

if TYPE_CHECKING:
    from wise_explorer.games.game_base import GameBase
    from wise_explorer.memory.game_memory import GameMemory


# ---------------------------------------------------------------------------
# Shared: Bell-aware effective scoring
# ---------------------------------------------------------------------------

def _collect_bell_scores(
    game: "GameBase",
    memory: "GameMemory",
    valid_moves: List[np.ndarray],
) -> Dict[tuple, Optional[float]]:
    """Collect propagated (Bellman) scores for all valid moves. Returns empty dict if unavailable."""
    if memory.is_markov or not hasattr(memory, 'get_propagated_score'):
        return {}

    from wise_explorer.core.hashing import hash_board
    from_hash = hash_board(game.get_state().board)

    scores: Dict[tuple, Optional[float]] = {}
    for move in valid_moves:
        clone = game.deep_clone()
        clone.apply_move(move, validated=True)
        to_hash = hash_board(clone.get_state().board)
        scores[tuple(move)] = memory.get_propagated_score(from_hash, to_hash)
    return scores


def _effective_score(
    bell: Optional[float],
    anchor_score: float,
    pick_best: bool,
) -> float:
    """Conservative combination: min(bell, anchor) for pick_best, max for prune.

    Only trust a move is good if both signals agree (min).
    Only trust a move is bad if both signals agree (max).
    """
    if bell is None:
        return anchor_score
    return min(bell, anchor_score) if pick_best else max(bell, anchor_score)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def select_move(
    game: "GameBase",
    memory: "GameMemory",
    is_prune: bool = False,
    random_in_anchor: bool = True,
    debug: bool = False,
) -> np.ndarray:
    """
    Select move for inference (competitive play).

    Each move's effective score is min(bell, anchor) — conservative:
    - Nim: min(0.98, 0.50)=0.50 still beats min(0.02, 0.50)=0.02
    - TTT: all Bell=0.50, ties broken by anchor then solo
    - Forced losses: Bell-LOW drags effective down, can't be overridden

    Returns:
        Selected move as numpy array
    """
    valid_moves = game.valid_moves()
    if len(valid_moves) == 0:
        raise ValueError("No valid moves")

    pick_best = not is_prune

    evaluation = memory.evaluate_moves(game, valid_moves)
    anchors_with_moves = evaluation.anchors_with_moves
    anchor_stats = evaluation.anchor_stats

    if not anchors_with_moves:
        return np.asarray(random.choice(valid_moves))

    bell_scores = _collect_bell_scores(game, memory, valid_moves)

    # Score each move: (effective, bell, solo) — tiebreak chain
    move_scored: list[tuple[np.ndarray, float, float, float]] = []
    for aid, moves in anchors_with_moves.items():
        a_score = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
        for move, stats in moves:
            bell = bell_scores.get(tuple(move))
            effective = _effective_score(bell, a_score, pick_best)
            solo = stats.mean_score
            move_scored.append((move, effective, bell if bell is not None else a_score, solo))

    # Sort by (effective, bell, solo) — best first
    if pick_best:
        selected_move = max(move_scored, key=lambda m: (m[1], m[2], m[3]))[0]
    else:
        selected_move = min(move_scored, key=lambda m: (m[1], m[2], m[3]))[0]

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected_move)

    return np.asarray(selected_move)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def select_move_for_training(
    game: "GameBase",
    memory: "GameMemory",
    is_prune: bool,
    debug: bool = False,
) -> np.ndarray:
    """
    Select move for training (probabilistic).

    Uses min(bell, anchor) to find the best anchor conservatively,
    then probabilistically explores or exploits within it.

    Returns:
        Selected move as numpy array
    """
    valid_moves = game.valid_moves()
    evaluation = memory.evaluate_moves(game, valid_moves)

    anchors_with_moves = evaluation.anchors_with_moves
    anchor_stats = evaluation.anchor_stats

    if not anchors_with_moves:
        return np.asarray(random.choice(valid_moves))

    pick_best = not is_prune
    bell_scores = _collect_bell_scores(game, memory, valid_moves)

    # Find best anchor using effective scores (bell-aware)
    best_anchor_id = _best_anchor_effective(
        anchors_with_moves, anchor_stats, bell_scores, pick_best,
    )

    # Probabilistic: explore or exploit within the best anchor
    exploration_weight = training._exploration_weight(anchor_stats[best_anchor_id], pick_best)

    if random.random() < exploration_weight:
        selected_move = random.choice(valid_moves)
    else:
        selected_move = _select_within_anchor_bell(
            anchors_with_moves[best_anchor_id], bell_scores, pick_best,
        )

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected_move)

    return np.asarray(selected_move)


def _select_within_anchor_bell(
    moves: List[Tuple[np.ndarray, Stats]],
    bell_scores: Dict[tuple, Optional[float]],
    pick_best: bool,
) -> np.ndarray:
    """Pick best move by (bell, solo) within anchor. Random among full ties."""
    scored = []
    for move, stats in moves:
        bell = bell_scores.get(tuple(move))
        scored.append((move, bell if bell is not None else 0.5, stats.mean_score))

    if pick_best:
        best_key = max(scored, key=lambda s: (s[1], s[2]))[1:]
        ties = [m for m, b, s in scored if (b, s) == best_key]
    else:
        best_key = min(scored, key=lambda s: (s[1], s[2]))[1:]
        ties = [m for m, b, s in scored if (b, s) == best_key]

    return random.choice(ties)


def _best_anchor_effective(
    anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]],
    anchor_stats: Dict[int, Stats],
    bell_scores: Dict[tuple, Optional[float]],
    pick_best: bool,
) -> int:
    """Pick the best anchor using conservative effective scores per move."""
    anchor_effective: Dict[int, float] = {}

    for aid, moves in anchors_with_moves.items():
        a_score = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
        best_in_anchor = a_score
        for move, _stats in moves:
            bell = bell_scores.get(tuple(move))
            eff = _effective_score(bell, a_score, pick_best)
            if pick_best:
                best_in_anchor = max(best_in_anchor, eff)
            else:
                best_in_anchor = min(best_in_anchor, eff)
        anchor_effective[aid] = best_in_anchor

    if pick_best:
        return max(anchor_effective, key=anchor_effective.get)  # type: ignore
    return min(anchor_effective, key=anchor_effective.get)  # type: ignore


__all__ = [
    "select_move",
    "select_move_for_training",
]
