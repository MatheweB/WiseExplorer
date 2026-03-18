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
# Shared: Bell-aware scoring via pick-best-signal
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


def _collect_predicate_scores(
    game: "GameBase",
    memory: "GameMemory",
    valid_moves: List[np.ndarray],
) -> Dict[tuple, Optional[float]]:
    """Collect predicate-matched scores for all valid moves.

    Checks each destination board against the predicate library. Returns
    the predicate's mean_score if a match is found, None otherwise.
    """
    if not hasattr(memory, 'predicate_library') or memory.predicate_library.count == 0:
        return {}

    current_board = game.get_state().board
    from_board_2d = current_board if current_board.ndim == 2 else current_board.reshape(1, -1)

    scores: Dict[tuple, Optional[float]] = {}
    for move in valid_moves:
        clone = game.deep_clone()
        clone.apply_move(move, validated=True)
        to_board = clone.get_state().board
        to_board_2d = to_board if to_board.ndim == 2 else to_board.reshape(1, -1)

        pred_stats = memory.predicate_library.match(to_board_2d, from_board_2d)
        scores[tuple(move)] = pred_stats.mean_score if pred_stats is not None else None
    return scores


def _rank_signals(
    bell_scores: Dict[tuple, Optional[float]],
    anchor_scores: Dict[tuple, float],
    solo_scores: Dict[tuple, float],
    pred_scores: Optional[Dict[tuple, Optional[float]]] = None,
) -> Tuple[str, ...]:
    """Rank signals by variance — most informative first.

    Compares variance of bell, anchor, solo, and predicate scores across
    all moves. Returns signal names ordered by descending variance.
    """
    bells = [b for b in bell_scores.values() if b is not None]
    anchors = list(anchor_scores.values())
    solos = list(solo_scores.values())

    variances = {
        "bell": np.var(bells) if len(bells) >= 2 else 0.0,
        "anchor": np.var(anchors) if len(anchors) >= 2 else 0.0,
        "solo": np.var(solos) if len(solos) >= 2 else 0.0,
    }

    if pred_scores:
        preds = [p for p in pred_scores.values() if p is not None]
        variances["pred"] = np.var(preds) if len(preds) >= 2 else 0.0

    ranked = sorted(variances, key=variances.get, reverse=True)  # type: ignore
    return tuple(ranked)


def _score_move(
    bell: Optional[float],
    anchor_score: float,
    solo_score: float,
    signal_order: Tuple[str, ...],
    pred_score: Optional[float] = None,
) -> Tuple[float, ...]:
    """Score a move as a tuple based on signal ranking.

    signal_order is ranked by variance (most informative first).
    The move is scored using the most informative signal first,
    with remaining signals as tiebreakers.
    """
    b = bell if bell is not None else anchor_score
    p = pred_score if pred_score is not None else anchor_score
    values = {"bell": b, "anchor": anchor_score, "solo": solo_score, "pred": p}
    return tuple(values[s] for s in signal_order)


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

    Uses pick-best-signal: compares variance of bell vs mean scores
    across all moves and ranks by whichever signal is more informative.
    The other signal serves as tiebreaker.

    - Nim-like (bell varies): ranks by (bell, anchor, solo)
    - TTT-like (bell flat):  ranks by (anchor, bell, solo)

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
    pred_scores = _collect_predicate_scores(game, memory, valid_moves)

    # Collect anchor and solo scores per move for signal comparison
    anchor_scores: Dict[tuple, float] = {}
    solo_scores: Dict[tuple, float] = {}
    for aid, moves in anchors_with_moves.items():
        a_score = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
        for move, stats in moves:
            mk = tuple(move)
            anchor_scores[mk] = a_score
            solo_scores[mk] = stats.mean_score

    signal_order = _rank_signals(bell_scores, anchor_scores, solo_scores, pred_scores)

    # Score each move using the ranked signal order
    move_scored: list[tuple[np.ndarray, Tuple[float, ...]]] = []
    for aid, moves in anchors_with_moves.items():
        a_score = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
        for move, stats in moves:
            mk = tuple(move)
            bell = bell_scores.get(mk)
            pred = pred_scores.get(mk) if pred_scores else None
            key = _score_move(bell, a_score, stats.mean_score, signal_order, pred)
            move_scored.append((move, key))

    if pick_best:
        selected_move = max(move_scored, key=lambda m: m[1])[0]
    else:
        selected_move = min(move_scored, key=lambda m: m[1])[0]

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

    Uses pick-best-signal to find the best anchor, then
    probabilistically explores or exploits within it.

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
    pred_scores = _collect_predicate_scores(game, memory, valid_moves)

    # Collect anchor and solo scores for signal ranking
    anchor_scores: Dict[tuple, float] = {}
    solo_scores: Dict[tuple, float] = {}
    for aid, moves in anchors_with_moves.items():
        a_score = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
        for move, stats in moves:
            mk = tuple(move)
            anchor_scores[mk] = a_score
            solo_scores[mk] = stats.mean_score

    signal_order = _rank_signals(bell_scores, anchor_scores, solo_scores, pred_scores)

    # Find best anchor using ranked signals
    best_anchor_id = _best_anchor_by_signal(
        anchors_with_moves, anchor_stats, bell_scores, signal_order, pick_best,
        pred_scores,
    )

    # Probabilistic: explore or exploit within the best anchor
    exploration_weight = training._exploration_weight(anchor_stats[best_anchor_id], pick_best)

    if random.random() < exploration_weight:
        selected_move = random.choice(valid_moves)
    else:
        selected_move = _select_within_anchor(
            anchors_with_moves[best_anchor_id], bell_scores, signal_order, pick_best,
            pred_scores,
        )

    if debug:
        from wise_explorer.debug.viz import debug_move_selection
        debug_move_selection(memory, game, valid_moves, selected_move)

    return np.asarray(selected_move)


def _select_within_anchor(
    moves: List[Tuple[np.ndarray, Stats]],
    bell_scores: Dict[tuple, Optional[float]],
    signal_order: Tuple[str, ...],
    pick_best: bool,
    pred_scores: Optional[Dict[tuple, Optional[float]]] = None,
) -> np.ndarray:
    """Pick best move within anchor using ranked signals. Random among full ties."""
    scored = []
    for move, stats in moves:
        mk = tuple(move)
        bell = bell_scores.get(mk)
        pred = pred_scores.get(mk) if pred_scores else None
        key = _score_move(bell, stats.mean_score, stats.mean_score, signal_order, pred)
        scored.append((move, key))

    if pick_best:
        best_key = max(scored, key=lambda s: s[1])[1]
        ties = [m for m, k in scored if k == best_key]
    else:
        best_key = min(scored, key=lambda s: s[1])[1]
        ties = [m for m, k in scored if k == best_key]

    return random.choice(ties)


def _best_anchor_by_signal(
    anchors_with_moves: Dict[int, List[Tuple[np.ndarray, Stats]]],
    anchor_stats: Dict[int, Stats],
    bell_scores: Dict[tuple, Optional[float]],
    signal_order: Tuple[str, ...],
    pick_best: bool,
    pred_scores: Optional[Dict[tuple, Optional[float]]] = None,
) -> int:
    """Pick the best anchor using ranked signals."""
    anchor_best: Dict[int, Tuple[float, ...]] = {}

    for aid, moves in anchors_with_moves.items():
        a_score = anchor_stats[aid].mean_score if aid in anchor_stats else 0.5
        best_key = _score_move(None, a_score, a_score, signal_order)
        for move, stats in moves:
            mk = tuple(move)
            bell = bell_scores.get(mk)
            pred = pred_scores.get(mk) if pred_scores else None
            key = _score_move(bell, a_score, stats.mean_score, signal_order, pred)
            if pick_best:
                best_key = max(best_key, key)
            else:
                best_key = min(best_key, key)
        anchor_best[aid] = best_key

    if pick_best:
        return max(anchor_best, key=anchor_best.get)  # type: ignore
    return min(anchor_best, key=anchor_best.get)  # type: ignore


__all__ = [
    "select_move",
    "select_move_for_training",
]
